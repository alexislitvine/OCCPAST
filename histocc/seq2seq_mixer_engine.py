import os
import time
from collections import Counter
from dataclasses import dataclass

import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from torch import nn
from sklearn.metrics import accuracy_score

from .formatter import BOS_IDX, EOS_IDX, PAD_IDX
from .utils import (
    create_mask,
    Averager,
    order_invariant_accuracy,
    update_summary,
)
from .model_assets import Seq2SeqMixerOccCANINE
from .loss import LossMixer
from .utils.decoder import mixer_greedy_decode


def _save_model_checkpoint(
        model: Seq2SeqMixerOccCANINE,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_step: int,
        save_dir: str,
        dataset_map_code_label: dict,
        ) -> None:
    """Helper function to save model checkpoint.
    
    Args:
        model: The model to save (will be unwrapped if DDP)
        optimizer: The optimizer state to save
        scheduler: The scheduler state to save
        current_step: Current training step
        save_dir: Directory to save checkpoints
        dataset_map_code_label: Dataset label mapping
    """
    # Unwrap DDP model if needed
    model_to_save = getattr(model, 'module', model)
    
    states = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': current_step,
        'key': dataset_map_code_label,
    }
    torch.save(states, os.path.join(save_dir, f'{current_step}.bin'))
    torch.save(states, os.path.join(save_dir, 'last.bin'))


def train_one_epoch(
        model: Seq2SeqMixerOccCANINE,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_step: int,
        epoch: int = 0,
        log_interval: int = 100,
        eval_interval: int | None = None,
        save_interval: int | None = None,
        save_each_epoch: bool = False,
        save_dir: str | None = None,
        data_loader_eval: torch.utils.data.DataLoader | None = None,
        log_wandb: bool = False,
        distributed: bool = False,
        is_main_process: bool = True,
        scaler: GradScaler | None = None,
        ) -> int:
    model = model.train()

    last_step = len(data_loader) - 1
    losses = Averager()
    batch_time = Averager()
    batch_time_data = Averager()
    samples_per_sec = Averager()
    
    # Check GPU availability once
    has_cuda = torch.cuda.is_available()

    # Need to initialize first "end time", as this is
    # calculated at bottom of batch loop
    end = time.time()
    
    # Use tqdm progress bar only on rank 0
    iterator = tqdm(data_loader, disable=not is_main_process, ncols=100, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(iterator):
        current_step += 1

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets_seq2seq = batch['targets_seq2seq'].to(device, non_blocking=True)
        targets_linear = batch['targets_linear'].to(device, non_blocking=True)

        batch_time_data.update(time.time() - end)

        # Prepare target as input for seq2seq model
        target_seq2seq_input = targets_seq2seq[:, :-1]
        target_mask, target_padding_mask = create_mask(target_seq2seq_input, PAD_IDX, device)

        # Forward pass with optional AMP
        if scaler is not None:
            with autocast('cuda'):
                out_seq2seq, out_linear = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target=target_seq2seq_input,
                    target_mask=target_mask,
                    target_padding_mask=target_padding_mask,
                )

                loss = loss_fn(
                    out_seq2seq=out_seq2seq,
                    out_linear=out_linear,
                    target_seq2seq=targets_seq2seq,
                    target_linear=targets_linear,
                    )
        else:
            out_seq2seq, out_linear = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target=target_seq2seq_input,
                target_mask=target_mask,
                target_padding_mask=target_padding_mask,
            )

            loss = loss_fn(
                out_seq2seq=out_seq2seq,
                out_linear=out_linear,
                target_seq2seq=targets_seq2seq,
                target_linear=targets_linear,
                )
        losses.update(loss.item(), out_seq2seq.size(0))

        # Backward pass & step with optional AMP
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()

        elapsed = time.time() - end
        batch_time.update(elapsed)
        samples_per_sec.update(out_seq2seq.size(0) / elapsed)

        if is_main_process and (batch_idx % log_interval == 0 or batch_idx == last_step):
            # Calculate ETA
            batches_remaining = len(data_loader) - (batch_idx + 1)
            eta_seconds = batches_remaining * batch_time.avg
            eta_str = f"{int(eta_seconds // 60)}m{int(eta_seconds % 60):02d}s"
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]
            
            tqdm.write(f'[Epoch {epoch}] Batch {batch_idx + 1}/{len(data_loader)} | '
                       f'Loss: {losses.avg:.6f} | '
                       f'LR: {current_lr:.2e} | '
                       f'Batch time: {batch_time.avg:.2f}s (data: {batch_time_data.avg:.2f}s) | '
                       f'Samples/sec: {samples_per_sec.avg:.1f} | '
                       f'ETA: {eta_str}')
            
            # Print GPU memory stats if using CUDA
            if has_cuda:
                tqdm.write(f'  GPU Memory - Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB | '
                           f'Reserved: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB')

        if save_interval is not None and current_step % save_interval == 0 and is_main_process and not save_each_epoch:
            _save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_step=current_step,
                save_dir=save_dir,
                dataset_map_code_label=data_loader.dataset.map_code_label,
            )

        if eval_interval is not None and current_step % eval_interval == 0 and is_main_process:
            tqdm.write('\n' + '='*80)
            tqdm.write('Starting evaluation pass...')
            eval_loss, eval_loss_linear, eval_loss_seq2seq, eval_seq_acc, eval_token_acc, eval_flat_acc = evaluate(
                model=model,
                data_loader=data_loader_eval,
                loss_fn=loss_fn,
                device=device,
            )
            model.train()
            
            # Print evaluation summary
            tqdm.write('='*80)
            tqdm.write(f'EVALUATION RESULTS (Step {current_step})')
            tqdm.write('='*80)
            tqdm.write(f'Validation Loss     : {eval_loss:.6f} (Linear: {eval_loss_linear:.6f}, Seq2Seq: {eval_loss_seq2seq:.6f})')
            tqdm.write(f'Training Loss       : {losses.avg:.6f}')
            tqdm.write(f'Sequence Accuracy   : {eval_seq_acc:.2f}%')
            tqdm.write(f'Token Accuracy      : {eval_token_acc:.2f}%')
            tqdm.write(f'Flat Accuracy       : {eval_flat_acc:.2f}%')
            tqdm.write(f'Learning Rate       : {scheduler.get_last_lr()[0]:.2e}')
            tqdm.write('='*80 + '\n')

            update_summary(
                current_step,
                metrics={
                    'batch_time': batch_time.avg,
                    'batch_time_data': batch_time_data.avg,
                    'train_loss': losses.avg,
                    'val_loss': eval_loss,
                    'val_loss_linear': eval_loss_linear,
                    'val_loss_seq2seq': eval_loss_seq2seq,
                    'seq_acc': eval_seq_acc,
                    'token_acc': eval_token_acc,
                    'flat_acc': eval_flat_acc,
                    'lr': scheduler.get_last_lr()[0],
                },
                filename=os.path.join(save_dir, 'logs.csv'),
                log_wandb=log_wandb,
            )

        end = time.time()

    return current_step


@torch.no_grad
def evaluate(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        log_interval: int = 100,
        ):
    model = model.eval()

    losses = Averager()
    losses_linear = Averager()
    losses_seq2seq = Averager()

    token_accs = Averager()
    seq_accs = Averager()
    flat_accs = Averager()

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets_seq2seq = batch['targets_seq2seq'].to(device, non_blocking=True)
        targets_linear = batch['targets_linear'].to(device, non_blocking=True)

        # Prepare target as input for seq2seq model
        target_seq2seq_input = targets_seq2seq[:, :-1]
        target_mask, target_padding_mask = create_mask(target_seq2seq_input, PAD_IDX, device)

        # Forward pass
        out_seq2seq, out_linear = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target_seq2seq_input,
            target_mask=target_mask,
            target_padding_mask=target_padding_mask,
        )

        loss = loss_fn(
            out_seq2seq=out_seq2seq,
            out_linear=out_linear,
            target_seq2seq=targets_seq2seq,
            target_linear=targets_linear,
            )
        loss_linear = loss_fn.loss_fn_linear(out_linear, targets_linear)
        loss_seq2seq = loss_fn.loss_fn_seq2seq(out_seq2seq, targets_seq2seq)

        losses.update(loss.item(), out_seq2seq.size(0))
        losses_linear.update(loss_linear.item(), out_seq2seq.size(0))
        losses_seq2seq.update(loss_seq2seq.item(), out_seq2seq.size(0))

        seq_acc, token_acc = order_invariant_accuracy(
            output=out_seq2seq,
            target=targets_seq2seq[:, 1:],
            pad_idx=PAD_IDX,
            nb_blocks=loss_fn.loss_fn_seq2seq.nb_blocks,
            block_size=loss_fn.loss_fn_seq2seq.block_size,
        )
        seq_accs.update(seq_acc.item(), out_seq2seq.size(0))
        token_accs.update(token_acc.item(), out_seq2seq.size(0))

        # Linear decoder accuracy
        preds_linear = torch.sigmoid(out_linear) > 0.5
        preds_linear = preds_linear.float().cpu()

        acc_flat = accuracy_score(preds_linear, targets_linear.cpu())
        flat_accs.update(acc_flat, preds_linear.size(0))

        if batch_idx % log_interval == 0:
            tqdm.write(f'  Eval Batch {batch_idx + 1}/{len(data_loader)} | '
                       f'Seq Acc: {seq_accs.avg:.2f}% | '
                       f'Token Acc: {token_accs.avg:.2f}% | '
                       f'Flat Acc: {flat_accs.avg:.2f}% | '
                       f'Val Loss: {losses.avg:.6f}')

    _run_pst2_eval_probe(
        model=model,
        data_loader=data_loader,
        device=device,
        sample_size=200,
        seed=42,
    )

    return losses.avg, losses_linear.avg, losses_seq2seq.avg, seq_accs.avg, token_accs.avg, flat_accs.avg


@dataclass
class _PST2ProbeRow:
    index: int
    occ1: str
    pst2_1: str
    pst2_2: str
    gold2_norm: str
    gold2_in_key: bool
    pred_block1_tokens: list[int]
    pred_block2_tokens: list[int]
    pred_block1_raw: str
    pred_block2_raw: str
    pred_block1_norm: str
    pred_block2_norm: str
    pred_block1_in_key: bool
    pred_block2_in_key: bool
    formatted_pred: str
    split_pred: list[str] | str
    block2_nonpad: bool


def _pst2_value_present(value: str | None) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return False
    value = str(value)
    return value not in {'', ' ', '?'}


def _split_str_s2s(pred: str, sep_value: str) -> list[str] | str:
    if sep_value and sep_value in pred:
        return pred.split(sep_value)
    return pred


def _normalize_code_for_lookup(code: str, inv_key: dict, use_within_block_sep: bool) -> str:
    if not use_within_block_sep:
        return code
    if code in inv_key:
        return code
    parts = code.split(',')
    while len(parts) > 1 and parts[-1] == '0':
        parts = parts[:-1]
        normalized = ','.join(parts)
        if normalized in inv_key:
            return normalized
    return code


def _decode_block_string(formatter, block_tokens: list[int], block_index: int) -> str:
    seq_len = formatter.max_seq_len
    block_size = formatter.block_size
    start = 1 + block_index * block_size
    end = start + block_size
    seq = [PAD_IDX] * seq_len
    seq[0] = BOS_IDX
    seq[-1] = EOS_IDX
    seq[start:end] = block_tokens
    return formatter.clean_pred(torch.tensor(seq).numpy())


def _run_pst2_eval_probe(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        sample_size: int = 200,
        seed: int = 42,
) -> None:
    dataset = data_loader.dataset
    formatter = dataset.formatter
    if not hasattr(dataset, 'frame'):
        return
    if 'pst2_2' not in dataset.frame.columns:
        return
    if dataset.map_code_label is None:
        print('PST2 eval probe skipped: dataset has no map_code_label.')
        return

    has_second = dataset.frame['pst2_2'].apply(_pst2_value_present).to_numpy()
    eligible_positions = [idx for idx, flag in enumerate(has_second) if flag]
    if not eligible_positions:
        print('PST2 eval probe: no rows with pst2_2 present in eval dataset.')
        return

    rng = torch.Generator().manual_seed(seed)
    sample_size = min(sample_size, len(eligible_positions))
    sample_tensor = torch.randperm(len(eligible_positions), generator=rng)[:sample_size]
    sample_indices = [eligible_positions[i] for i in sample_tensor.tolist()]

    inv_key = dataset.map_code_label
    use_within_block_sep = bool(getattr(formatter, 'within_block_sep', None))

    model_to_decode = model.module if hasattr(model, 'module') else model
    model_to_decode.eval()
    block2_nonpad_count = 0
    norm2_in_key_count = 0
    format_contains_sep_value_count = 0
    split_returns_2_count = 0
    norm2_miss_counter = Counter()
    gold2_in_key_count = 0
    gold2_miss_counter = Counter()

    examples_a: list[_PST2ProbeRow] = []
    examples_b: list[_PST2ProbeRow] = []
    examples_c: list[_PST2ProbeRow] = []

    print('\n' + '=' * 80)
    print('PST2 EVAL PROBE (deterministic sample)')
    print(f'  sample_size={sample_size} seed={seed}')
    print(f'  PAD_IDX={PAD_IDX} block_size={formatter.block_size} max_num_codes={formatter.max_num_codes}')

    batch_size = 32
    for offset in range(0, len(sample_indices), batch_size):
        batch_indices = sample_indices[offset:offset + batch_size]
        batch_items = [dataset[idx] for idx in batch_indices]
        input_ids = torch.stack([item['input_ids'] for item in batch_items]).to(device, non_blocking=True)
        attention_mask = torch.stack([item['attention_mask'] for item in batch_items]).to(device, non_blocking=True)

        outputs = mixer_greedy_decode(
            model=model_to_decode,
            descr=input_ids,
            input_attention_mask=attention_mask,
            device=device,
            max_len=formatter.max_seq_len,
            start_symbol=BOS_IDX,
        )
        preds_seq = outputs[0].cpu().numpy()

        for row_pos, dataset_idx in enumerate(batch_indices):
            record = dataset.frame.iloc[dataset_idx]
            raw_seq = preds_seq[row_pos].tolist()
            block1_tokens = raw_seq[1:1 + formatter.block_size]
            block2_tokens = raw_seq[1 + formatter.block_size:1 + 2 * formatter.block_size]
            block2_nonpad = any(tok != PAD_IDX for tok in block2_tokens)

            pred_block1_raw = _decode_block_string(formatter, block1_tokens, 0)
            pred_block2_raw = _decode_block_string(formatter, block2_tokens, 1)
            pred_block1_norm = _normalize_code_for_lookup(pred_block1_raw, inv_key, use_within_block_sep)
            pred_block2_norm = _normalize_code_for_lookup(pred_block2_raw, inv_key, use_within_block_sep)
            pred_block1_in_key = pred_block1_norm in inv_key
            pred_block2_in_key = pred_block2_norm in inv_key
            gold2_raw = str(record['pst2_2'])
            gold2_norm = _normalize_code_for_lookup(gold2_raw, inv_key, use_within_block_sep)
            gold2_in_key = gold2_norm in inv_key

            formatted_pred = formatter.clean_pred(torch.tensor(raw_seq).numpy())
            split_pred = _split_str_s2s(formatted_pred, formatter.sep_value)
            split_pred_list = split_pred if isinstance(split_pred, list) else [split_pred]

            if block2_nonpad:
                block2_nonpad_count += 1
            if pred_block2_in_key:
                norm2_in_key_count += 1
            if gold2_in_key:
                gold2_in_key_count += 1
            if formatter.sep_value and formatter.sep_value in formatted_pred:
                format_contains_sep_value_count += 1
            if len(split_pred_list) == 2:
                split_returns_2_count += 1
            if block2_nonpad and not pred_block2_in_key:
                norm2_miss_counter[pred_block2_norm] += 1
            if not gold2_in_key:
                gold2_miss_counter[gold2_norm] += 1

            row = _PST2ProbeRow(
                index=int(dataset_idx),
                occ1=str(record['occ1']),
                pst2_1=str(record['pst2_1']),
                pst2_2=str(record['pst2_2']),
                gold2_norm=gold2_norm,
                gold2_in_key=gold2_in_key,
                pred_block1_tokens=block1_tokens,
                pred_block2_tokens=block2_tokens,
                pred_block1_raw=pred_block1_raw,
                pred_block2_raw=pred_block2_raw,
                pred_block1_norm=pred_block1_norm,
                pred_block2_norm=pred_block2_norm,
                pred_block1_in_key=pred_block1_in_key,
                pred_block2_in_key=pred_block2_in_key,
                formatted_pred=formatted_pred,
                split_pred=split_pred,
                block2_nonpad=block2_nonpad,
            )

            if block2_nonpad and not pred_block2_in_key and len(examples_a) < 10:
                examples_a.append(row)
            if block2_nonpad and pred_block2_in_key and len(split_pred_list) == 1 and len(examples_b) < 10:
                examples_b.append(row)
            if not block2_nonpad and len(examples_c) < 10:
                examples_c.append(row)

            if offset == 0 and row_pos == 0:
                print(f'  first_pred_tokens_head={raw_seq[:5]} tail={raw_seq[-5:]}')

    total = float(sample_size)
    print('\nSummary counters:')
    print(f'  % pred_block2_nonpad: {block2_nonpad_count / total:.2%}')
    print(f'  % norm2_in_key: {norm2_in_key_count / total:.2%}')
    print(f'  % gold2_in_key: {gold2_in_key_count / total:.2%}')
    print(f'  % format_contains_sep_value: {format_contains_sep_value_count / total:.2%}')
    print(f'  % split_returns_2: {split_returns_2_count / total:.2%}')

    print('\nTop-20 normalized block-2 strings missing from key:')
    for code, count in norm2_miss_counter.most_common(20):
        print(f'  {code!r}: {count}')
    if gold2_miss_counter:
        print('\nTop-20 gold pst2_2 strings missing from key:')
        for code, count in gold2_miss_counter.most_common(20):
            print(f'  {code!r}: {count}')

    def _print_examples(label: str, rows: list[_PST2ProbeRow]) -> None:
        print(f'\nExamples ({label}):')
        if not rows:
            print('  (none)')
            return
        for row in rows:
            print(f'  row_index={row.index}')
            print(f'    occ1={row.occ1!r}')
            print(f'    gold pst2_1={row.pst2_1!r} pst2_2={row.pst2_2!r}')
            print(f'    pred_block1_tokens={row.pred_block1_tokens}')
            print(f'    pred_block2_tokens={row.pred_block2_tokens}')
            print(f'    pred_block1_raw={row.pred_block1_raw!r}')
            print(f'    pred_block2_raw={row.pred_block2_raw!r}')
            print(f'    pred_block1_norm={row.pred_block1_norm!r} in_key={row.pred_block1_in_key}')
            print(f'    pred_block2_norm={row.pred_block2_norm!r} in_key={row.pred_block2_in_key}')
            print(f'    gold2_norm={row.gold2_norm!r} in_key={row.gold2_in_key}')
            print(f'    formatted_pred={row.formatted_pred!r}')
            print(f'    split_pred={row.split_pred}')

    _print_examples('A) block2_nonpad=True but norm2_in_key=False', examples_a)
    _print_examples('B) block2_nonpad=True, norm2_in_key=True but split_returns_1', examples_b)
    _print_examples('C) block2_nonpad=False', examples_c)
    print('=' * 80 + '\n')


def train(
        model: Seq2SeqMixerOccCANINE,
        data_loaders: dict[str, torch.utils.data.DataLoader], # TODO split or use dataclass
        train_sampler: torch.utils.data.distributed.DistributedSampler | None = None,
        loss_fn: LossMixer = None,
        optimizer: torch.optim.Optimizer = None,
        device: torch.device = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        save_dir: str = None,
        total_steps: int = None,
        current_step: int = 0,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 1000,
        save_each_epoch: bool = False,
        log_wandb: bool = False,
        distributed: bool = False,
        is_main_process: bool = True,
        use_amp: bool = False,
        ):
    # Initialize GradScaler for AMP if enabled
    scaler = GradScaler('cuda') if use_amp else None
    
    epoch = 0
    while current_step < total_steps:
        if is_main_process:
            print('\n' + '='*80)
            print(f'Starting Epoch {epoch} (Step {current_step}/{total_steps} - {100*current_step/total_steps:.1f}% complete)')
            print('='*80)
        
        # Set epoch for distributed sampler
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        current_step = train_one_epoch(
            model,
            data_loaders['data_loader_train'],
            loss_fn,
            optimizer,
            device,
            scheduler,
            current_step=current_step,
            epoch=epoch,
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            save_each_epoch=save_each_epoch,
            save_dir=save_dir,
            data_loader_eval=data_loaders['data_loader_val'],
            log_wandb=log_wandb,
            distributed=distributed,
            is_main_process=is_main_process,
            scaler=scaler,
        )
        
        # Save at the end of each epoch if the flag is set
        if save_each_epoch and is_main_process:
            _save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_step=current_step,
                save_dir=save_dir,
                dataset_map_code_label=data_loaders['data_loader_train'].dataset.map_code_label,
            )
            print(f'Model saved at end of epoch {epoch} (step {current_step})')
        
        epoch += 1
    
    # Save model at the end of training to ensure latest version is always saved
    if is_main_process:
        _save_model_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            current_step=current_step,
            save_dir=save_dir,
            dataset_map_code_label=data_loaders['data_loader_train'].dataset.map_code_label,
        )
        print('\n' + '='*80)
        print(f'TRAINING COMPLETE - Final model saved at step {current_step}')
        print('='*80)
