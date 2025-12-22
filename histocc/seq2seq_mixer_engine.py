import os
import time

import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from torch import nn
from sklearn.metrics import accuracy_score

from .formatter import PAD_IDX
from .utils import (
    create_mask,
    Averager,
    order_invariant_accuracy,
    update_summary,
)
from .model_assets import Seq2SeqMixerOccCANINE
from .loss import LossMixer


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
        diagnostic_log_loss_breakdown: bool = False,
        diagnostic_log_target_blocks: bool = False,
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

        if (diagnostic_log_loss_breakdown or diagnostic_log_target_blocks) and is_main_process and batch_idx % log_interval == 0:
            seq2seq_loss = getattr(loss_fn, "loss_fn_seq2seq", None)
            if seq2seq_loss is not None and hasattr(seq2seq_loss, "_push_to_pad"):
                with torch.no_grad():
                    yhat = out_seq2seq.permute(0, 2, 1)[:, :, :-1]
                    target_trim = targets_seq2seq[:, 1:-1]
                    target_mask_diag = None
                    if hasattr(seq2seq_loss, "_get_target_mask"):
                        target_mask_diag = seq2seq_loss._get_target_mask(target_trim)
                    if diagnostic_log_target_blocks and target_mask_diag is not None:
                        num_target_blocks = (~target_mask_diag).sum(dim=1)
                        tqdm.write(f"  [Diag] target blocks (mean/min/max): "
                                   f"{num_target_blocks.float().mean():.2f} / "
                                   f"{int(num_target_blocks.min())} / "
                                   f"{int(num_target_blocks.max())}")
                    if diagnostic_log_loss_breakdown:
                        if target_mask_diag is None:
                            order_loss = seq2seq_loss._order_invariant_loss(yhat, target_trim)
                            padding_loss = seq2seq_loss._push_to_pad(yhat)
                        else:
                            order_loss = seq2seq_loss._order_invariant_loss(yhat, target_trim, target_mask_diag)
                            padding_loss = seq2seq_loss._push_to_pad(yhat, target_mask_diag)
                        tqdm.write(f"  [Diag] loss components -> order: {order_loss.item():.6f} | "
                                   f"padding: {padding_loss.item():.6f} | "
                                   f"scale: {getattr(seq2seq_loss, 'push_to_pad_scale_factor', 'n/a')}")

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

    return losses.avg, losses_linear.avg, losses_seq2seq.avg, seq_accs.avg, token_accs.avg, flat_accs.avg


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
        diagnostic_log_loss_breakdown: bool = False,
        diagnostic_log_target_blocks: bool = False,
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
            diagnostic_log_loss_breakdown=diagnostic_log_loss_breakdown,
            diagnostic_log_target_blocks=diagnostic_log_target_blocks,
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
