import os
import time
import statistics
import random
import faulthandler
import signal

import numpy as np
from collections import Counter
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from torch import nn
from sklearn.metrics import accuracy_score

from .formatter import BOS_IDX, EOS_IDX, PAD_IDX, SEP_IDX
from .utils import (
    create_mask,
    Averager,
    order_invariant_accuracy,
    update_summary,
)
from .utils.masking import generate_square_subsequent_mask
from .model_assets import Seq2SeqMixerOccCANINE
from .loss import LossMixer
from .utils.decoder import mixer_greedy_decode


_ddp_collective_seq = {"value": 0}


def _ddp_debug_collective(tag: str, step: int) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if os.getenv("DEBUG_DDP") != "1":
        return
    _ddp_collective_seq["value"] += 1
    seq = _ddp_collective_seq["value"]
    rank = dist.get_rank()
    print(f"[DDP SYNC] rank{rank} entering {tag} step={step} seq={seq}", flush=True)


def ddp_sync_point(tag: str, step: int, device: torch.device) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    _ddp_debug_collective(f"barrier:{tag}", step)
    if os.getenv("DEBUG_DDP") == "1":
        rank = dist.get_rank()
        print(f"[rank{rank}] entering barrier: {tag}", flush=True)
    dist.barrier()
    if os.getenv("DEBUG_DDP") == "1":
        rank = dist.get_rank()
        print(f"[rank{rank}] leaving barrier: {tag}", flush=True)


def ddp_broadcast(tensor: torch.Tensor, tag: str, step: int, device: torch.device) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    _ddp_debug_collective(f"broadcast:{tag}", step)
    if os.getenv("DEBUG_DDP") == "1":
        rank = dist.get_rank()
        print(f"[rank{rank}] entering broadcast: {tag}", flush=True)
    dist.broadcast(tensor, src=0)
    if os.getenv("DEBUG_DDP") == "1":
        rank = dist.get_rank()
        print(f"[rank{rank}] leaving broadcast: {tag}", flush=True)
    return tensor


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


def collate_sampled_items(
        sampled_items: list[dict[str, torch.Tensor | str | int | float]],
        *,
        rank: int | None = None,
        ) -> dict[str, torch.Tensor | list]:
    stacked_batch: dict[str, torch.Tensor | list] = {}
    for key in sampled_items[0].keys():
        values = [item[key] for item in sampled_items]
        types_seen = {type(v) for v in values}
        if all(torch.is_tensor(v) for v in values):
            shapes = [v.shape for v in values]
            if len({s for s in shapes}) == 1:
                stacked_batch[key] = torch.stack(values)
            else:
                if all(len(s) == 1 for s in shapes):
                    max_len = max(s[0] for s in shapes)
                    padded = [
                        torch.nn.functional.pad(v, (0, max_len - v.shape[0]))
                        for v in values
                    ]
                    stacked_batch[key] = torch.stack(padded)
                else:
                    raise ValueError(f"Failed to stack key='{key}' with shapes {shapes}")
            continue
        if all(isinstance(v, np.ndarray) for v in values):
            stacked_batch[key] = torch.stack([torch.from_numpy(v) for v in values])
            continue
        if all(isinstance(v, (int, float, bool)) for v in values):
            stacked_batch[key] = torch.tensor(values)
            continue
        if any(isinstance(v, str) for v in values) or len(types_seen) > 1:
            if rank == 0:
                logged = getattr(collate_sampled_items, "_logged_keys", set())
                if key not in logged:
                    logged.add(key)
                    collate_sampled_items._logged_keys = logged
                    example = values[0]
                    print(
                        "Doubles-quota sampling: non-stackable key "
                        f"key={key!r} types={sorted(t.__name__ for t in types_seen)} example={example!r}"
                    )
            stacked_batch[key] = values
            continue
        stacked_batch[key] = values
    return stacked_batch


def _is_gate_stable(late_phase_state: dict) -> bool:
    history = late_phase_state["gate_metric_history"]
    window = late_phase_state["gate_stabilize_window"]
    if window <= 0 or len(history) < window:
        return False
    recent = history[-window:]
    return (
        max(recent) - min(recent) <= late_phase_state["gate_stabilize_delta"]
        and min(recent) >= late_phase_state["gate_stabilize_min"]
    )


def _apply_late_phase_switch(
        late_phase_state: dict,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_step: int,
        save_dir: str | None,
        log_wandb: bool,
        ) -> None:
    if late_phase_state["late_switch_once"] and late_phase_state["enabled"]:
        late_phase_state["pending_switch"] = False
        return

    target_lrs = [group["lr"] * late_phase_state["late_lr_mult"] for group in optimizer.param_groups]
    for group, target_lr in zip(optimizer.param_groups, target_lrs):
        group["lr"] = target_lr
    scheduler.base_lrs = list(target_lrs)
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = list(target_lrs)

    late_phase_state["enabled"] = True
    late_phase_state["pending_switch"] = False
    late_phase_state["grad_accum_steps"] = late_phase_state["late_grad_accum"]
    late_phase_state["late_warmup_total"] = late_phase_state["late_warmup_steps"]
    late_phase_state["late_warmup_remaining"] = late_phase_state["late_warmup_steps"]
    late_phase_state["late_warmup_step"] = 0
    late_phase_state["late_warmup_target_lrs"] = target_lrs

    if save_dir is None:
        return

    effective_batch = (
        late_phase_state["batch_size"]
        * late_phase_state["world_size"]
        * late_phase_state["grad_accum_steps"]
    )
    update_summary(
        current_step,
        metrics={
            "late_phase_enabled": int(late_phase_state["enabled"]),
            "grad_accum_steps": late_phase_state["grad_accum_steps"],
            "effective_batch": effective_batch,
            "late_switch_lr": target_lrs[0],
        },
        filename=os.path.join(save_dir, 'logs.csv'),
        log_wandb=log_wandb,
    )


def _apply_late_warmup_step(
        late_phase_state: dict,
        optimizer: torch.optim.Optimizer,
        current_step: int,
        save_dir: str | None,
        log_wandb: bool,
        ) -> None:
    if late_phase_state["late_warmup_remaining"] <= 0:
        return

    late_phase_state["late_warmup_step"] += 1
    warmup_total = max(late_phase_state["late_warmup_total"], 1)
    warmup_factor = late_phase_state["late_warmup_step"] / warmup_total
    target_lrs = late_phase_state["late_warmup_target_lrs"]
    for group, target_lr in zip(optimizer.param_groups, target_lrs):
        group["lr"] = target_lr * warmup_factor

    late_phase_state["late_warmup_remaining"] -= 1
    if late_phase_state["late_warmup_remaining"] != 0 or save_dir is None:
        return

    for group, target_lr in zip(optimizer.param_groups, target_lrs):
        group["lr"] = target_lr

    effective_batch = (
        late_phase_state["batch_size"]
        * late_phase_state["world_size"]
        * late_phase_state["grad_accum_steps"]
    )
    update_summary(
        current_step,
        metrics={
            "late_phase_enabled": int(late_phase_state["enabled"]),
            "grad_accum_steps": late_phase_state["grad_accum_steps"],
            "effective_batch": effective_batch,
            "late_warmup_end_lr": target_lrs[0],
        },
        filename=os.path.join(save_dir, 'logs.csv'),
        log_wandb=log_wandb,
    )


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
        disallow_pad_inside_block: bool = False,
        disallow_zero_at_block_start: bool = False,
        min_double_steps: int = 0,
        min_double_ratio: float = 0.0,
        late_phase_state: dict | None = None,
        ) -> int:
    model = model.train()
    if os.getenv("HANG_DEBUG") == "1" and not hasattr(train_one_epoch, "_faulthandler_registered"):
        faulthandler.register(signal.SIGUSR1)
        train_one_epoch._faulthandler_registered = True

    last_step = len(data_loader) - 1
    losses = Averager()
    batch_time = Averager()
    batch_time_data = Averager()
    samples_per_sec = Averager()
    grad_accum_steps = 1 if late_phase_state is None else late_phase_state["grad_accum_steps"]
    accum_counter = 0
    
    # Check GPU availability once
    has_cuda = torch.cuda.is_available()

    # Need to initialize first "end time", as this is
    # calculated at bottom of batch loop
    end = time.time()
    
    # Use tqdm progress bar only on rank 0
    iterator = tqdm(data_loader, disable=not is_main_process, ncols=100, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(iterator):
        if is_main_process and dist.is_available() and dist.is_initialized():
            debug_detail = os.getenv("TORCH_DISTRIBUTED_DEBUG")
            if debug_detail and not hasattr(train_one_epoch, "_logged_ddp_debug"):
                train_one_epoch._logged_ddp_debug = True
                print(f"[DDP] TORCH_DISTRIBUTED_DEBUG={debug_detail}", flush=True)
        # Only switch late-phase settings right after an optimizer step (accum_counter == 0).
        if late_phase_state is not None and late_phase_state["pending_switch"] and accum_counter == 0:
            _apply_late_phase_switch(
                late_phase_state=late_phase_state,
                optimizer=optimizer,
                scheduler=scheduler,
                current_step=current_step,
                save_dir=save_dir,
                log_wandb=log_wandb,
            )
            grad_accum_steps = late_phase_state["grad_accum_steps"]

        current_step += 1

        if min_double_steps and current_step <= min_double_steps and min_double_ratio > 0:
            dataset = data_loader.dataset
            if not hasattr(dataset, "_double_indices"):
                target_cols = getattr(dataset, "target_cols", [])
                double_indices = []
                single_indices = []
                if hasattr(dataset, "frame") and target_cols:
                    second_col = target_cols[1] if len(target_cols) > 1 else None
                    if second_col is not None:
                        for idx, val in enumerate(dataset.frame[second_col].tolist()):
                            if _pst2_value_present(val):
                                double_indices.append(idx)
                            else:
                                single_indices.append(idx)
                dataset._double_indices = double_indices
                dataset._single_indices = single_indices

            double_indices = getattr(dataset, "_double_indices", [])
            if double_indices:
                gold_num_codes = batch['gold_num_codes']
                batch_size = gold_num_codes.size(0)
                min_doubles = int(min_double_ratio * batch_size + 0.999)
                current_doubles = int((gold_num_codes >= 2).sum().item())
                if current_doubles < min_doubles:
                    singles_idx = (gold_num_codes < 2).nonzero(as_tuple=False).flatten().tolist()
                    replace_count = min(len(singles_idx), min_doubles - current_doubles)
                    if replace_count > 0:
                        replace_idx = singles_idx[:replace_count]
                        sampled_indices = random.choices(double_indices, k=replace_count)
                        sampled_items = [dataset[idx] for idx in sampled_indices]
                        rank = None
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            rank = torch.distributed.get_rank()
                        stacked_items = collate_sampled_items(sampled_items, rank=rank)
                        for key, stacked in stacked_items.items():
                            if torch.is_tensor(batch[key]):
                                if not torch.is_tensor(stacked):
                                    raise ValueError(f"Expected tensor for key '{key}' but got {type(stacked)}")
                                batch[key][replace_idx] = stacked.to(batch[key].device)
                            else:
                                for idx, row_idx in enumerate(replace_idx):
                                    batch[key][row_idx] = stacked[idx]

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets_seq2seq = batch['targets_seq2seq'].to(device, non_blocking=True)
        targets_linear = batch['targets_linear'].to(device, non_blocking=True)
        gold_num_codes = batch['gold_num_codes'].to(device, non_blocking=True)

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
        if accum_counter == 0:
            optimizer.zero_grad()
        loss = loss / grad_accum_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum_counter += 1
        if accum_counter == grad_accum_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            accum_counter = 0

            if late_phase_state is not None and late_phase_state["late_warmup_remaining"] > 0:
                _apply_late_warmup_step(
                    late_phase_state=late_phase_state,
                    optimizer=optimizer,
                    current_step=current_step,
                    save_dir=save_dir,
                    log_wandb=log_wandb,
                )
            else:
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
            current_lr = optimizer.param_groups[0]['lr']
            
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

        if save_interval is not None and current_step % save_interval == 0 and distributed and dist.is_available() and dist.is_initialized():
            ddp_sync_point("pre_checkpoint", current_step, device)
            if is_main_process and not save_each_epoch:
                _save_model_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    current_step=current_step,
                    save_dir=save_dir,
                    dataset_map_code_label=data_loader.dataset.map_code_label,
                )
            ddp_sync_point("post_checkpoint", current_step, device)
        elif save_interval is not None and current_step % save_interval == 0 and is_main_process and not save_each_epoch:
            _save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_step=current_step,
                save_dir=save_dir,
                dataset_map_code_label=data_loader.dataset.map_code_label,
            )

        is_eval_step = eval_interval is not None and current_step % eval_interval == 0
        debug_ddp_eval = os.getenv("DEBUG_DDP_EVAL") == "1"
        if eval_interval is not None and distributed and dist.is_available() and dist.is_initialized():
            eval_tensor = torch.tensor(1 if is_eval_step and is_main_process else 0, device=device)
            ddp_broadcast(eval_tensor, "eval_flag", current_step, device)
            is_eval_step = bool(eval_tensor.item())
        if is_eval_step and distributed and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if debug_ddp_eval:
                print(f"[DDP EVAL] rank{rank} entering eval section step {current_step}", flush=True)
            if debug_ddp_eval and is_main_process:
                print(f"[DDP EVAL] rank0 pre_eval barrier step {current_step}", flush=True)
            ddp_sync_point("pre_eval", current_step, device)
            eval_error = None
            probe_error = None
            eval_loss = float("nan")
            eval_loss_linear = float("nan")
            eval_loss_seq2seq = float("nan")
            eval_seq_acc = float("nan")
            eval_token_acc = float("nan")
            eval_flat_acc = float("nan")
            gating_summary = {}
            late_phase_metrics = {}
            try:
                if is_main_process:
                    if debug_ddp_eval:
                        print(f"[DDP EVAL] rank0 entering eval step {current_step}", flush=True)
                    try:
                        tqdm.write('\n' + '='*80)
                        tqdm.write('Starting evaluation pass...')
                        compute_gating_metrics = late_phase_state is not None
                        eval_loss, eval_loss_linear, eval_loss_seq2seq, eval_seq_acc, eval_token_acc, eval_flat_acc, gating_metrics = evaluate(
                            model=model,
                            data_loader=data_loader_eval,
                            loss_fn=loss_fn,
                            device=device,
                            disallow_pad_inside_block=disallow_pad_inside_block,
                            disallow_zero_at_block_start=disallow_zero_at_block_start,
                            compute_gating_metrics=compute_gating_metrics,
                            require_gold_num_codes=compute_gating_metrics,
                            run_probe=False,
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
                        tqdm.write(f'Learning Rate       : {optimizer.param_groups[0]["lr"]:.2e}')
                        tqdm.write('='*80 + '\n')

                        if late_phase_state is not None:
                            effective_batch = late_phase_state["batch_size"] * late_phase_state["world_size"] * late_phase_state["grad_accum_steps"]
                            late_phase_metrics.update(
                                {
                                    "late_phase_enabled": int(late_phase_state["enabled"]),
                                    "grad_accum_steps": late_phase_state["grad_accum_steps"],
                                    "effective_batch": effective_batch,
                                }
                            )

                        gating_summary = gating_metrics or {}
                        if late_phase_state is not None and gating_summary:
                            gate_metric = late_phase_state["gate_stabilize_metric"]
                            if gate_metric in gating_summary:
                                history = late_phase_state["gate_metric_history"]
                                history.append(gating_summary[gate_metric])
                                if (
                                    (not late_phase_state["enabled"] or not late_phase_state["late_switch_once"])
                                    and _is_gate_stable(late_phase_state)
                                ):
                                    late_phase_state["pending_switch"] = True

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
                                'lr': optimizer.param_groups[0]['lr'],
                                **gating_summary,
                                **late_phase_metrics,
                            },
                            filename=os.path.join(save_dir, 'logs.csv'),
                            log_wandb=log_wandb,
                        )
                    except Exception as exc:
                        eval_error = exc
                    try:
                        _run_pst2_eval_probe(
                            model=model,
                            data_loader=data_loader_eval,
                            device=device,
                            sample_size=200,
                            seed=42,
                            disallow_pad_inside_block=disallow_pad_inside_block,
                            disallow_zero_at_block_start=disallow_zero_at_block_start,
                        )
                    except Exception as exc:
                        probe_error = exc
            finally:
                if debug_ddp_eval and is_main_process:
                    print(f"[DDP EVAL] rank0 post_eval barrier step {current_step}", flush=True)
                ddp_sync_point("post_eval", current_step, device)
                if debug_ddp_eval:
                    print(f"[DDP EVAL] rank{rank} exiting eval section step {current_step}", flush=True)

            eval_failed = torch.tensor(
                1 if (eval_error is not None or probe_error is not None) else 0,
                device=device,
            )
            ddp_broadcast(eval_failed, "eval_failed", current_step, device)
            metrics_to_broadcast = [
                ("val_loss", eval_loss),
                ("val_loss_linear", eval_loss_linear),
                ("val_loss_seq2seq", eval_loss_seq2seq),
                ("seq_acc", eval_seq_acc),
                ("token_acc", eval_token_acc),
                ("flat_acc", eval_flat_acc),
                ("gating_precision", gating_summary.get("gating_precision", float("nan"))),
                ("gating_recall", gating_summary.get("gating_recall", float("nan"))),
                ("gating_f1", gating_summary.get("gating_f1", float("nan"))),
                ("gating_tp", gating_summary.get("gating_tp", float("nan"))),
                ("gating_fp", gating_summary.get("gating_fp", float("nan"))),
                ("gating_fn", gating_summary.get("gating_fn", float("nan"))),
                ("gating_tn", gating_summary.get("gating_tn", float("nan"))),
                ("late_phase_enabled", late_phase_metrics.get("late_phase_enabled", 0)),
                ("grad_accum_steps", late_phase_metrics.get("grad_accum_steps", 1)),
                ("effective_batch", late_phase_metrics.get("effective_batch", float("nan"))),
            ]
            for name, value in metrics_to_broadcast:
                tensor = torch.tensor(value, device=device)
                if debug_ddp_eval and is_main_process:
                    print(f"[DDP EVAL] rank0 broadcasting {name} step {current_step}", flush=True)
                ddp_broadcast(tensor, f"metric:{name}", current_step, device)
            ddp_sync_point("post_eval_broadcasts", current_step, device)
            if eval_failed.item() == 1:
                if eval_error is not None:
                    raise eval_error
                if probe_error is not None:
                    raise probe_error
                raise RuntimeError("Eval/probe failed on rank0; aborting on all ranks.")

            switch_tensor = torch.tensor(
                1 if late_phase_state is not None and late_phase_state["pending_switch"] else 0,
                device=device,
            )
            ddp_broadcast(switch_tensor, "switch_flag", current_step, device)
            if late_phase_state is not None and switch_tensor.item() == 1:
                late_phase_state["pending_switch"] = True
        elif is_eval_step and is_main_process:
            tqdm.write('\n' + '='*80)
            tqdm.write('Starting evaluation pass...')
            compute_gating_metrics = late_phase_state is not None
            eval_loss, eval_loss_linear, eval_loss_seq2seq, eval_seq_acc, eval_token_acc, eval_flat_acc, gating_metrics = evaluate(
                model=model,
                data_loader=data_loader_eval,
                loss_fn=loss_fn,
                device=device,
                disallow_pad_inside_block=disallow_pad_inside_block,
                disallow_zero_at_block_start=disallow_zero_at_block_start,
                compute_gating_metrics=compute_gating_metrics,
                require_gold_num_codes=compute_gating_metrics,
                run_probe=True,
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
            tqdm.write(f'Learning Rate       : {optimizer.param_groups[0]["lr"]:.2e}')
            tqdm.write('='*80 + '\n')

            late_phase_metrics = {}
            if late_phase_state is not None:
                effective_batch = late_phase_state["batch_size"] * late_phase_state["world_size"] * late_phase_state["grad_accum_steps"]
                late_phase_metrics.update(
                    {
                        "late_phase_enabled": int(late_phase_state["enabled"]),
                        "grad_accum_steps": late_phase_state["grad_accum_steps"],
                        "effective_batch": effective_batch,
                    }
                )

            gating_summary = gating_metrics or {}
            if late_phase_state is not None and gating_summary:
                gate_metric = late_phase_state["gate_stabilize_metric"]
                if gate_metric in gating_summary:
                    history = late_phase_state["gate_metric_history"]
                    history.append(gating_summary[gate_metric])
                    if (
                        (not late_phase_state["enabled"] or not late_phase_state["late_switch_once"])
                        and _is_gate_stable(late_phase_state)
                    ):
                        late_phase_state["pending_switch"] = True

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
                    'lr': optimizer.param_groups[0]['lr'],
                    **gating_summary,
                    **late_phase_metrics,
                },
                filename=os.path.join(save_dir, 'logs.csv'),
                log_wandb=log_wandb,
            )

        end = time.time()

    if accum_counter:
        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        accum_counter = 0
        if late_phase_state is not None and late_phase_state["late_warmup_remaining"] > 0:
            _apply_late_warmup_step(
                late_phase_state=late_phase_state,
                optimizer=optimizer,
                current_step=current_step,
                save_dir=save_dir,
                log_wandb=log_wandb,
            )
        else:
            scheduler.step()

    return current_step


@torch.no_grad
def evaluate(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        log_interval: int = 100,
        disallow_pad_inside_block: bool = False,
        disallow_zero_at_block_start: bool = False,
        require_gold_num_codes: bool = False,
        compute_gating_metrics: bool = False,
        run_probe: bool = True,
        ):
    model = model.eval()
    if not hasattr(evaluate, "_logged_file"):
        evaluate._logged_file = True
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if rank == 0:
            print(f'evaluate() running from {__file__}')

    losses = Averager()
    losses_linear = Averager()
    losses_seq2seq = Averager()

    token_accs = Averager()
    seq_accs = Averager()
    flat_accs = Averager()
    gating_tp = 0
    gating_fp = 0
    gating_fn = 0
    gating_tn = 0
    formatter = getattr(data_loader.dataset, "formatter", None)

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets_seq2seq = batch['targets_seq2seq'].to(device, non_blocking=True)
        targets_linear = batch['targets_linear'].to(device, non_blocking=True)
        gold_num_codes = batch.get('gold_num_codes')
        if gold_num_codes is not None:
            gold_num_codes = gold_num_codes.to(device, non_blocking=True)
        elif require_gold_num_codes:
            raise ValueError("gold_num_codes is required for evaluate(), but was not found in the batch.")

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
            gold_num_codes=gold_num_codes,
            )
        loss_linear = loss_fn.loss_fn_linear(out_linear, targets_linear)
        loss_seq2seq = loss_fn.loss_fn_seq2seq(out_seq2seq, targets_seq2seq, gold_num_codes=gold_num_codes)

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

        if compute_gating_metrics:
            if formatter is None:
                raise ValueError("compute_gating_metrics=True requires dataset.formatter to be present.")
            if gold_num_codes is None:
                raise ValueError("compute_gating_metrics=True requires gold_num_codes in the batch.")
            decode_max_num_codes = min(2, formatter.max_num_codes)
            zero_idx = formatter.map_char_idx.get('0') if hasattr(formatter, "map_char_idx") else None
            outputs = mixer_greedy_decode(
                model=model,
                descr=input_ids,
                input_attention_mask=attention_mask,
                device=device,
                max_len=formatter.max_seq_len,
                start_symbol=BOS_IDX,
                pad_idx=PAD_IDX,
                block_size=formatter.block_size,
                max_num_codes=decode_max_num_codes,
                disallow_pad_inside_block=disallow_pad_inside_block,
                disallow_zero_at_block_start=disallow_zero_at_block_start,
                zero_idx=zero_idx,
            )
            preds_seq = outputs[0].cpu().numpy()
            block2_start = 1 + formatter.block_size
            block2_tokens = preds_seq[:, block2_start:block2_start + formatter.block_size]
            pred_has2 = (block2_tokens != PAD_IDX).any(axis=1)
            gold_has2 = (gold_num_codes >= 2).detach().cpu().numpy()
            gating_tp += int((pred_has2 & gold_has2).sum())
            gating_fn += int((~pred_has2 & gold_has2).sum())
            gating_fp += int((pred_has2 & ~gold_has2).sum())
            gating_tn += int((~pred_has2 & ~gold_has2).sum())

        if batch_idx % log_interval == 0:
            tqdm.write(f'  Eval Batch {batch_idx + 1}/{len(data_loader)} | '
                       f'Seq Acc: {seq_accs.avg:.2f}% | '
                       f'Token Acc: {token_accs.avg:.2f}% | '
                       f'Flat Acc: {flat_accs.avg:.2f}% | '
                       f'Val Loss: {losses.avg:.6f}')

    if run_probe:
        _run_pst2_eval_probe(
            model=model,
            data_loader=data_loader,
            device=device,
            sample_size=200,
            seed=42,
            disallow_pad_inside_block=disallow_pad_inside_block,
            disallow_zero_at_block_start=disallow_zero_at_block_start,
        )

    precision = gating_tp / (gating_tp + gating_fp) if (gating_tp + gating_fp) else 0.0
    recall = gating_tp / (gating_tp + gating_fn) if (gating_tp + gating_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    gating_metrics = {
        "gating_precision": precision,
        "gating_recall": recall,
        "gating_f1": f1,
        "gating_tp": gating_tp,
        "gating_fp": gating_fp,
        "gating_fn": gating_fn,
        "gating_tn": gating_tn,
    } if compute_gating_metrics else None

    return losses.avg, losses_linear.avg, losses_seq2seq.avg, seq_accs.avg, token_accs.avg, flat_accs.avg, gating_metrics


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
    if hasattr(formatter, "map_idx_char"):
        rev_mapping = formatter.map_idx_char
        missing = [
            int(tok) for tok in block_tokens
            if int(tok) not in rev_mapping and int(tok) not in {PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX}
        ]
        if missing and not getattr(_decode_block_string, "_logged_missing", False):
            _decode_block_string._logged_missing = True
            rank = 0
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            if rank == 0:
                min_tok = int(min(block_tokens))
                max_tok = int(max(block_tokens))
                sample_missing = sorted(set(missing))[:10]
                print(
                    "PST2 probe warning: tokens missing from rev_mapping "
                    f"min_tok={min_tok} max_tok={max_tok} missing_sample={sample_missing}"
                )
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
        disallow_pad_inside_block: bool = False,
        disallow_zero_at_block_start: bool = False,
) -> None:
    strict_probe = os.getenv("STRICT_PROBE") == "1"
    try:
        _run_pst2_eval_probe_inner(
            model=model,
            data_loader=data_loader,
            device=device,
            sample_size=sample_size,
            seed=seed,
            disallow_pad_inside_block=disallow_pad_inside_block,
            disallow_zero_at_block_start=disallow_zero_at_block_start,
        )
    except Exception as exc:
        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        if rank == 0:
            print(f'PST2 eval probe failed: {exc}')
        if strict_probe:
            raise
        return


def _run_pst2_eval_probe_inner(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        sample_size: int = 200,
        seed: int = 42,
        disallow_pad_inside_block: bool = False,
        disallow_zero_at_block_start: bool = False,
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
    single_positions = [idx for idx, flag in enumerate(has_second) if not flag]

    inv_key = dataset.map_code_label
    use_within_block_sep = bool(getattr(formatter, 'within_block_sep', None))

    model_to_decode = model.module if hasattr(model, 'module') else model
    model_to_decode.eval()

    examples_a: list[_PST2ProbeRow] = []
    examples_b: list[_PST2ProbeRow] = []
    examples_c: list[_PST2ProbeRow] = []

    print('\n' + '=' * 80)
    print('PST2 EVAL PROBE (deterministic sample)')
    print(f'  sample_size={sample_size} seed={seed}')
    print(f'  PAD_IDX={PAD_IDX} block_size={formatter.block_size} max_num_codes={formatter.max_num_codes}')

    eval_configs = [
        ("realistic", 0.05),
        ("balanced", 0.50),
    ]

    for label, target_double_rate in eval_configs:
        rng = torch.Generator().manual_seed(seed + int(target_double_rate * 100))
        sample_total = min(sample_size, len(dataset.frame))
        desired_doubles = min(int(sample_total * target_double_rate), len(eligible_positions))
        desired_singles = min(sample_total - desired_doubles, len(single_positions))
        if desired_doubles + desired_singles == 0:
            continue

        double_indices = []
        if desired_doubles:
            double_perm = torch.randperm(len(eligible_positions), generator=rng)[:desired_doubles]
            double_indices = [eligible_positions[i] for i in double_perm.tolist()]

        single_indices = []
        if desired_singles:
            single_perm = torch.randperm(len(single_positions), generator=rng)[:desired_singles]
            single_indices = [single_positions[i] for i in single_perm.tolist()]

        sample_indices = double_indices + single_indices

        block2_nonpad_count = 0
        block2_nonpad_with_pad_count = 0
        pad_inside_block_pred_count = 0
        pad_inside_block_pred_total = 0
        pad_inside_block_gold_count = 0
        pad_inside_block_gold_total = 0
        blocks_emitted_counter = Counter()
        block_start_zero_count = 0
        block_start_total = 0
        norm2_in_key_count = 0
        format_contains_sep_value_count = 0
        split_returns_2_count = 0
        norm2_miss_counter = Counter()
        pred_block2_raw_counter = Counter()
        gold2_in_key_count = 0
        gold2_miss_counter = Counter()
        pred_has2_count = 0
        gold_has2_count = 0
        gold_has2_with_pred_has2 = 0
        gold_single_with_pred_has2 = 0
        gating_tp = 0
        gating_fp = 0
        gating_fn = 0
        gating_tn = 0
        gold_has2_exact_match = 0
        gold_has2_block2_in_key = 0
        pred_has2_block2_in_key = 0
        pred_has2_valid_count = 0
        block2_token_match = 0
        block2_token_total = 0
        pad_prob_bins = {i: {"count": 0, "gold_has2": 0} for i in range(5)}
        pad_prob_singles = []
        pad_prob_doubles = []

        batch_size = 32
        for offset in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[offset:offset + batch_size]
            batch_items = [dataset[idx] for idx in batch_indices]
            input_ids = torch.stack([item['input_ids'] for item in batch_items]).to(device, non_blocking=True)
            attention_mask = torch.stack([item['attention_mask'] for item in batch_items]).to(device, non_blocking=True)

            decode_max_num_codes = min(2, formatter.max_num_codes)
            zero_idx = formatter.map_char_idx.get('0')
            outputs = mixer_greedy_decode(
                model=model_to_decode,
                descr=input_ids,
                input_attention_mask=attention_mask,
                device=device,
                max_len=formatter.max_seq_len,
                start_symbol=BOS_IDX,
                pad_idx=PAD_IDX,
                block_size=formatter.block_size,
                max_num_codes=decode_max_num_codes,
                disallow_pad_inside_block=disallow_pad_inside_block,
                disallow_zero_at_block_start=disallow_zero_at_block_start,
                zero_idx=zero_idx,
            )
            preds_seq = outputs[0].cpu().numpy()

            block2_start = 1 + formatter.block_size
            prefix_len = block2_start
            prefix_seq = torch.tensor(
                [preds_seq[i][:prefix_len] for i in range(len(batch_indices))],
                device=device,
                dtype=torch.long,
            )
            target_mask = generate_square_subsequent_mask(prefix_len, device).type(torch.bool)
            memory = model_to_decode.encode(input_ids, attention_mask)
            if isinstance(memory, tuple):
                memory = memory[0]
            block2_logits = model_to_decode.decode(
                memory=memory,
                target=prefix_seq,
                target_mask=target_mask,
                target_padding_mask=None,
            )[:, -1, :]
            block2_pad_probs = torch.softmax(block2_logits, dim=1)[:, PAD_IDX].detach().cpu().numpy()

            for row_pos, dataset_idx in enumerate(batch_indices):
                record = dataset.frame.iloc[dataset_idx]
                raw_seq = preds_seq[row_pos].tolist()
                block1_tokens = raw_seq[1:1 + formatter.block_size]
                block2_tokens = raw_seq[1 + formatter.block_size:1 + 2 * formatter.block_size]
                block2_nonpad = any(tok != PAD_IDX for tok in block2_tokens)
                block2_has_pad = any(tok == PAD_IDX for tok in block2_tokens)
                code_region_tokens = raw_seq[1:1 + decode_max_num_codes * formatter.block_size]
                emitted_blocks = 0
                for block_start in range(1, 1 + decode_max_num_codes * formatter.block_size, formatter.block_size):
                    if raw_seq[block_start] == PAD_IDX:
                        break
                    emitted_blocks += 1
                blocks_emitted_counter[emitted_blocks] += 1
                if zero_idx is not None:
                    for block_start in range(1, 1 + decode_max_num_codes * formatter.block_size, formatter.block_size):
                        block_start_total += 1
                        if raw_seq[block_start] == zero_idx:
                            block_start_zero_count += 1

                pred_block1_raw = _decode_block_string(formatter, block1_tokens, 0)
                pred_block2_raw = _decode_block_string(formatter, block2_tokens, 1)
                pred_block1_norm = _normalize_code_for_lookup(pred_block1_raw, inv_key, use_within_block_sep)
                pred_block2_norm = _normalize_code_for_lookup(pred_block2_raw, inv_key, use_within_block_sep)
                pred_block1_in_key = pred_block1_norm in inv_key
                pred_block2_in_key = pred_block2_norm in inv_key
                gold2_raw = str(record['pst2_2'])
                gold2_norm = _normalize_code_for_lookup(gold2_raw, inv_key, use_within_block_sep)
                gold2_in_key = gold2_norm in inv_key
                gold_has2 = _pst2_value_present(record['pst2_2'])

                formatted_pred = formatter.clean_pred(torch.tensor(raw_seq).numpy())
                split_pred = _split_str_s2s(formatted_pred, formatter.sep_value)
                split_pred_list = split_pred if isinstance(split_pred, list) else [split_pred]

                pred_has2 = block2_nonpad
                pred_has2_valid = pred_has2 and pred_block2_in_key and not block2_has_pad
                if pred_has2:
                    pred_has2_count += 1
                if gold_has2:
                    gold_has2_count += 1
                    if pred_has2:
                        gold_has2_with_pred_has2 += 1
                        if pred_block2_norm == gold2_norm:
                            gold_has2_exact_match += 1
                        gating_tp += 1
                    else:
                        gating_fn += 1
                    gold_has2_block2_in_key += int(pred_block2_in_key)
                    gold_block2_tokens = batch_items[row_pos]['targets_seq2seq'][1 + formatter.block_size:1 + 2 * formatter.block_size]
                    block2_token_match += int((torch.tensor(block2_tokens) == gold_block2_tokens).sum())
                    block2_token_total += formatter.block_size
                else:
                    if pred_has2:
                        gold_single_with_pred_has2 += 1
                        gating_fp += 1
                    else:
                        gating_tn += 1

                if pred_has2:
                    pad_inside_block_pred_total += formatter.block_size - 1
                    pad_inside_block_pred_count += sum(
                        tok == PAD_IDX for idx, tok in enumerate(block2_tokens) if idx % formatter.block_size != 0
                    )
                if gold_has2:
                    pad_inside_block_gold_total += formatter.block_size - 1
                    pad_inside_block_gold_count += sum(
                        tok == PAD_IDX for idx, tok in enumerate(gold_block2_tokens.tolist()) if idx % formatter.block_size != 0
                    )

                if pred_has2 and pred_block2_in_key:
                    pred_has2_block2_in_key += 1
                if pred_has2_valid:
                    pred_has2_valid_count += 1

                pad_prob = float(block2_pad_probs[row_pos])
                bin_idx = min(int(pad_prob * 5), 4)
                pad_prob_bins[bin_idx]["count"] += 1
                pad_prob_bins[bin_idx]["gold_has2"] += int(gold_has2)
                if gold_has2:
                    pad_prob_doubles.append(pad_prob)
                else:
                    pad_prob_singles.append(pad_prob)

                if block2_nonpad:
                    block2_nonpad_count += 1
                    if block2_has_pad:
                        block2_nonpad_with_pad_count += 1
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
                pred_block2_raw_counter[pred_block2_raw] += 1

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

        total = float(len(sample_indices))
        print(f'\n[{label}] Summary counters:')
        print(f'  % pred_block2_nonpad: {block2_nonpad_count / total:.2%}')
        if block2_nonpad_count:
            print(f'  % block2_nonpad_with_pad: {block2_nonpad_with_pad_count / block2_nonpad_count:.2%}')
            if block2_nonpad_with_pad_count:
                print('  WARN: block2_nonpad rows still contain PAD tokens inside the block.')
        if pad_inside_block_pred_total:
            print(f'  % pad_inside_block | pred_has2: {pad_inside_block_pred_count / pad_inside_block_pred_total:.2%}')
        if pad_inside_block_gold_total:
            print(f'  % pad_inside_block | gold_has2: {pad_inside_block_gold_count / pad_inside_block_gold_total:.2%}')
        print(f'  blocks_emitted distribution: {dict(blocks_emitted_counter)}')
        if block_start_total:
            print(f'  % block_starts_predicted_zero: {block_start_zero_count / block_start_total:.2%}')
        print(f'  % norm2_in_key: {norm2_in_key_count / total:.2%}')
        print(f'  % gold2_in_key: {gold2_in_key_count / total:.2%}')
        print(f'  % format_contains_sep_value: {format_contains_sep_value_count / total:.2%}')
        print(f'  % split_returns_2: {split_returns_2_count / total:.2%}')

        precision = gold_has2_with_pred_has2 / pred_has2_count if pred_has2_count else 0.0
        recall = gold_has2_with_pred_has2 / gold_has2_count if gold_has2_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f'  gating precision/recall/F1: {precision:.2%}/{recall:.2%}/{f1:.2%}')
        print(f'  gating confusion: TP={gating_tp} FP={gating_fp} FN={gating_fn} TN={gating_tn}')
        if gold_has2_count:
            print(f'  EM_block2 | gold_has2: {gold_has2_exact_match / gold_has2_count:.2%}')
            print(f'  token_acc_block2 | gold_has2: {block2_token_match / block2_token_total:.2%}')
            print(f'  % block2_in_key | gold_has2: {gold_has2_block2_in_key / gold_has2_count:.2%}')
        if pred_has2_count:
            print(f'  % block2_in_key | pred_has2: {pred_has2_block2_in_key / pred_has2_count:.2%}')
            print(f'  % block2_valid_post_sanitize | pred_has2: {pred_has2_valid_count / pred_has2_count:.2%}')
        single_total = total - gold_has2_count
        if single_total:
            print(f'  FPR(pred_has2 | gold_has2=False): {gold_single_with_pred_has2 / single_total:.2%}')

        if pad_prob_singles:
            print(f'  p(PAD@pos8) singles mean/median: {statistics.fmean(pad_prob_singles):.4f}/{statistics.median(pad_prob_singles):.4f}')
        if pad_prob_doubles:
            print(f'  p(PAD@pos8) doubles mean/median: {statistics.fmean(pad_prob_doubles):.4f}/{statistics.median(pad_prob_doubles):.4f}')

        print('  Calibration (PAD prob at block2 start):')
        for bin_idx in range(5):
            bucket = pad_prob_bins[bin_idx]
            if bucket["count"] == 0:
                continue
            rate = bucket["gold_has2"] / bucket["count"]
            print(f'    bin[{bin_idx}] count={bucket["count"]} gold_has2_rate={rate:.2%}')

        print('\nTop-20 normalized block-2 strings missing from key:')
        for code, count in norm2_miss_counter.most_common(20):
            print(f'  {code!r}: {count}')
        if gold2_miss_counter:
            print('\nTop-20 gold pst2_2 strings missing from key:')
            for code, count in gold2_miss_counter.most_common(20):
                print(f'  {code!r}: {count}')
        print('\nTop-10 block-2 raw strings:')
        for code, count in pred_block2_raw_counter.most_common(10):
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
        disallow_pad_inside_block: bool = False,
        disallow_zero_at_block_start: bool = False,
        min_double_steps: int = 0,
        min_double_ratio: float = 0.0,
        gate_stabilize_metric: str = "gating_f1",
        gate_stabilize_window: int = 5,
        gate_stabilize_delta: float = 0.02,
        gate_stabilize_min: float = 0.90,
        late_grad_accum: int = 1,
        late_lr_mult: float = 1.0,
        late_warmup_steps: int = 0,
        late_switch_once: bool = True,
        batch_size: int | None = None,
        ):
    # Initialize GradScaler for AMP if enabled
    scaler = GradScaler('cuda') if use_amp else None

    world_size = 1
    if distributed and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    elif data_loaders.get("data_loader_train") is not None:
        world_size = getattr(data_loaders["data_loader_train"].sampler, "num_replicas", 1)
    if batch_size is None:
        batch_size = data_loaders["data_loader_train"].batch_size

    enable_late_phase = (
        late_grad_accum > 1
        or late_lr_mult != 1.0
        or late_warmup_steps > 0
    )
    late_phase_state = None
    if enable_late_phase:
        late_phase_state = {
            "enabled": False,
            "pending_switch": False,
            "grad_accum_steps": 1,
            "late_grad_accum": late_grad_accum,
            "late_lr_mult": late_lr_mult,
            "late_warmup_steps": late_warmup_steps,
            "late_warmup_total": 0,
            "late_warmup_remaining": 0,
            "late_warmup_step": 0,
            "late_warmup_target_lrs": [],
            "gate_metric_history": [],
            "gate_stabilize_metric": gate_stabilize_metric,
            "gate_stabilize_window": gate_stabilize_window,
            "gate_stabilize_delta": gate_stabilize_delta,
            "gate_stabilize_min": gate_stabilize_min,
            "late_switch_once": late_switch_once,
            "batch_size": batch_size,
            "world_size": world_size,
        }
    
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
            disallow_pad_inside_block=disallow_pad_inside_block,
            disallow_zero_at_block_start=disallow_zero_at_block_start,
            min_double_steps=min_double_steps,
            min_double_ratio=min_double_ratio,
            late_phase_state=late_phase_state,
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
