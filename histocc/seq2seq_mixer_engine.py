import os
import time
import statistics
import random

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
from .target_cleaning import clean_target_value


_LOSS_DIAGNOSTICS_RAN = False


def ddp_sync_point(tag: str, step: int, device: torch.device) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    dist.barrier()


def ddp_broadcast(tensor: torch.Tensor, tag: str, step: int, device: torch.device) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    dist.broadcast(tensor, src=0)
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


def _log_loss_debug(
        *,
        debug_info: dict | None,
        gold_num_codes: torch.Tensor,
        step: int,
        prefix: str = "LOSS_DEBUG",
        extra_metrics: dict | None = None,
        ) -> None:
    if debug_info is None:
        return

    def _masked_mean(values: torch.Tensor | None, mask: torch.Tensor) -> float:
        if values is None:
            return float("nan")
        if not mask.any():
            return float("nan")
        masked = values[mask]
        if masked.numel() == 0:
            return 0.0
        return float(masked.mean().item())

    def _masked_pctl(values: torch.Tensor | None, mask: torch.Tensor, q: float) -> float:
        if values is None:
            return float("nan")
        if mask.dtype is not torch.bool:
            mask = mask.bool()
        if not mask.any():
            return float("nan")
        masked = values[mask]
        if masked.numel() == 0:
            return float("nan")
        masked = masked.float()
        masked = masked[torch.isfinite(masked)]
        if masked.numel() == 0:
            return float("nan")
        return float(torch.quantile(masked, q).item())

    gold_num_codes = gold_num_codes.detach().cpu()
    singles = gold_num_codes == 1
    doubles = gold_num_codes >= 2

    order_per_sample = debug_info.get("order_invariant_per_sample")
    push_per_sample = debug_info.get("push_to_pad_per_sample")
    gate_per_sample = debug_info.get("gate_loss_per_sample")
    coverage_per_sample = debug_info.get("coverage_loss_per_sample")
    double_coverage_per_sample = debug_info.get("double_coverage_loss_per_sample")
    if order_per_sample is not None:
        order_per_sample = order_per_sample.detach().cpu()
    if push_per_sample is not None:
        push_per_sample = push_per_sample.detach().cpu()
    if gate_per_sample is not None:
        gate_per_sample = gate_per_sample.detach().cpu()
    if coverage_per_sample is not None:
        coverage_per_sample = coverage_per_sample.detach().cpu()
    if double_coverage_per_sample is not None:
        double_coverage_per_sample = double_coverage_per_sample.detach().cpu()

    if gate_per_sample is not None and gate_per_sample.shape[0] != gold_num_codes.shape[0]:
        tqdm.write(
            f"{prefix} step={step} gate_loss_per_sample_shape_mismatch "
            f"gate_shape={tuple(gate_per_sample.shape)} gold_shape={tuple(gold_num_codes.shape)}"
        )

    msg = (
        f"{prefix} step={step} "
        f"order_mean_s={_masked_mean(order_per_sample, singles):.4f} "
        f"order_mean_d={_masked_mean(order_per_sample, doubles):.4f} "
        f"push_mean_s={_masked_mean(push_per_sample, singles):.4f} "
        f"push_mean_d={_masked_mean(push_per_sample, doubles):.4f} "
        f"gate_mean_s={_masked_mean(gate_per_sample, singles):.4f} "
        f"gate_mean_d={_masked_mean(gate_per_sample, doubles):.4f} "
        f"coverage_mean_s={_masked_mean(coverage_per_sample, singles):.4f} "
        f"coverage_mean_d={_masked_mean(coverage_per_sample, doubles):.4f} "
        f"double_cov_mean_d={_masked_mean(double_coverage_per_sample, doubles):.4f}"
    )
    msg += (
        f" order_p90_s={_masked_pctl(order_per_sample, singles, 0.9):.4f} "
        f"order_p90_d={_masked_pctl(order_per_sample, doubles, 0.9):.4f} "
        f"gate_p90_d={_masked_pctl(gate_per_sample, doubles, 0.9):.4f} "
        f"coverage_p90_d={_masked_pctl(coverage_per_sample, doubles, 0.9):.4f} "
        f"double_cov_p90_d={_masked_pctl(double_coverage_per_sample, doubles, 0.9):.4f}"
    )

    best_idx = debug_info.get("matching_best_idx")
    best_loss = debug_info.get("matching_best_loss")
    valid_blocks = debug_info.get("matching_valid_blocks")
    if best_idx is not None and valid_blocks is not None:
        best_idx = best_idx.detach().cpu()
        valid_blocks = valid_blocks.detach().cpu()
        if best_loss is not None:
            best_loss = best_loss.detach().cpu()

        block2_valid = doubles & valid_blocks[:, 1]
        if block2_valid.any():
            block2_assign = best_idx[block2_valid, 1]
            block2_to_block1 = float((block2_assign == 0).float().mean().item())
            block2_to_block2 = float((block2_assign == 1).float().mean().item())
            block2_best_loss = float(best_loss[block2_valid, 1].mean().item()) if best_loss is not None else float("nan")
            msg += (
                f" block2_assign_to1={block2_to_block1:.3f}"
                f" block2_assign_to2={block2_to_block2:.3f}"
                f" block2_best_loss={block2_best_loss:.4f}"
            )

    if extra_metrics:
        extra_str = " ".join(f"{key}={value}" for key, value in extra_metrics.items())
        msg += f" {extra_str}"

    tqdm.write(msg)


def _run_loss_controlled_experiment(
        *,
        loss_fn_seq2seq: nn.Module,
        out_seq2seq: torch.Tensor,
        targets_seq2seq: torch.Tensor,
        gold_num_codes: torch.Tensor,
        output_path: str,
        ) -> None:
    global _LOSS_DIAGNOSTICS_RAN
    if _LOSS_DIAGNOSTICS_RAN:
        return
    _LOSS_DIAGNOSTICS_RAN = True

    device = out_seq2seq.device
    batch_size = out_seq2seq.size(0)
    vocab_size = out_seq2seq.size(-1)
    seq_len = out_seq2seq.size(1)
    block_size = loss_fn_seq2seq.block_size
    block2_start = block_size
    block2_end = block2_start + block_size

    def _one_hot_logits(tokens: torch.Tensor, high: float = 30.0, low: float = -30.0) -> torch.Tensor:
        logits = torch.full((tokens.size(0), tokens.size(1), vocab_size), low, device=device)
        logits.scatter_(2, tokens.unsqueeze(-1), high)
        pad_logits = torch.full((tokens.size(0), 1, vocab_size), low, device=device)
        pad_logits[:, 0, PAD_IDX] = high
        return torch.cat([logits, pad_logits], dim=1)

    tokens = targets_seq2seq[:, 1:-1]
    tokens_only_block1 = tokens.clone()
    tokens_only_block1[:, block2_start:block2_end] = PAD_IDX

    fixed_logits = out_seq2seq.detach()
    logits_full = _one_hot_logits(tokens)
    logits_block1_only = _one_hot_logits(tokens_only_block1)

    prev_debug = getattr(loss_fn_seq2seq, "debug", False)
    loss_fn_seq2seq.debug = True

    loss_fixed_with_gold = loss_fn_seq2seq(fixed_logits, targets_seq2seq, gold_num_codes=gold_num_codes)
    debug_fixed_with_gold = loss_fn_seq2seq.last_debug
    loss_fixed_without_gold = loss_fn_seq2seq(fixed_logits, targets_seq2seq, gold_num_codes=None)
    debug_fixed_without_gold = loss_fn_seq2seq.last_debug

    loss_full_with_gold = loss_fn_seq2seq(logits_full, targets_seq2seq, gold_num_codes=gold_num_codes)
    debug_full_with_gold = loss_fn_seq2seq.last_debug
    loss_full_without_gold = loss_fn_seq2seq(logits_full, targets_seq2seq, gold_num_codes=None)
    debug_full_without_gold = loss_fn_seq2seq.last_debug

    loss_block1_with_gold = loss_fn_seq2seq(logits_block1_only, targets_seq2seq, gold_num_codes=gold_num_codes)
    debug_block1_with_gold = loss_fn_seq2seq.last_debug
    loss_block1_without_gold = loss_fn_seq2seq(logits_block1_only, targets_seq2seq, gold_num_codes=None)
    debug_block1_without_gold = loss_fn_seq2seq.last_debug

    loss_fn_seq2seq.debug = prev_debug

    def _fmt_loss(value: torch.Tensor) -> str:
        return f"{value.item():.6f}"

    def _debug_summary(debug: dict | None) -> dict:
        if debug is None:
            return {}
        return {
            "order_invariant": float(debug["order_invariant_loss"].item()),
            "push_to_pad": float(debug["push_to_pad_loss"].item()),
            "gate": float(debug["gate_loss"].item()) if debug.get("gate_loss") is not None else 0.0,
            "coverage": float(debug["coverage_loss"].item()) if debug.get("coverage_loss") is not None else 0.0,
            "double_coverage": float(debug["double_coverage_loss"].item()) if debug.get("double_coverage_loss") is not None else 0.0,
        }

    report = [
        "# Loss Controlled Experiment",
        "",
        f"- Batch size: {batch_size}",
        f"- Seq len (no BOS/EOS): {seq_len}",
        f"- Gold num codes distribution: {gold_num_codes.detach().cpu().tolist()}",
        "",
        "## Fixed model outputs (detached logits)",
        "",
        "| Setting | Total loss | Order-invariant | Push-to-pad | Gate | Coverage | Double-coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for label, loss_value, debug in [
        ("gold_num_codes", loss_fixed_with_gold, debug_fixed_with_gold),
        ("no_gold_num_codes", loss_fixed_without_gold, debug_fixed_without_gold),
    ]:
        summary = _debug_summary(debug)
        report.append(
            f"| {label} | {_fmt_loss(loss_value)} | {summary.get('order_invariant', float('nan')):.6f} | "
            f"{summary.get('push_to_pad', float('nan')):.6f} | {summary.get('gate', float('nan')):.6f} | "
            f"{summary.get('coverage', float('nan')):.6f} | {summary.get('double_coverage', float('nan')):.6f} |"
        )

    report.extend(
        [
            "",
            "## Crafted predictions",
            "",
            "| Prediction | gold_num_codes | Total loss | Order-invariant | Push-to-pad | Gate | Coverage | Double-coverage |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    crafted_rows = [
        ("both_blocks_correct", "gold_num_codes", loss_full_with_gold, debug_full_with_gold),
        ("both_blocks_correct", "no_gold_num_codes", loss_full_without_gold, debug_full_without_gold),
        ("block1_only_pad_block2", "gold_num_codes", loss_block1_with_gold, debug_block1_with_gold),
        ("block1_only_pad_block2", "no_gold_num_codes", loss_block1_without_gold, debug_block1_without_gold),
    ]
    for label, setting, loss_value, debug in crafted_rows:
        summary = _debug_summary(debug)
        report.append(
            f"| {label} | {setting} | {_fmt_loss(loss_value)} | {summary.get('order_invariant', float('nan')):.6f} | "
            f"{summary.get('push_to_pad', float('nan')):.6f} | {summary.get('gate', float('nan')):.6f} | "
            f"{summary.get('coverage', float('nan')):.6f} | {summary.get('double_coverage', float('nan')):.6f} |"
        )

    report_text = "\n".join(report) + "\n"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)


def _run_loss_nan_audit(
        *,
        debug_info: dict | None,
        gold_num_codes: torch.Tensor,
        step: int,
        prefix: str = "LOSS_NAN_AUDIT",
        ) -> None:
    if debug_info is None:
        return
    gold_num_codes = gold_num_codes.detach().cpu()
    singles = gold_num_codes == 1
    doubles = gold_num_codes >= 2

    def _count(mask: torch.Tensor) -> int:
        return int(mask.sum().item())

    def _finite_stats(values: torch.Tensor | None) -> tuple[int, int]:
        if values is None:
            return 0, 0
        values = values.detach()
        return int(values.numel()), int(torch.isfinite(values).sum().item())

    gate_per_sample = debug_info.get("gate_loss_per_sample")
    coverage_per_sample = debug_info.get("coverage_loss_per_sample")

    gate_numel, gate_finite = _finite_stats(gate_per_sample)
    cov_numel, cov_finite = _finite_stats(coverage_per_sample)
    gate_pos_count = _count(doubles)
    gate_neg_count = _count(singles)
    gate_nonfinite_s = 0
    gate_nonfinite_d = 0
    cov_nonfinite_s = 0
    cov_nonfinite_d = 0
    if gate_per_sample is not None:
        gate_vals = gate_per_sample.detach().cpu()
        gate_nonfinite_s = int((~torch.isfinite(gate_vals[singles])).sum().item()) if singles.any() else 0
        gate_nonfinite_d = int((~torch.isfinite(gate_vals[doubles])).sum().item()) if doubles.any() else 0
    if coverage_per_sample is not None:
        cov_vals = coverage_per_sample.detach().cpu()
        cov_nonfinite_s = int((~torch.isfinite(cov_vals[singles])).sum().item()) if singles.any() else 0
        cov_nonfinite_d = int((~torch.isfinite(cov_vals[doubles])).sum().item()) if doubles.any() else 0

    msg = (
        f"{prefix} step={step} "
        f"gold_counts_s={_count(singles)} gold_counts_d={_count(doubles)} "
        f"gate_pos_count={gate_pos_count} gate_neg_count={gate_neg_count} "
        f"gate_numel={gate_numel} gate_finite={gate_finite} "
        f"gate_nonfinite_s={gate_nonfinite_s} gate_nonfinite_d={gate_nonfinite_d} "
        f"coverage_numel={cov_numel} coverage_finite={cov_finite} "
        f"coverage_nonfinite_s={cov_nonfinite_s} coverage_nonfinite_d={cov_nonfinite_d}"
    )
    tqdm.write(msg)


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


def _normalize_batch_schedule(
        batch_sizes: list[int] | None,
        batch_steps: list[int] | None,
        start_step: int | None,
        lr_mults: list[float] | None,
        current_global_batch: int,
        world_size: int,
        is_main_process: bool,
        ) -> dict | None:
    if batch_sizes is None and batch_steps is None and start_step is None:
        return None
    if batch_sizes is None:
        raise ValueError("late_phase_batch_sizes must be set when enabling batch scaling.")

    batch_sizes = [int(size) for size in batch_sizes]
    prepended_current_batch = False
    if batch_sizes[0] != current_global_batch:
        if is_main_process:
            print(
                "Late-phase batch scaling: prepending current global batch size "
                f"{current_global_batch} to schedule {batch_sizes}."
            )
        batch_sizes = [current_global_batch] + batch_sizes
        prepended_current_batch = True

    # Auto-correct non-divisible batch sizes by rounding down
    corrected_batch_sizes = []
    for size in batch_sizes:
        if size % world_size != 0:
            corrected_size = (size // world_size) * world_size
            if corrected_size <= 0:
                raise ValueError(
                    f"Batch size {size} is too small for world_size {world_size}. "
                    f"After rounding down, batch size would be {corrected_size}. "
                    f"Minimum batch size should be at least {world_size}."
                )
            if is_main_process:
                print(
                    f"Warning: Batch size {size} is not divisible by world_size {world_size}. "
                    f"Rounding down to {corrected_size}."
                )
            corrected_batch_sizes.append(corrected_size)
        else:
            corrected_batch_sizes.append(size)
    batch_sizes = corrected_batch_sizes

    if batch_steps is None:
        if start_step is None:
            raise ValueError("late_phase_start_step is required when batch steps are not provided.")
        if len(batch_sizes) != 2:
            raise ValueError(
                "late_phase_batch_steps must be provided for multi-step batch schedules."
            )
        batch_steps = [int(start_step)]
    else:
        batch_steps = [int(step) for step in batch_steps]
        # If we prepended the current batch size, we need to prepend start_step to batch_steps
        if prepended_current_batch:
            if start_step is None:
                raise ValueError("late_phase_start_step is required when batch schedule is adjusted.")
            batch_steps = [int(start_step)] + batch_steps
        if len(batch_steps) != len(batch_sizes) - 1:
            raise ValueError(
                "late_phase_batch_steps must have length len(late_phase_batch_sizes) - 1."
            )
    if any(step <= 0 for step in batch_steps):
        raise ValueError("late_phase_batch_steps must be positive integers.")
    if any(next_step <= prev_step for prev_step, next_step in zip(batch_steps, batch_steps[1:])):
        raise ValueError("late_phase_batch_steps must be strictly increasing.")

    if lr_mults is None:
        lr_mults = [0.7] * (len(batch_sizes) - 1)
    else:
        lr_mults = [float(mult) for mult in lr_mults]
        # If we prepended the current batch size, we need to prepend a default lr_mult as well
        if prepended_current_batch:
            lr_mults = [0.7] + lr_mults
        if len(lr_mults) != len(batch_sizes) - 1:
            raise ValueError(
                "late_phase_lr_mults must have length len(late_phase_batch_sizes) - 1."
            )

    return {
        "batch_sizes": batch_sizes,
        "batch_steps": batch_steps,
        "lr_mults": lr_mults,
        "next_index": 1,
    }


def _apply_batch_transition(
        late_phase_state: dict,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_step: int,
        save_dir: str | None,
        log_wandb: bool,
        is_main_process: bool,
        ) -> None:
    schedule = late_phase_state["batch_schedule"]
    transition_idx = schedule["next_index"]
    if transition_idx >= len(schedule["batch_sizes"]):
        return

    old_global_batch = schedule["batch_sizes"][transition_idx - 1]
    new_global_batch = schedule["batch_sizes"][transition_idx]
    lr_mult = schedule["lr_mults"][transition_idx - 1]

    old_lr = optimizer.param_groups[0]["lr"]
    target_lrs = [group["lr"] * lr_mult for group in optimizer.param_groups]
    for group, target_lr in zip(optimizer.param_groups, target_lrs):
        group["lr"] = target_lr
    scheduler.base_lrs = list(target_lrs)
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = list(target_lrs)

    per_rank_batch = new_global_batch // late_phase_state["world_size"]
    # Note: Cannot set data_loader.batch_size directly after initialization in PyTorch
    # Instead, we modify the batch_sampler's batch_size if available
    if hasattr(data_loader, "batch_sampler") and hasattr(data_loader.batch_sampler, "batch_size"):
        data_loader.batch_sampler.batch_size = per_rank_batch

    late_phase_state["batch_size"] = per_rank_batch
    schedule["next_index"] += 1

    effective_batch = (
        per_rank_batch
        * late_phase_state["world_size"]
        * late_phase_state["grad_accum_steps"]
    )
    if is_main_process:
        tqdm.write(
            "Late-phase batch scaling transition "
            f"step={current_step} "
            f"global_batch={old_global_batch}->{new_global_batch} "
            f"per_rank_batch={per_rank_batch} "
            f"grad_accum={late_phase_state['grad_accum_steps']} "
            f"lr={old_lr:.2e}->{optimizer.param_groups[0]['lr']:.2e} "
            f"effective_batch={effective_batch}"
        )

    if save_dir is None:
        return

    update_summary(
        current_step,
        metrics={
            "batch_scale_old_global": old_global_batch,
            "batch_scale_new_global": new_global_batch,
            "batch_scale_per_rank": per_rank_batch,
            "grad_accum_steps": late_phase_state["grad_accum_steps"],
            "effective_batch": effective_batch,
            "batch_scale_lr": optimizer.param_groups[0]["lr"],
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
        constrain_to_valid_pst2: bool = True,
        valid_pst2_decode_mode: str = "trie",
        min_double_steps: int = 0,
        min_double_ratio: float = 0.0,
        debug_double_audit: bool = False,
        debug_double_audit_every: int = 200,
        debug_double_audit_samples: int = 5,
        debug_double_assert_min_ratio: float | None = None,
        debug_double_audit_info: dict | None = None,
        late_phase_state: dict | None = None,
        use_gold_num_codes_loss: bool = False,
        loss_debug_every: int = 0,
        ) -> int:
    model = model.train()

    last_step = len(data_loader) - 1
    losses = Averager()
    batch_time = Averager()
    batch_time_data = Averager()
    samples_per_sec = Averager()
    grad_accum_steps = 1 if late_phase_state is None else late_phase_state["grad_accum_steps"]
    accum_counter = 0
    optimizer_step_count = 0
    startup_audit_logged = False
    
    # Check GPU availability once
    has_cuda = torch.cuda.is_available()

    # Need to initialize first "end time", as this is
    # calculated at bottom of batch loop
    end = time.time()
    
    # Use tqdm progress bar only on rank 0
    iterator = tqdm(data_loader, disable=not is_main_process, ncols=100, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(iterator):
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

        if debug_double_audit and is_main_process and not startup_audit_logged:
            _print_debug_double_startup(debug_double_audit_info)
            startup_audit_logged = True

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

        will_step = (accum_counter + 1) == grad_accum_steps
        next_optimizer_step = optimizer_step_count + 1 if will_step else optimizer_step_count
        do_debug_audit = (
            debug_double_audit
            and is_main_process
            and will_step
            and debug_double_audit_every > 0
            and (next_optimizer_step % debug_double_audit_every == 0)
        )
        if do_debug_audit:
            _debug_double_audit_batch(
                batch=batch,
                dataset=data_loader.dataset,
                step=current_step,
                optimizer_step=next_optimizer_step,
                min_double_ratio=min_double_ratio,
                min_double_steps=min_double_steps,
                debug_samples=debug_double_audit_samples,
                debug_assert_min_ratio=debug_double_assert_min_ratio,
            )
            # TODO: Thread order_invariant_loss/push_to_pad_loss/gate_loss components for singles vs doubles.

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets_seq2seq = batch['targets_seq2seq'].to(device, non_blocking=True)
        targets_linear = batch['targets_linear'].to(device, non_blocking=True)
        gold_num_codes = batch['gold_num_codes'].to(device, non_blocking=True)

        if os.getenv("DEBUG_BLOCK2") == "1" and is_main_process:
            block_size = loss_fn.loss_fn_seq2seq.block_size
            block2_start = 1 + block_size
            block2_end = block2_start + block_size
            block2_targets = targets_seq2seq[:, block2_start:block2_end]
            gold_has2 = gold_num_codes >= 2
            block2_all_pad = (block2_targets == PAD_IDX).all(dim=1)
            doubles_ratio = float(gold_has2.float().mean().item())
            if gold_has2.any():
                block2_all_pad_doubles = float(block2_all_pad[gold_has2].float().mean().item())
            else:
                block2_all_pad_doubles = float("nan")
            block2_all_pad_overall = float(block2_all_pad.float().mean().item())
            if current_step <= min_double_steps or current_step % log_interval == 0:
                tqdm.write(
                    "DEBUG_BLOCK2 "
                    f"step={current_step} "
                    f"doubles_ratio={doubles_ratio:.3f} "
                    f"block2_all_pad_overall={block2_all_pad_overall:.3f} "
                    f"block2_all_pad_doubles={block2_all_pad_doubles:.3f}"
                )

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
                    gold_num_codes=gold_num_codes if use_gold_num_codes_loss else None,
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
                gold_num_codes=gold_num_codes if use_gold_num_codes_loss else None,
                )
        losses.update(loss.item(), out_seq2seq.size(0))
        if loss_debug_every > 0 and is_main_process and current_step % loss_debug_every == 0:
            debug_info = loss_fn.loss_fn_seq2seq.last_debug
            block2_start = loss_fn.loss_fn_seq2seq.block_size
            doubles_mask = gold_num_codes >= 2
            pred_tokens = out_seq2seq.argmax(dim=-1)
            block2_tokens = pred_tokens[:, block2_start:block2_start + loss_fn.loss_fn_seq2seq.block_size]
            pred_has2 = (block2_tokens != PAD_IDX).any(dim=1)
            pad_inside_block_mask = (block2_tokens[:, 1:] == PAD_IDX).any(dim=1)
            if doubles_mask.any():
                block2_nonpad_rate = float(pred_has2[doubles_mask].float().mean().item())
                pad_probs = torch.softmax(out_seq2seq[:, block2_start, :], dim=-1)[:, PAD_IDX]
                pad_start_prob = float(pad_probs[doubles_mask].mean().item())
                pad_probs_block2 = torch.softmax(out_seq2seq[:, block2_start:block2_start + loss_fn.loss_fn_seq2seq.block_size, :], dim=-1)[..., PAD_IDX]
                p_pad_block2_mean_d = float(pad_probs_block2[doubles_mask].mean().item())
                pad_inside_pred_has2 = float(pad_inside_block_mask[pred_has2].float().mean().item()) if pred_has2.any() else float("nan")
                pad_inside_gold_has2 = float(pad_inside_block_mask[doubles_mask].float().mean().item())
            else:
                block2_nonpad_rate = float("nan")
                pad_start_prob = float("nan")
                p_pad_block2_mean_d = float("nan")
                pad_inside_pred_has2 = float("nan")
                pad_inside_gold_has2 = float("nan")

            gate_target_pos = 1 + block2_start
            gate_target = targets_seq2seq[:, gate_target_pos]
            gate_target_pad_d = int(((gate_target == PAD_IDX) & doubles_mask).sum().item())
            gate_target_nonpad_d = int(((gate_target != PAD_IDX) & doubles_mask).sum().item())

            _log_loss_debug(
                debug_info=debug_info,
                gold_num_codes=gold_num_codes,
                step=current_step,
                extra_metrics={
                    "block2_nonpad_rate_d": f"{block2_nonpad_rate:.3f}",
                    "p_pad_block2_start_d": f"{pad_start_prob:.3f}",
                    "p_pad_block2_mean_d": f"{p_pad_block2_mean_d:.3f}",
                    "pad_inside_block|pred_has2": f"{pad_inside_pred_has2:.3f}",
                    "pad_inside_block|gold_has2": f"{pad_inside_gold_has2:.3f}",
                    "gate_target_pad_d": gate_target_pad_d,
                    "gate_target_nonpad_d": gate_target_nonpad_d,
                },
            )
            _run_loss_nan_audit(
                debug_info=debug_info,
                gold_num_codes=gold_num_codes,
                step=current_step,
            )

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
            optimizer_step_count += 1

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

            if late_phase_state is not None and late_phase_state.get("batch_schedule"):
                schedule = late_phase_state["batch_schedule"]
                if schedule["next_index"] < len(schedule["batch_sizes"]):
                    next_step = schedule["batch_steps"][schedule["next_index"] - 1]
                    should_transition = current_step >= next_step
                    if distributed and dist.is_available() and dist.is_initialized():
                        transition_tensor = torch.tensor(
                            1 if (should_transition and is_main_process) else 0,
                            device=device,
                        )
                        ddp_broadcast(transition_tensor, "batch_scale", current_step, device)
                        should_transition = bool(transition_tensor.item())
                    if should_transition:
                        _apply_batch_transition(
                            late_phase_state=late_phase_state,
                            data_loader=data_loader,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            current_step=current_step,
                            save_dir=save_dir,
                            log_wandb=log_wandb,
                            is_main_process=is_main_process,
                        )
                        last_step = len(data_loader) - 1
                        if is_main_process:
                            iterator.total = len(data_loader)
                            iterator.refresh()

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
        if eval_interval is not None and distributed and dist.is_available() and dist.is_initialized():
            eval_tensor = torch.tensor(1 if is_eval_step and is_main_process else 0, device=device)
            ddp_broadcast(eval_tensor, "eval_flag", current_step, device)
            is_eval_step = bool(eval_tensor.item())
        if is_eval_step and distributed and dist.is_available() and dist.is_initialized():
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
                try:
                    if is_main_process:
                        tqdm.write('\n' + '='*80)
                        tqdm.write('Starting evaluation pass...')
                    compute_gating_metrics = (
                        late_phase_state is not None
                        and late_phase_state.get("gate_switch_enabled", False)
                        and is_main_process
                    )
                    eval_loss, eval_loss_linear, eval_loss_seq2seq, eval_seq_acc, eval_token_acc, eval_flat_acc, gating_metrics, lang_metrics = evaluate(
                        model=model,
                        data_loader=data_loader_eval,
                        loss_fn=loss_fn,
                        device=device,
                        disallow_pad_inside_block=disallow_pad_inside_block,
                        disallow_zero_at_block_start=disallow_zero_at_block_start,
                        constrain_to_valid_pst2=constrain_to_valid_pst2,
                        valid_pst2_decode_mode=valid_pst2_decode_mode,
                        compute_gating_metrics=compute_gating_metrics,
                        require_gold_num_codes=compute_gating_metrics,
                        run_probe=False,
                        log_interval=log_interval if is_main_process else max(log_interval, 1_000_000),
                    )
                    model.train()

                    if is_main_process:
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
                        
                        # Print language-specific metrics
                        if lang_metrics:
                            tqdm.write('-'*80)
                            tqdm.write('LANGUAGE-SPECIFIC METRICS:')
                            langs = sorted(set(k.split('_')[-1] for k in lang_metrics.keys() if k.startswith('seq_acc_')))
                            for lang in langs:
                                seq_key = f'seq_acc_{lang}'
                                token_key = f'token_acc_{lang}'
                                count_key = f'count_{lang}'
                                if seq_key in lang_metrics and token_key in lang_metrics:
                                    tqdm.write(f'  {lang:>6s}: Seq Acc: {lang_metrics[seq_key]:.2f}% | Token Acc: {lang_metrics[token_key]:.2f}% | Count: {lang_metrics.get(count_key, 0)}')
                        
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
                        if (
                            late_phase_state is not None
                            and late_phase_state.get("gate_switch_enabled", False)
                            and gating_summary
                        ):
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
                                **lang_metrics,
                            },
                            filename=os.path.join(save_dir, 'logs.csv'),
                            log_wandb=log_wandb,
                        )
                except Exception as exc:
                    eval_error = exc
                if is_main_process:
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
                ddp_sync_point("post_eval", current_step, device)

            eval_failed = torch.tensor(
                1 if (eval_error is not None or probe_error is not None) else 0,
                device=device,
                dtype=torch.float32,
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
                tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
                ddp_broadcast(tensor, f"metric:{name}", current_step, device)
            
            # Broadcast language-specific metrics
            for lang_key, lang_value in lang_metrics.items():
                tensor = torch.tensor(float(lang_value), device=device, dtype=torch.float32)
                ddp_broadcast(tensor, f"metric:{lang_key}", current_step, device)
            
            if eval_failed.item() == 1:
                if eval_error is not None:
                    raise eval_error
                if probe_error is not None:
                    raise probe_error
                raise RuntimeError("Eval/probe failed on rank0; aborting on all ranks.")

            switch_tensor = torch.tensor(
                1
                if (
                    late_phase_state is not None
                    and late_phase_state.get("gate_switch_enabled", False)
                    and late_phase_state["pending_switch"]
                )
                else 0,
                device=device,
            )
            ddp_broadcast(switch_tensor, "switch_flag", current_step, device)
            if late_phase_state is not None and switch_tensor.item() == 1:
                late_phase_state["pending_switch"] = True
        elif is_eval_step and is_main_process:
            tqdm.write('\n' + '='*80)
            tqdm.write('Starting evaluation pass...')
            compute_gating_metrics = (
                late_phase_state is not None
                and late_phase_state.get("gate_switch_enabled", False)
            )
            eval_loss, eval_loss_linear, eval_loss_seq2seq, eval_seq_acc, eval_token_acc, eval_flat_acc, gating_metrics, lang_metrics = evaluate(
                model=model,
                data_loader=data_loader_eval,
                loss_fn=loss_fn,
                device=device,
                disallow_pad_inside_block=disallow_pad_inside_block,
                disallow_zero_at_block_start=disallow_zero_at_block_start,
                constrain_to_valid_pst2=constrain_to_valid_pst2,
                valid_pst2_decode_mode=valid_pst2_decode_mode,
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
            
            # Print language-specific metrics
            if lang_metrics:
                tqdm.write('-'*80)
                tqdm.write('LANGUAGE-SPECIFIC METRICS:')
                langs = sorted(set(k.split('_')[-1] for k in lang_metrics.keys() if k.startswith('seq_acc_')))
                for lang in langs:
                    seq_key = f'seq_acc_{lang}'
                    token_key = f'token_acc_{lang}'
                    count_key = f'count_{lang}'
                    if seq_key in lang_metrics and token_key in lang_metrics:
                        tqdm.write(f'  {lang:>6s}: Seq Acc: {lang_metrics[seq_key]:.2f}% | Token Acc: {lang_metrics[token_key]:.2f}% | Count: {lang_metrics.get(count_key, 0)}')
            
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
            if (
                late_phase_state is not None
                and late_phase_state.get("gate_switch_enabled", False)
                and gating_summary
            ):
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
                    **lang_metrics,
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
        if late_phase_state is not None and late_phase_state.get("batch_schedule"):
            schedule = late_phase_state["batch_schedule"]
            if schedule["next_index"] < len(schedule["batch_sizes"]):
                next_step = schedule["batch_steps"][schedule["next_index"] - 1]
                should_transition = current_step >= next_step
                if distributed and dist.is_available() and dist.is_initialized():
                    transition_tensor = torch.tensor(
                        1 if (should_transition and is_main_process) else 0,
                        device=device,
                    )
                    ddp_broadcast(transition_tensor, "batch_scale", current_step, device)
                    should_transition = bool(transition_tensor.item())
                if should_transition:
                    _apply_batch_transition(
                        late_phase_state=late_phase_state,
                        data_loader=data_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        current_step=current_step,
                        save_dir=save_dir,
                        log_wandb=log_wandb,
                        is_main_process=is_main_process,
                    )

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
        constrain_to_valid_pst2: bool = True,
        valid_pst2_decode_mode: str = 'trie',
        require_gold_num_codes: bool = False,
        compute_gating_metrics: bool = False,
        run_probe: bool = True,
    ):
    model = model.eval()
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
    gate_lang_confusion: dict[str, dict[str, int]] = {}
    gate_gold1_fp = 0
    gate_gold1_total = 0
    gate_gold2_tp = 0
    gate_gold2_total = 0
    block2_nonpad_rate = Averager()
    block2_pad_start_rate = Averager()
    formatter = getattr(data_loader.dataset, "formatter", None)
    valid_block_token_ids = _build_valid_block_token_ids_from_dataset(data_loader.dataset) if constrain_to_valid_pst2 and valid_pst2_decode_mode == 'trie' else None

    if hasattr(data_loader.dataset, "frame") and "pst2_2" in data_loader.dataset.frame.columns:
        first_batch_size = getattr(data_loader, "batch_size", 32) or 32
        raw_vals = data_loader.dataset.frame["pst2_2"].head(first_batch_size)
        cleaned_vals = raw_vals.map(clean_target_value)
        raw_counts = Counter("none" if v is None else ("float_nan" if isinstance(v, float) else "str") for v in raw_vals.tolist())
        cleaned_counts = Counter("none" if v is None else "str" for v in cleaned_vals.tolist())
        tqdm.write(f"  Eval PST2_2 first-batch dist raw={dict(raw_counts)} cleaned={dict(cleaned_counts)}")
        if any(isinstance(v, str) and v.lower() == "nan" for v in cleaned_vals.tolist()):
            raise AssertionError("cleaned pst2_2 contains literal 'nan' string")

    # Language-specific metrics tracking
    lang_token_accs = {}
    lang_seq_accs = {}
    lang_counts = {}

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

        if os.getenv("METRIC_SANITY") == "1" and batch_idx == 0 and formatter is not None and getattr(data_loader.dataset, "map_code_label", None) is not None:
            inv_key = data_loader.dataset.map_code_label
            use_within_block_sep = bool(getattr(formatter, 'within_block_sep', None))
            pred_with_bos = torch.cat([
                torch.full((out_seq2seq.size(0), 1), BOS_IDX, device=out_seq2seq.device, dtype=torch.long),
                out_seq2seq.argmax(dim=-1),
            ], dim=1)
            shown = 0
            for i in range(out_seq2seq.size(0)):
                sacc_i, tacc_i = order_invariant_accuracy(
                    output=out_seq2seq[i:i+1],
                    target=targets_seq2seq[i:i+1, 1:],
                    pad_idx=PAD_IDX,
                    nb_blocks=loss_fn.loss_fn_seq2seq.nb_blocks,
                    block_size=loss_fn.loss_fn_seq2seq.block_size,
                )
                if tacc_i.item() >= 90.0 and sacc_i.item() < 100.0:
                    pred_blocks = _extract_normalized_blocks_from_seq(pred_with_bos[i].detach().cpu().numpy(), formatter, inv_key, use_within_block_sep)
                    gold_blocks = _extract_normalized_blocks_from_seq(targets_seq2seq[i].detach().cpu().numpy(), formatter, inv_key, use_within_block_sep)
                    tqdm.write(f"  METRIC_SANITY idx={i} token_acc={tacc_i.item():.2f} seq_acc={sacc_i.item():.2f} flat_match={Counter(pred_blocks)==Counter(gold_blocks)} pred={pred_blocks} gold={gold_blocks}")
                    shown += 1
                    if shown >= 5:
                        break

        if gold_num_codes is not None:
            pred_tokens = out_seq2seq.argmax(dim=-1)
            block2_start = loss_fn.loss_fn_seq2seq.block_size
            block2_end = block2_start + loss_fn.loss_fn_seq2seq.block_size
            pred_block2_tokens = pred_tokens[:, block2_start:block2_end]
            pred_block2_nonpad = (pred_block2_tokens != PAD_IDX).any(dim=1)
            gold_has2 = gold_num_codes >= 2
            if gold_has2.any():
                block2_nonpad_rate.update(pred_block2_nonpad[gold_has2].float().mean().item(), gold_has2.sum().item())
                block2_pad_start = pred_tokens[:, block2_start] == PAD_IDX
                block2_pad_start_rate.update(block2_pad_start[gold_has2].float().mean().item(), gold_has2.sum().item())

            gate_gold1_total += int((gold_num_codes == 1).sum().item())
            gate_gold1_fp += int((pred_block2_nonpad & (gold_num_codes == 1)).sum().item())
            gate_gold2_total += int((gold_has2).sum().item())
            gate_gold2_tp += int((pred_block2_nonpad & gold_has2).sum().item())
        
        # Track language-specific accuracies
        langs = batch.get('lang', None)
        if langs is not None:
            # If langs is a list of strings (not batched into a tensor)
            for i, lang in enumerate(langs):
                if lang not in lang_token_accs:
                    lang_token_accs[lang] = Averager()
                    lang_seq_accs[lang] = Averager()
                    lang_counts[lang] = 0
                
                # Compute per-sample accuracy
                sample_seq_acc, sample_token_acc = order_invariant_accuracy(
                    output=out_seq2seq[i:i+1],
                    target=targets_seq2seq[i:i+1, 1:],
                    pad_idx=PAD_IDX,
                    nb_blocks=loss_fn.loss_fn_seq2seq.nb_blocks,
                    block_size=loss_fn.loss_fn_seq2seq.block_size,
                )
                lang_token_accs[lang].update(sample_token_acc.item(), 1)
                lang_seq_accs[lang].update(sample_seq_acc.item(), 1)
                lang_counts[lang] += 1

            if gold_num_codes is not None:
                for i, lang in enumerate(langs):
                    if lang not in gate_lang_confusion:
                        gate_lang_confusion[lang] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                    gold_has2_lang = bool(gold_num_codes[i].item() >= 2)
                    pred_has2_lang = bool(pred_block2_nonpad[i].item())
                    if pred_has2_lang and gold_has2_lang:
                        gate_lang_confusion[lang]["tp"] += 1
                    elif pred_has2_lang and not gold_has2_lang:
                        gate_lang_confusion[lang]["fp"] += 1
                    elif (not pred_has2_lang) and gold_has2_lang:
                        gate_lang_confusion[lang]["fn"] += 1
                    else:
                        gate_lang_confusion[lang]["tn"] += 1

        # Flat accuracy (seq2seq): first two blocks, order-invariant, normalized
        if formatter is not None and getattr(data_loader.dataset, "map_code_label", None) is not None:
            inv_key = data_loader.dataset.map_code_label
            use_within_block_sep = bool(getattr(formatter, 'within_block_sep', None))
            pred_with_bos = torch.cat(
                [
                    torch.full((out_seq2seq.size(0), 1), BOS_IDX, device=out_seq2seq.device, dtype=torch.long),
                    out_seq2seq.argmax(dim=-1),
                ],
                dim=1,
            )
            acc_flat = _flat_accuracy_from_seq2seq(
                pred_with_bos,
                targets_seq2seq,
                formatter,
                inv_key,
                use_within_block_sep,
            )
            flat_accs.update(acc_flat, out_seq2seq.size(0))
        else:
            preds_linear = torch.sigmoid(out_linear) > 0.5
            preds_linear = preds_linear.float().cpu()
            acc_flat = accuracy_score(preds_linear, targets_linear.cpu()) * 100.0
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
                constrain_to_valid_blocks=bool(valid_block_token_ids),
                valid_block_token_ids=valid_block_token_ids,
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

        if os.getenv("LOSS_DIAGNOSTICS") == "1" and gold_num_codes is not None:
            is_main = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
            if is_main:
                output_path = os.getenv("LOSS_DIAGNOSTICS_PATH", "loss_debug_report.md")
                _run_loss_controlled_experiment(
                    loss_fn_seq2seq=loss_fn.loss_fn_seq2seq,
                    out_seq2seq=out_seq2seq,
                    targets_seq2seq=targets_seq2seq,
                    gold_num_codes=gold_num_codes,
                    output_path=output_path,
                )

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
            constrain_to_valid_pst2=constrain_to_valid_pst2,
            valid_pst2_decode_mode=valid_pst2_decode_mode,
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
    
    # Build language-specific metrics dictionary
    lang_metrics = {}
    for lang in sorted(lang_token_accs.keys()):
        lang_metrics[f'seq_acc_{lang}'] = lang_seq_accs[lang].avg
        lang_metrics[f'token_acc_{lang}'] = lang_token_accs[lang].avg
        lang_metrics[f'count_{lang}'] = lang_counts[lang]

    if block2_nonpad_rate.count > 0:
        lang_metrics["block2_nonpad_rate_gold2"] = block2_nonpad_rate.avg
    if block2_pad_start_rate.count > 0:
        lang_metrics["block2_pad_start_rate_gold2"] = block2_pad_start_rate.avg
    if gate_gold1_total > 0:
        lang_metrics["gate_fp_rate_gold1"] = gate_gold1_fp / gate_gold1_total
    if gate_gold2_total > 0:
        lang_metrics["gate_recall_gold2"] = gate_gold2_tp / gate_gold2_total

    for lang, conf in sorted(gate_lang_confusion.items()):
        tp = conf["tp"]
        fp = conf["fp"]
        fn = conf["fn"]
        precision_lang = tp / (tp + fp) if (tp + fp) else 0.0
        recall_lang = tp / (tp + fn) if (tp + fn) else 0.0
        f1_lang = 2 * precision_lang * recall_lang / (precision_lang + recall_lang) if (precision_lang + recall_lang) else 0.0
        lang_metrics[f"gate_precision_{lang}"] = precision_lang
        lang_metrics[f"gate_recall_{lang}"] = recall_lang
        lang_metrics[f"gate_f1_{lang}"] = f1_lang

    return losses.avg, losses_linear.avg, losses_seq2seq.avg, seq_accs.avg, token_accs.avg, flat_accs.avg, gating_metrics, lang_metrics


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
    normalized = clean_target_value(value)
    return normalized not in {None, '?'}


def _truncate_text(value: str | None, limit: int = 120) -> str:
    if value is None:
        return "None"
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _print_debug_double_startup(info: dict | None) -> None:
    details = info or {}
    drop_bad_labels = details.get("drop_bad_labels", False)
    use_within_block_sep = details.get("use_within_block_sep", False)
    print(
        "DEBUG_DBL_AUDIT_STARTUP "
        "pst2_2_missing_detection=(_pst2_value_present: None/float->missing; "
        "strings lower() vs {'', ' ', '?', 'nan', 'none', 'null'}). "
        "prepare_target_cols replaces {' ', 'nan', 'NaN', 'NAN', 'none', 'None', 'null', 'NULL'} with None; "
        "first target col fills missing with '?'. "
        "_get_gold_num_codes strips() and treats '', ' ', '?' as missing. "
        f"use_within_block_sep={use_within_block_sep} "
        f"drop_bad_labels={drop_bad_labels} (drops entire rows failing formatter; does not surgically remove block2)."
    )


def _debug_double_audit_batch(
        batch: dict,
        dataset,
        *,
        step: int,
        optimizer_step: int,
        min_double_ratio: float,
        min_double_steps: int,
        debug_samples: int,
        debug_assert_min_ratio: float | None,
        ) -> None:
    raw_pst2_2 = batch.get("raw_pst2_2")
    serialized_targets = batch.get("serialized_target")
    if raw_pst2_2 is None or serialized_targets is None:
        logged = getattr(_debug_double_audit_batch, "_logged_missing", False)
        if not logged:
            print(
                "DEBUG_DBL_AUDIT_MISSING "
                "raw_pst2_2/serialized_target not found in batch; "
                "enable dataset.debug_double_audit to include raw fields."
            )
            _debug_double_audit_batch._logged_missing = True
        return

    raw_pst2_1 = batch.get("raw_pst2_1")
    raw_occ1 = batch.get("raw_occ1", batch.get("occ1"))
    langs = batch.get("lang", batch.get("raw_lang"))

    sep_value = getattr(getattr(dataset, "formatter", None), "sep_value", "&")
    max_num_codes = getattr(getattr(dataset, "formatter", None), "max_num_codes", None)
    if max_num_codes is None:
        max_num_codes = len(getattr(dataset, "target_cols", []) or [])

    gold_num_codes = batch.get("gold_num_codes")
    if gold_num_codes is not None and torch.is_tensor(gold_num_codes):
        gold_num_codes_list = gold_num_codes.cpu().tolist()
    else:
        gold_num_codes_list = None

    batch_size = len(serialized_targets)
    has2_raw = []
    has2_serial = []
    gold_ge2 = []
    gold_num_codes_est = []

    for idx in range(batch_size):
        raw_val = raw_pst2_2[idx]
        has2_raw_val = _pst2_value_present(raw_val)
        has2_raw.append(has2_raw_val)

        serialized = serialized_targets[idx]
        if serialized is None or sep_value == "":
            sep_count = 0
        else:
            sep_count = str(serialized).count(sep_value)
        has2_serial_val = sep_count >= 1
        has2_serial.append(has2_serial_val)
        if serialized is None:
            gold_num_codes_est.append(0)
        else:
            gold_num_codes_est.append(min(1 + sep_count, max_num_codes))

        if gold_num_codes_list is not None:
            gold_ge2.append(gold_num_codes_list[idx] >= 2)

    if gold_num_codes_list is None:
        gold_num_codes_list = gold_num_codes_est
        gold_ge2 = [val >= 2 for val in gold_num_codes_list]

    raw_count = sum(has2_raw)
    serial_count = sum(has2_serial)
    gold_count = sum(gold_ge2)

    observed_raw_ratio = raw_count / max(1, batch_size)
    observed_serial_ratio = serial_count / max(1, batch_size)
    observed_gold_ratio = gold_count / max(1, batch_size)

    lang_counts = Counter(langs) if langs is not None else {}
    print(
        "DEBUG_DBL_AUDIT "
        f"step={step} opt_step={optimizer_step} "
        f"has2_raw={observed_raw_ratio:.3f} "
        f"has2_ser={observed_serial_ratio:.3f} "
        f"gold2={observed_gold_ratio:.3f} "
        f"min_double_ratio={min_double_ratio:.3f} "
        f"lang_counts={dict(lang_counts)} "
        f"counts=raw:{raw_count} ser:{serial_count} gold2:{gold_count} total:{batch_size} "
        f"sep={sep_value!r}"
    )

    if debug_assert_min_ratio is not None and step <= min_double_steps:
        if observed_raw_ratio < debug_assert_min_ratio:
            raise RuntimeError(
                "DEBUG_DBL_ASSERT "
                f"step={step} opt_step={optimizer_step} "
                f"observed_has2_raw={observed_raw_ratio:.3f} "
                f"required_min={debug_assert_min_ratio:.3f} "
                f"min_double_steps={min_double_steps}"
            )

    mismatch_rows = []
    for idx in range(batch_size):
        raw_flag = has2_raw[idx]
        serial_flag = has2_serial[idx]
        gold_flag = gold_ge2[idx]
        if raw_flag and not gold_flag:
            reason = "raw_has2_but_gold_lt2"
        elif serial_flag and not raw_flag:
            reason = "serial_has2_but_raw_no"
        elif gold_flag and not serial_flag:
            reason = "gold_ge2_but_serial_no_sep"
        else:
            continue
        mismatch_rows.append((idx, reason))

    if mismatch_rows:
        for idx, reason in mismatch_rows[:debug_samples]:
            if isinstance(raw_occ1, (list, tuple)):
                occ1_val = raw_occ1[idx]
            else:
                occ1_val = raw_occ1
            if isinstance(langs, (list, tuple)):
                lang_val = langs[idx]
            else:
                lang_val = langs
            pst2_1_val = raw_pst2_1[idx] if raw_pst2_1 is not None else None
            pst2_2_val = raw_pst2_2[idx]
            serialized_val = serialized_targets[idx]
            gold_val = gold_num_codes_list[idx]
            token_len = None
            if "targets_seq2seq" in batch and torch.is_tensor(batch["targets_seq2seq"]):
                token_len = int((batch["targets_seq2seq"][idx] != PAD_IDX).sum().item())
            input_len = None
            if "input_ids" in batch and torch.is_tensor(batch["input_ids"]):
                input_len = int(batch["input_ids"][idx].numel())

            print(
                "DEBUG_DBL_AUDIT_SAMPLE "
                f"reason={reason} "
                f"lang={lang_val!r} "
                f"occ1={_truncate_text(occ1_val)!r} "
                f"pst2_1={pst2_1_val!r} "
                f"pst2_2={pst2_2_val!r} "
                f"serialized={serialized_val!r} "
                f"gold_num_codes={gold_val} "
                f"gold_num_codes_est={gold_num_codes_est[idx]} "
                f"seq2seq_len={token_len} "
                f"input_len={input_len}"
            )


def _split_str_s2s(pred: str, sep_value: str) -> list[str] | str:
    if sep_value and sep_value in pred:
        return pred.split(sep_value)
    return pred


def _normalize_code_for_lookup(code: str | None, inv_key: dict, use_within_block_sep: bool) -> str:
    normalized = clean_target_value(code)
    if normalized is None:
        return ""
    code = normalized.replace(' ', '')
    if not use_within_block_sep:
        return code
    if code in inv_key:
        return code
    parts = code.split(',')
    while len(parts) > 1 and parts[-1] == '0':
        parts = parts[:-1]
        candidate = ','.join(parts)
        if candidate in inv_key:
            return candidate
    return code




def _extract_normalized_blocks_from_seq(
        seq_tokens: np.ndarray,
        formatter,
        inv_key: dict,
        use_within_block_sep: bool,
) -> list[str]:
    formatted = formatter.clean_pred(seq_tokens)
    split_pred = _split_str_s2s(formatted, formatter.sep_value)
    blocks = split_pred if isinstance(split_pred, list) else [split_pred]
    blocks = [
        _normalize_code_for_lookup(block, inv_key, use_within_block_sep)
        for block in blocks[:2]
        if _normalize_code_for_lookup(block, inv_key, use_within_block_sep) != ""
    ]
    return blocks


def _flat_accuracy_from_seq2seq(
        pred_tokens: torch.Tensor,
        gold_tokens_with_bos: torch.Tensor,
        formatter,
        inv_key: dict,
        use_within_block_sep: bool,
) -> float:
    """Flat accuracy over first two blocks, order-invariant after normalization.

    - uses formatter.clean_pred + lookup normalization
    - ignores block order
    - compares only first two blocks
    """
    correct = 0
    total = pred_tokens.size(0)
    for i in range(total):
        pred_blocks = _extract_normalized_blocks_from_seq(
            pred_tokens[i].detach().cpu().numpy(), formatter, inv_key, use_within_block_sep
        )
        gold_blocks = _extract_normalized_blocks_from_seq(
            gold_tokens_with_bos[i].detach().cpu().numpy(), formatter, inv_key, use_within_block_sep
        )
        if Counter(pred_blocks) == Counter(gold_blocks):
            correct += 1
    return 100.0 * correct / max(1, total)

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


def _build_valid_block_token_ids_from_dataset(dataset) -> list[list[int]] | None:
    formatter = getattr(dataset, "formatter", None)
    inv_key = getattr(dataset, "map_code_label", None)
    if formatter is None or not inv_key:
        return None

    valid_blocks: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for code in inv_key.keys():
        try:
            encoded = formatter.transform_label(str(code))
        except Exception:
            continue
        if encoded is None:
            continue
        block = tuple(int(tok) for tok in encoded[1:1 + formatter.block_size])
        if len(block) != formatter.block_size:
            continue
        if block in seen:
            continue
        seen.add(block)
        valid_blocks.append(list(block))

    return valid_blocks or None


def _run_pst2_eval_probe(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        sample_size: int = 200,
        seed: int = 42,
        disallow_pad_inside_block: bool = False,
        disallow_zero_at_block_start: bool = False,
        constrain_to_valid_pst2: bool = True,
        valid_pst2_decode_mode: str = 'trie',
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
            constrain_to_valid_pst2=constrain_to_valid_pst2,
            valid_pst2_decode_mode=valid_pst2_decode_mode,
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
        constrain_to_valid_pst2: bool = True,
        valid_pst2_decode_mode: str = 'trie',
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

    raw_pst2_2 = dataset.frame['pst2_2'].head(sample_size)
    raw_stats = Counter('none' if v is None else ('float_nan' if isinstance(v, float) else 'str') for v in raw_pst2_2.tolist())
    cleaned_pst2_2 = raw_pst2_2.map(clean_target_value)
    cleaned_stats = Counter('none' if v is None else 'str' for v in cleaned_pst2_2.tolist())
    print(f"  debug_pst2_2_dist raw={dict(raw_stats)} cleaned={dict(cleaned_stats)}")
    if any(isinstance(v, str) and v.lower() == 'nan' for v in cleaned_pst2_2.tolist()):
        raise AssertionError("cleaned pst2_2 contains literal 'nan' string")

    cleaned_pst2_2_full = dataset.frame['pst2_2'].map(clean_target_value)
    has_second = cleaned_pst2_2_full.apply(_pst2_value_present).to_numpy()
    eligible_positions = [idx for idx, flag in enumerate(has_second) if flag]
    if not eligible_positions:
        print('PST2 eval probe: no rows with pst2_2 present in eval dataset.')
        return
    single_positions = [idx for idx, flag in enumerate(has_second) if not flag]

    inv_key = dataset.map_code_label
    use_within_block_sep = bool(getattr(formatter, 'within_block_sep', None))

    model_to_decode = model.module if hasattr(model, 'module') else model
    model_to_decode.eval()
    valid_block_token_ids = _build_valid_block_token_ids_from_dataset(dataset) if constrain_to_valid_pst2 and valid_pst2_decode_mode == 'trie' else None

    examples_a: list[_PST2ProbeRow] = []
    examples_b: list[_PST2ProbeRow] = []
    examples_c: list[_PST2ProbeRow] = []
    examples_d: list[_PST2ProbeRow] = []

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
        block1_emitted_count = 0
        block1_emitted_in_key = 0
        block2_emitted_in_key = 0
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
                constrain_to_valid_blocks=bool(valid_block_token_ids),
                valid_block_token_ids=valid_block_token_ids,
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
                if any(tok != PAD_IDX for tok in block1_tokens):
                    block1_emitted_count += 1
                    if pred_block1_in_key:
                        block1_emitted_in_key += 1
                if block2_nonpad and pred_block2_in_key:
                    block2_emitted_in_key += 1
                gold2_raw_clean = clean_target_value(record['pst2_2'])
                gold2_norm = _normalize_code_for_lookup(gold2_raw_clean, inv_key, use_within_block_sep)
                gold2_in_key = gold2_norm in inv_key if gold2_norm else False
                gold_has2 = _pst2_value_present(gold2_raw_clean)

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
                    pst2_2=str(clean_target_value(record['pst2_2'])),
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

                if block2_nonpad and pred_block2_in_key and gold_has2 and gold2_in_key and pred_block2_norm != gold2_norm and len(examples_d) < 10:
                    examples_d.append(
                        _PST2ProbeRow(
                            index=int(dataset_idx),
                            occ1=_truncate_text(record.get('occ1')),
                            pst2_1=str(record.get('pst2_1')),
                            pst2_2=str(record.get('pst2_2')),
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
        if block1_emitted_count:
            print(f'  % block1_in_key | block1_emitted: {block1_emitted_in_key / block1_emitted_count:.2%}')
        if pred_has2_count:
            print(f'  % block2_in_key | pred_has2: {pred_has2_block2_in_key / pred_has2_count:.2%}')
            print(f'  % block2_in_key | block2_emitted: {block2_emitted_in_key / pred_has2_count:.2%}')
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
    _print_examples('D) block2 mismatch where pred/gold are both in_key', examples_d)
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
        constrain_to_valid_pst2: bool = True,
        valid_pst2_decode_mode: str = "trie",
        min_double_steps: int = 0,
        min_double_ratio: float = 0.0,
        debug_double_audit: bool = False,
        debug_double_audit_every: int = 200,
        debug_double_audit_samples: int = 5,
        debug_double_assert_min_ratio: float | None = None,
        debug_double_audit_info: dict | None = None,
        gate_stabilize_metric: str = "gating_f1",
        gate_stabilize_window: int = 5,
        gate_stabilize_delta: float = 0.02,
        gate_stabilize_min: float = 0.90,
        late_grad_accum: int = 1,
        late_lr_mult: float = 1.0,
        late_warmup_steps: int = 0,
        late_switch_once: bool = True,
        batch_size: int | None = None,
        late_phase_start_step: int | None = None,
        late_phase_batch_sizes: list[int] | None = None,
        late_phase_batch_steps: list[int] | None = None,
        late_phase_lr_mults: list[float] | None = None,
        use_gold_num_codes_loss: bool = False,
        ):
    # Initialize GradScaler for AMP if enabled
    scaler = GradScaler('cuda') if use_amp else None
    loss_debug_every = int(os.getenv("LOSS_DEBUG_EVERY", "0"))
    if loss_debug_every > 0 and loss_fn is not None:
        loss_fn.loss_fn_seq2seq.debug = True

    world_size = 1
    if distributed and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    elif data_loaders.get("data_loader_train") is not None:
        world_size = getattr(data_loaders["data_loader_train"].sampler, "num_replicas", 1)
    if batch_size is None:
        batch_size = data_loaders["data_loader_train"].batch_size
    current_global_batch = batch_size * world_size
    batch_schedule = _normalize_batch_schedule(
        batch_sizes=late_phase_batch_sizes,
        batch_steps=late_phase_batch_steps,
        start_step=late_phase_start_step,
        lr_mults=late_phase_lr_mults,
        current_global_batch=current_global_batch,
        world_size=world_size,
        is_main_process=is_main_process,
    )

    gate_switch_enabled = (
        late_grad_accum > 1
        or late_lr_mult != 1.0
        or late_warmup_steps > 0
    )
    enable_late_phase = gate_switch_enabled or batch_schedule is not None
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
            "gate_switch_enabled": gate_switch_enabled,
            "batch_size": batch_size,
            "world_size": world_size,
            "batch_schedule": batch_schedule,
        }
        if batch_schedule is not None and is_main_process:
            tqdm.write(
                "Late-phase batch scaling schedule "
                f"global_batches={batch_schedule['batch_sizes']} "
                f"steps={batch_schedule['batch_steps']} "
                f"lr_mults={batch_schedule['lr_mults']}"
            )
    
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
            constrain_to_valid_pst2=constrain_to_valid_pst2,
            valid_pst2_decode_mode=valid_pst2_decode_mode,
            min_double_steps=min_double_steps,
            min_double_ratio=min_double_ratio,
            debug_double_audit=debug_double_audit,
            debug_double_audit_every=debug_double_audit_every,
            debug_double_audit_samples=debug_double_audit_samples,
            debug_double_assert_min_ratio=debug_double_assert_min_ratio,
            debug_double_audit_info=debug_double_audit_info,
            late_phase_state=late_phase_state,
            use_gold_num_codes_loss=use_gold_num_codes_loss,
            loss_debug_every=loss_debug_every,
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
