# Reproducing Original histocc Training Behavior

This document explains how to run training with settings that exactly match the original histocc implementation.

## Summary of Differences

The OCCPAST implementation has **one training parameter difference** from the original histocc:

| Parameter | Original histocc | OCCPAST Default | To Match Original |
|-----------|-----------------|-----------------|-------------------|
| Warmup | 0 steps (0%) | 5% of total steps | Add `--warmup-pct 0.0` |

**All other parameters are identical**: loss functions, learning rate, batch size, optimizer, gradient clipping, etc.

## Running with Original Settings

To train with **exact original histocc behavior**, add `--warmup-pct 0.0`:

```bash
python finetune.py \
  --dataset path/to/data.csv \
  --target-cols col1 col2 col3 \
  --save-path ./Finetuned/ \
  --warmup-pct 0.0
```

## Running with Distributed Training (Recommended)

For distributed GPU training with modern best practices (includes warmup):

```bash
# Single GPU (uses default 5% warmup)
python finetune.py \
  --dataset path/to/data.csv \
  --target-cols col1 col2 col3 \
  --save-path ./Finetuned/

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 finetune.py \
  --dataset path/to/data.csv \
  --target-cols col1 col2 col3 \
  --save-path ./Finetuned/
```

## Why the Warmup Change?

Warmup (gradually increasing learning rate at the start) is a modern best practice that:
- ✅ Improves training stability
- ✅ Helps with distributed training and large batch sizes
- ✅ Reduces sensitivity to initial learning rate choice
- ⚠️ Changes the learning rate schedule slightly

### When to Use Each Setting

**Use `--warmup-pct 0.0` (original behavior) when**:
- You need exact reproduction of original experiments
- You're doing careful ablation studies
- You have a specific reason to avoid warmup

**Use `--warmup-pct 0.05` (default, recommended) when**:
- Using distributed training across multiple GPUs
- Using large batch sizes (>256)
- Following modern training best practices
- Training from scratch or with new data

## Verification

To verify your training matches the original:

1. **Check loss function**: Both implementations use `BlockOrderInvariantLoss` with identical parameters
2. **Check optimizer**: Both use AdamW with same learning rate (2e-05)
3. **Check gradient clipping**: Both use `max_norm=1.0`
4. **Check warmup**: Add `--warmup-pct 0.0` to match original

## Additional Options

The OCCPAST implementation includes these optional enhancements:

```bash
# Mixed precision training (faster, uses less memory)
python finetune.py ... --use-amp

# More data loading workers (faster data loading)
python finetune.py ... --num-workers 4

# Pin memory for faster GPU transfers
python finetune.py ... --pin-memory

# Persistent workers (faster multi-epoch training)
python finetune.py ... --persistent-workers
```

These options don't change the training algorithm, only the speed.

## Questions?

See `COMPARISON_REPORT.md` for a detailed technical analysis of all differences between the implementations.
