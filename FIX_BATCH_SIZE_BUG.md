# Fix for Batch Size Divisibility Bug

## Problem
When running distributed training with 8 GPUs (world_size=8), the training fails with:
```
ValueError: All late_phase_batch_sizes must be divisible by world_size (world_size=8, batch_sizes=[4096, 1024, 1096, 2020]).
```

## Root Cause
The `--late-phase-batch-sizes` parameter in the SLURM script specifies batch sizes that must be divisible by the number of GPUs (world_size). This is because in distributed training, the global batch is split evenly across all GPUs.

## Analysis
Checking the specified batch sizes for divisibility by 8:
- **1024**: ✓ Divisible (128 samples per GPU)
- **1096**: ✓ Divisible (137 samples per GPU)  
- **2020**: ✗ NOT divisible (remainder=4)

## Solution
Change the batch size 2020 to either:
- **2016** (252 samples per GPU) - rounds down
- **2024** (253 samples per GPU) - rounds up ← **RECOMMENDED**

## Fixed SLURM Script Parameter
Change this line in your SLURM script:
```bash
# OLD (incorrect):
--late-phase-batch-sizes 1024 1096 2020 \

# NEW (correct):
--late-phase-batch-sizes 1024 1096 2024 \
```

This ensures all batch sizes are evenly divisible by 8 GPUs.
