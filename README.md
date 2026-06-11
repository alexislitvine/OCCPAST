OCCPAST
=======

This repository extends the original OccCANINE project for OCCUPATIONSPAST.ORG workflows, including training, finetuning, prediction, and post-processing.

Original OccCANINE repository:
https://github.com/christianvedels/OccCANINE

Project Structure
-----------------

- histocc: Core package (models, dataloaders, training/eval utilities)
- tests: Test suite
- Data: Example data and keys
- predictions: Prediction inputs/outputs and lookup JSONs
- train.py, train_mixer.py, finetune.py, finetune-2.py, predict_OCCPAST.py, format_preds.py: Main runnable scripts

Requirements
------------

- Python 3.10+
- PyTorch
- pandas
- numpy
- tqdm
- scikit-learn
- pyarrow (if reading/writing parquet)

Example setup:

```bash
conda create -n occpast python=3.10 -y && conda activate occpast && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && pip install pandas numpy pyarrow tqdm scikit-learn
```

Script Quick Reference
----------------------

All example commands below are one-liners for easy copy/paste.

### train.py

Purpose: Train baseline OccCANINE flow.

Help:

```bash
python3 train.py --help
```

Example:

```bash
python3 train.py --epochs 1 --batch-size 64 --learning-rate 2e-5
```

### train_mixer.py

Purpose: Train mixer architecture variant.

Help:

```bash
python3 train_mixer.py --help
```

Example:

```bash
python3 train_mixer.py --train-data Data/TOYDATA.csv --val-data Data/TOYDATA.csv --target-col-naming hisco --target-cols hisco_1 hisco_2 --num-epochs 1 --batch-size 64
```

### finetune.py

Purpose: Finetune a model on a custom dataset. Training is step-centric:
`global_step` means optimizer update step, while `global_micro_step` means one dataloader batch per rank.

Help:

```bash
python3 finetune.py --help
```

Example:

```bash
python3 finetune.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep --max-steps 1000 --per-gpu-batch-size 64 --grad-accum-steps 1 --save-path ./Finetuned
```

Batch-size semantics:

- `per_gpu_batch_size`: samples per GPU/process per dataloader forward/backward pass.
- `world_size`: number of distributed processes/GPUs.
- `grad_accum_steps`: micro-batches accumulated before `optimizer.step()`.
- `global_micro_batch_size = per_gpu_batch_size * world_size`.
- `effective_batch_size = per_gpu_batch_size * world_size * grad_accum_steps`.

`--batch-size` is still accepted as a deprecated alias for `--per-gpu-batch-size`.

Print the resolved schedule without training:

```bash
python3 finetune.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep --max-steps 1000 --per-gpu-batch-size 64 --grad-accum-steps 1 --save-path ./Finetuned --print-training-schedule-only
```

Late-phase accumulation example:

```bash
python3 finetune.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep --max-steps 100000 --per-gpu-batch-size 512 --grad-accum-steps 1 --late-phase-start-step 50000 --late-phase-steps 75000 --late-phase-grad-accum-steps 2 4 --late-phase-lr-mults 0.7 0.7 --save-path ./Finetuned
```

All scheduling arguments (`--max-steps`, `--eval-interval`, `--save-interval`, `--late-phase-start-step`, `--late-phase-steps`, LR schedule steps, W&B step logging) use optimizer update steps. Epochs are only a backward-compatible way to derive `--max-steps` when `--max-steps` is omitted.

The script writes `training_schedule.csv` in `--save-path` and logs the same table to W&B when `--log-wandb` is enabled.

Restart behavior:

- Checkpoints are written to `<save-path>/last.bin` and `<save-path>/<global_step>.bin`.
- Restart with the same `--save-path`; model, optimizer, scheduler, `global_step`, and `global_micro_step` are restored.
- In Slurm, send `SIGUSR1` before the time limit so the trainer saves at the next optimizer-step boundary and exits cleanly.
- The included `slurm_finetune_pst2.sbatch` uses `#SBATCH --signal=USR1@600` for this purpose.

### finetune-2.py

Purpose: Wrapper entrypoint that calls finetune.py main() (useful for cluster script naming compatibility).

Help (same args as finetune.py):

```bash
python3 finetune-2.py --help
```

Example:

```bash
python3 finetune-2.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep --max-steps 1000 --per-gpu-batch-size 64 --grad-accum-steps 1 --save-path ./Finetuned
```

### predict_OCCPAST.py

Purpose: Run prediction pipeline (cleaning, inference, CSV outputs, optional merged JSON formatting when predict-system is both).

Recent behavior updates:

- The PST lookup JSON prompt was removed in normal both-mode flow.
- If --lookup is not passed, the built-in default path is used.
- If an input CSV is missing `occ1_original`, the script prompts for the column to use for prediction.
- Interactive CSV selection accepts multiple files separated by commas, ranges, or `all`.
- New format-only mode can skip inference and build final JSON from existing CSV outputs.
- In format-only mode, if --hisco-csv / --pst-csv are omitted, the script auto-selects the most recent matching files.

Help:

```bash
python3 predict_OCCPAST.py --help
```

Example (full inference, non-interactive input):

```bash
python3 predict_OCCPAST.py --input predictions/to_predict/cedric_french_strings.csv --predict-system both --lookup predictions/occpast/updatedPST2CodeDict.json --output-dir predictions/predicted
```

Example (multiple non-interactive inputs):

```bash
python3 predict_OCCPAST.py --input predictions/to_predict/file1.csv,predictions/to_predict/file2.csv --predict-system both --lookup predictions/occpast/updatedPST2CodeDict.json --output-dir predictions/predicted
```

Example (explicit prediction column):

```bash
python3 predict_OCCPAST.py --input predictions/to_predict/jobs.csv --prediction-column job_title --predict-system both --lookup predictions/occpast/updatedPST2CodeDict.json --output-dir predictions/predicted
```

Example (parallel HISCO+PST inference on one node/GPU):

```bash
python3 predict_OCCPAST.py --input predictions/to_predict/cedric_french_strings.csv --predict-system both --parallel-systems --batch-size 512 --parallel-workers 2 --lookup predictions/occpast/updatedPST2CodeDict.json --output-dir predictions/predicted
```

Example (format-only using latest files automatically):

```bash
python3 predict_OCCPAST.py --format-only --predict-system both --lookup predictions/occpast/updatedPST2CodeDict.json --output-dir predictions/predicted
```

Example (format-only with explicit CSV paths):

```bash
python3 predict_OCCPAST.py --format-only --predict-system both --hisco-csv predictions/predicted/cedric_french_strings_predictions_hisco_2025-11-29_163101.csv --pst-csv predictions/predicted/cedric_french_strings_predictions_pst_2025-09-17_214339.csv --lookup predictions/occpast/updatedPST2CodeDict.json --output-dir predictions/predicted
```

### format_preds.py

Purpose: Merge HISCO + PST prediction CSVs into formatted JSON; optionally write sampled quarter chunks.

Help:

```bash
python3 format_preds.py --help
```

Example:

```bash
python3 format_preds.py --hisco predictions/predicted/cedric_french_strings_predictions_hisco_2025-11-29_163101.csv --pst2 predictions/predicted/cedric_french_strings_predictions_pst_2025-09-17_214339.csv --lookup predictions/occpast/updatedPST2CodeDict.json --out predictions/predicted/formatted_predictions.json --chunks predictions/predicted/chunks --base sample --n 300 --seed 42
```

Distributed Notes
-----------------

When using Slurm, use one launcher strategy consistently. For multi-GPU finetuning in this repo, the recommended pattern is one `srun` task per node and `torchrun` spawning one rank per GPU on that node.

Example single-node local launch:

```bash
torchrun --nproc_per_node=4 finetune.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep --max-steps 1000 --per-gpu-batch-size 64 --grad-accum-steps 1
```

Example Slurm script:

```bash
export WANDB_API_KEY=...
sbatch slurm_finetune_pst2.sbatch
```

For 4 GPUs with `--per-gpu-batch-size 512` and `--grad-accum-steps 1`, the initial `global_micro_batch_size` and `effective_batch_size` are both `2048`. A late schedule of `--late-phase-grad-accum-steps 2 4 8` changes effective batch to `4096`, `8192`, and `16384` without changing per-GPU memory use.

Legacy direct Slurm launch without torchrun is still possible when using one task per GPU:

```bash
srun --mpi=pmix_v3 --ntasks=4 --gpus-per-task=1 python3 finetune.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep --max-steps 1000 --per-gpu-batch-size 64 --grad-accum-steps 1 --distributed
```

Notes
-----

- Run commands from the repository root so imports of histocc resolve correctly.
- Prefer python3 in this repository environment.
- Use each script's --help for the complete and current flag list.
