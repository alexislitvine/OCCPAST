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

Purpose: Finetune a model on a custom dataset.

Help:

```bash
python3 finetune.py --help
```

Example:

```bash
python3 finetune.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep --num-epochs 1 --batch-size 64 --save-path ./Finetuned
```

Late-phase example (one-liner):

```bash
python3 finetune.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep --late-grad-accum 4 --late-lr-mult 0.5 --late-warmup-steps 100 --num-epochs 1 --batch-size 64 --save-path ./Finetuned
```

### finetune-2.py

Purpose: Wrapper entrypoint that calls finetune.py main() (useful for cluster script naming compatibility).

Help (same args as finetune.py):

```bash
python3 finetune-2.py --help
```

Example:

```bash
python3 finetune-2.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep --num-epochs 1 --batch-size 64 --save-path ./Finetuned
```

### predict_OCCPAST.py

Purpose: Run prediction pipeline (cleaning, inference, CSV outputs, optional merged JSON formatting when predict-system is both).

Recent behavior updates:

- The PST lookup JSON prompt was removed in normal both-mode flow.
- If --lookup is not passed, the built-in default path is used.
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

When using Slurm, use one launcher strategy consistently:

- torchrun without srun
- srun without torchrun

Examples:

```bash
torchrun --nproc_per_node=4 finetune.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep
```

```bash
srun --mpi=pmix_v3 --ntasks=4 --gpus-per-task=1 python3 finetune.py --dataset Data/Training_data_other/pst2.csv --input-col occ1 --target-cols pst2_1 pst2_2 --use-within-block-sep --distributed
```

Notes
-----

- Run commands from the repository root so imports of histocc resolve correctly.
- Prefer python3 in this repository environment.
- Use each script's --help for the complete and current flag list.
