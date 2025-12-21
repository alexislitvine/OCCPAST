This is a development for OCCPAST of the original OccCANINE model developed by Christian Møller Dahl, Torben Johansen, Christian Vedel from the University of Southern Denmark. The original repo is accesible at https://github.com/christianvedels/OccCANINE.git

Overview
--------

This repository provides everything needed to train, finetune, and use OccCANINE for OCCUPATIONSPAST.ORG


Structure
---------

*   **histocc**: Core Python code containing all logic for training, finetuning, and prediction. This is provided via a git submodule linked to the upstream OccCANINE repository (commit 844e6be6aa08c00235094ac3cd42698c9cf0c09b) to avoid divergence from the original repo.
*   **occpast_extensions**: OCCPAST-specific extensions to histocc, including data conversion utilities (CSV to Parquet) and description loading functions.
*   **tests**: Test suite for the histocc package and OCCPAST extensions.
*   **Data**: Contains key.csv mapping and toy data for testing.

## Important: Initializing the Repository

After cloning this repository, you must initialize the histocc submodule:

```bash
git submodule update --init --recursive
cd OccCANINE_upstream
git checkout 844e6be6aa08c00235094ac3cd42698c9cf0c09b
cd ..
```

The `histocc` directory is a symbolic link to `OccCANINE_upstream/histocc` from the upstream repository.

## Main Scripts

### Training
*   **train.py**: Train OccCANINE from scratch
*   **train_mixer.py**: Train OccCANINE with mixer architecture
*   **finetune.py**: Finetune a pre-trained OccCANINE model

### Prediction
*   **predict_OCCPAST.py**: Prediction script for PST data with formatting

### Utilities
*   **format_preds.py**: Format prediction outputs (used by predict_OCCPAST.py)

## Getting Started

Run everything locally without installing a package.

### Environment
Ensure Python 3.10+ and required libraries (PyTorch, pandas, numpy, tqdm). If you use conda:

```bash
conda create -n occpast python=3.10 -y
conda activate occpast
# Install core deps (adjust CUDA per your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy pyarrow tqdm scikit-learn
```

### Train from scratch
```bash
python train.py --epochs 1 --batch_size 64
```
View all options:
```bash
python train.py --help
```

### Train mixer variant
```bash
python train_mixer.py --epochs 1 --batch_size 64
```
View all options:
```bash
python train_mixer.py --help
```

### Finetune an existing model
```bash
python finetune.py --checkpoint path/to/checkpoint.pt --epochs 1
```
View all options:
```bash
python finetune.py --help
```

### Predict on PST data
```bash
# Non-interactive run with explicit paths
python predict_OCCPAST.py \
	--input predictions/to_predict/your_file.csv \
	--lookup predictions/occpast/updatedPST2CodeDict.json \
	--output-dir predictions/predicted
```
View all options:
```bash
python predict_OCCPAST.py --help
```

Examples using repo data:
```bash
# Predict interactively from predictions/to_predict/ (press Enter at prompts)
python predict_OCCPAST.py
```

### Format predictions (optional)
## Script Options Summary

### train.py
- **checkpoint-path**: model checkpoint directory
- **model-name**: optional model name label
- **model-domain**: training data domain label
- **sample-size**: training size modifier
- **epochs**: training epochs
- **batch-size**: batch size
- **learning-rate**: optimizer learning rate
- **warmup-pct**: warmup proportion of total steps
- **upsample-minimum**: minimum count for class upsampling
- **alt-prob**: alternative sampling probability
- **dropout**: final-layer dropout rate
- **max-len**: input length (tokens/characters)
- **skip-insert-words**: disable word-insertion augmentation

### train_mixer.py
- **save-dir**: directory to save checkpoints
- **save-interval**: steps between saves
- **save-each-epoch**: save after every epoch
- **initial-checkpoint**: weights for initialization
- **only-encoder**: load encoder part only
- **train-data/val-data**: one or more dataset paths
- **target-col-naming**: label schema name
- **target-cols**: label columns
- **block-size**: target code length (e.g., 5 for HISCO)
- **use-within-block-sep**: use comma within code blocks
- **distributed/local_rank/world-size**: multi-GPU training
- **log-interval/eval-interval**: logging cadence
- **log-wandb/wandb-project-name**: W&B logging
- **num-epochs/batch-size**: training schedule
- **num-workers/pin-memory**: dataloader settings
- **learning-rate/warmup-pct/dropout/max-len**: training hyperparams
- **decoder-dim-feedforward**: decoder FF dim (defaults to encoder hidden)
- **seq2seq-weight**: sequence loss weight
- **formatter**: {hisco, occ1950, gpf}
- **num-transformations/augmentation-prob/unk-lang-prob**: augmentation
- **fn-word-freq**: CSV for word frequencies used in augmentation

### finetune.py
- **save-path/save-interval**: fine-tune save location and cadence
- **dataset/input-col/target-cols**: data source and labels
- **language/language-col**: language metadata
- **distributed/local_rank/world-size**: multi-GPU training
- **block-size/share-val/use-within-block-sep**: target formatting & split
- **drop-bad-labels**: filter invalid labels
- **allow-codes-shorter-than-block-size**: permit shorter codes
- **log-interval/eval-interval/log-wandb/wandb-project-name**: logging
- **num-epochs/batch-size/num-workers/prefetch-factor/pin-memory/persistent-workers**: dataloader/training settings
- **learning-rate/seq2seq-weight/warmup-pct**: hyperparameters
- **initial-checkpoint/only-encoder/freeze-encoder**: initialization & freezing
- **use-amp**: Automatic Mixed Precision
- **prepare-only**: write prepared data then exit
- **include-descriptions/description-prob/description-template**: description augmentation
- **descriptions-file/..-code-col/..-text-col/..-lang-col**: description sources
- **all-codes-file/all-codes-col**: include all unique codes in key

### predict_OCCPAST.py
- **debug**: print raw greedy outputs

### format_preds.py
- **hisco**: path to predictions_hisco.csv
- **pst2**: path to predictions_pst2.csv
- **lookup**: path to updatedPST2CodeDict.json
- **out**: output JSON path
- **chunks**: directory to write quarter chunks
- **base**: base filename for chunk files
- **n**: sample size per quarter
- **seed**: random seed (omit for non-deterministic)
```bash
python format_preds.py --input predictions/predicted/your_file_predictions.csv --output predictions/predicted/your_file_formatted.csv
```
View all options:
```bash
python format_preds.py --help
```

Examples using repo data:
```bash
# Format existing prediction outputs into JSON + optional quarter chunks
python format_preds.py \
	--hisco predictions/predicted/cedric_french_strings_predictions_hisco_2025-11-29_163101.csv \
	--pst2 predictions/predicted/cedric_french_strings_predictions_pst_2025-09-17_214339.csv \
	--lookup data/occpast/updatedPST2CodeDict.json \
	--out predictions/predicted/formatted_predictions.json \
	--chunks predictions/predicted/chunks \
	--base sample \
	--n 1000 \
	--seed 42
```

### Data keys and lookups
- HISCO key file: Data/Key.csv
- PST2 code lookup: data/occpast/updatedPST2CodeDict.json

Notes:
- Commands above are examples; check script `--help` for full options.
- Keep the working directory at the repo root so imports of `histocc` resolve.

## histocc Code

The `histocc` folder contains all the code used for training and application of OccCANINE.

Note: This repository is not currently set up for Python packaging or distribution (no `setup.py`/`pyproject.toml`). If you need an installable package in the future, we can add a `pyproject.toml` and restore packaging instructions.

*   **Data/**: Contains 'Key.csv' which maps integer codes (generated by OccCANINE) to HISCO codes based on definitions from https://github.com/cedarfoundation/hisco. Also contains TOYDATA for testing.
*   **model_assets.py**: Defines the underlying PyTorch model
*   **attacker.py**: Text attack procedure for text augmentation during training
*   **trainer.py**: Training procedures
*   **dataloader.py**: Data loading and batching for model training
*   **prediction_assets.py**: Functions and classes for using OccCANINE. Contains the 'OccCANINE' class, the main user interface.
*   **formatter/**: Code formatters for different output formats (HISCO, OCC1950, general purpose)
*   **loss/**: Loss functions including order-invariant loss
*   **utils/**: Utility functions for data conversion, metrics, masking, etc.

