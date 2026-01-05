This is a development for OCCPAST of the original OccCANINE model developed by Christian Møller Dahl, Torben Johansen, Christian Vedel from the University of Southern Denmark. The original repo is accesible at https://github.com/christianvedels/OccCANINE.git

Overview
--------

This repository provides everything needed to train, finetune, and use OccCANINE for OCCUPATIONSPAST.ORG


Structure
---------

*   **histocc**: Core Python code containing all logic for training, finetuning, and prediction. This repo is intended for local use; packaging is currently disabled.
*   **tests**: Test suite for the histocc package.
*   **Data**: Contains key.csv mapping and toy data for testing.

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

## Late Phase Training

Late phase training is an advanced training feature that automatically switches training settings when the model reaches a stable state. This allows for more efficient training by using larger effective batch sizes and adjusted learning rates after the model has learned the basic gating behavior.

### When Late Phase Training is Enabled

Late phase training is enabled when **any** of the following parameters are set:

- `--late-grad-accum > 1`: Gradient accumulation steps for late phase
- `--late-lr-mult != 1.0`: Learning rate multiplier for late phase
- `--late-warmup-steps > 0`: Warmup steps after switching to late phase
- `--late-phase-batch-sizes`: Batch size scaling schedule (requires `--late-phase-start-step`)

### Automatic Triggering

Late phase training is automatically triggered when the model's gating performance stabilizes. Stabilization is detected by monitoring a specific metric over a window of evaluation steps:

**Stabilization Criteria:**
- **Metric to monitor**: Specified by `--gate-stabilize-metric` (default: `gating_f1`)
- **Window size**: Specified by `--gate-stabilize-window` (default: 5 evaluation steps)
- **Maximum variation**: The metric must vary by less than `--gate-stabilize-delta` (default: 0.02) within the window
- **Minimum value**: The metric must be at least `--gate-stabilize-min` (default: 0.90) within the window

Once these criteria are met, the late phase training automatically activates.

### Late Phase Training Parameters

**finetune.py parameters:**

- `--late-grad-accum`: Number of gradient accumulation steps after late phase activation (default: 1)
- `--late-lr-mult`: Multiplier applied to learning rate when switching to late phase (default: 1.0)
- `--late-warmup-steps`: Number of warmup steps after switching to late phase (default: 0)
- `--late-switch-once`: Only switch to late phase once when stabilization is detected (default: True)

**Gating stabilization detection:**

- `--gate-stabilize-metric`: Metric name to monitor for gating stabilization (default: `gating_f1`)
- `--gate-stabilize-window`: Number of eval points to check for stabilization (default: 5)
- `--gate-stabilize-delta`: Maximum allowed metric variation within window (default: 0.02)
- `--gate-stabilize-min`: Minimum metric value required within window (default: 0.90)

**Batch size scaling (advanced):**

- `--late-phase-start-step`: Step to begin late-phase batch scaling
- `--late-phase-batch-sizes`: Global batch sizes for late-phase scaling (e.g., `512 1024 2048`)
- `--late-phase-batch-steps`: Absolute steps for each batch-size transition (length = len(batch_sizes) - 1)
- `--late-phase-lr-mults`: LR multipliers per batch-size transition (default: 0.7 per transition)

### Example Usage

```bash
# Basic late phase training with gradient accumulation
python finetune.py \
  --checkpoint path/to/checkpoint.pt \
  --late-grad-accum 4 \
  --late-lr-mult 0.5 \
  --late-warmup-steps 100

# Advanced late phase with batch size scaling
python finetune.py \
  --checkpoint path/to/checkpoint.pt \
  --late-phase-start-step 10000 \
  --late-phase-batch-sizes 256 512 1024 \
  --late-phase-batch-steps 10000 15000 \
  --late-phase-lr-mults 0.7 0.7
```

### How It Works

1. Training begins in normal mode with standard settings
2. At each evaluation interval, the specified gating metric (e.g., `gating_f1`) is computed
3. The metric history is monitored using a sliding window
4. When the metric stabilizes (low variation and high value), late phase is triggered
5. Upon activation:
   - Learning rate is multiplied by `late_lr_mult`
   - Gradient accumulation changes to `late_grad_accum` steps
   - Optional warmup is performed over `late_warmup_steps`
   - Optional batch size scaling begins if configured

This approach allows the model to train with smaller batches initially (for better exploration) and then switch to larger effective batches (for more stable updates) once the basic behavior is learned.

## Key Metrics

During training and evaluation, several key metrics are computed to assess model performance. These metrics are logged to CSV files, printed to the console, and optionally logged to Weights & Biases (W&B).

### Sequence Accuracy (seq_acc)

**What it measures:** The percentage of complete sequences that are predicted exactly correctly.

**How it's calculated:**
1. The model predicts multiple occupation codes for each input (e.g., up to 5 codes for HISCO)
2. For each predicted sequence, check if **all** predicted codes match their target codes in the correct positions (order-invariant)
3. A sequence is correct only if every single code block matches a target code block exactly
4. Sequence accuracy = (Number of completely correct sequences / Total sequences) × 100

**Implementation:** See `order_invariant_accuracy()` in `histocc/utils/metrics.py`

### Token Accuracy (token_acc)

**What it measures:** The percentage of individual tokens (characters within codes) that are predicted correctly.

**How it's calculated:**
1. Each occupation code is composed of multiple tokens (e.g., 5 characters for HISCO)
2. For each predicted code block, find the best matching target code block (order-invariant matching)
3. Calculate the percentage of tokens that match between predicted and target blocks
4. Token accuracy = (Total matching tokens / Total tokens) × 100

This metric weighs longer sequences higher within each batch, providing a more granular view of model performance than sequence accuracy.

**Implementation:** See `order_invariant_accuracy()` in `histocc/utils/metrics.py`

### Flat Accuracy (flat_acc)

**What it measures:** The accuracy of the linear classifier head in predicting which codes are present in the input.

**How it's calculated:**
1. The model has a separate linear decoder that predicts a binary vector
2. Each dimension represents whether a specific code is present in the occupation description
3. Predictions are thresholded at 0.5 (sigmoid > 0.5 = code present)
4. Flat accuracy = sklearn's `accuracy_score()` comparing predicted and target binary vectors

This metric assesses the model's ability to identify relevant codes regardless of order or position.

**Implementation:** See `evaluate()` in `histocc/seq2seq_mixer_engine.py`

### F1 Gating (gating_f1)

**What it measures:** The model's ability to correctly predict when a second occupation code is present (gating decision).

**How it's calculated:**

The gating decision determines whether the model predicts one or two occupation codes. This is critical for datasets like PST where some entries have two codes.

1. **Prediction:** Decode the model's output for the second code block. If any non-padding tokens are generated, the model predicts "has 2 codes"
2. **Ground truth:** Compare against the `gold_num_codes` field (whether the true data has 2+ codes)
3. **Confusion matrix:**
   - True Positives (TP): Model predicts 2 codes, gold has 2 codes
   - False Positives (FP): Model predicts 2 codes, gold has 1 code
   - False Negatives (FN): Model predicts 1 code, gold has 2 codes
   - True Negatives (TN): Model predicts 1 code, gold has 1 code

4. **Metrics:**
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Additional gating metrics:**
- `gating_precision`: Precision of the gating decision
- `gating_recall`: Recall of the gating decision
- `gating_tp`, `gating_fp`, `gating_fn`, `gating_tn`: Raw counts from confusion matrix

**Implementation:** See `evaluate()` in `histocc/seq2seq_mixer_engine.py` (requires `compute_gating_metrics=True`)

### Language-Specific Metrics

When training on multilingual datasets, the system also tracks sequence and token accuracy per language:

- `seq_acc_{lang}`: Sequence accuracy for specific language (e.g., `seq_acc_en`, `seq_acc_de`)
- `token_acc_{lang}`: Token accuracy for specific language
- `count_{lang}`: Number of samples for each language

These metrics help identify performance differences across languages. See `LANGUAGE_METRICS.md` for more details.

### Viewing Metrics

**During training:**
- Metrics are printed to console at each evaluation interval
- Example output shows validation loss, accuracies, and learning rate

**In logs:**
- All metrics are saved to `{save_dir}/logs.csv`
- Each row represents one evaluation step with all computed metrics

**In W&B:**
- If `--log-wandb` is enabled, all metrics are automatically logged to Weights & Biases
- Metrics can be visualized in real-time during training

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

