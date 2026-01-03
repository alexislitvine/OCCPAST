import argparse
import os

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import yaml

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    get_linear_schedule_with_warmup,
    CanineTokenizer,
)

import pandas as pd

from histocc import (
    OccDatasetMixerInMemMultipleFiles,
    load_tokenizer,
    Seq2SeqMixerOccCANINE,
    BlockOrderInvariantLoss,
    LossMixer,
)
from histocc.seq2seq_mixer_engine import train
from histocc.formatter import (
    BlockyFormatter,
    construct_general_purpose_formatter,
    PAD_IDX,
    EOS_IDX,
)
from histocc.utils import wandb_init

try:
    # want to do import to set has_wandb even if not used directly
    import wandb # pylint: disable=W0611
    has_wandb = True # pylint: disable=C0103
except ImportError:
    has_wandb = False # pylint: disable=C0103

from train_mixer import load_states


def parse_args():
    parser = argparse.ArgumentParser()

    # File paths, data & model choices
    parser.add_argument('--save-path', type=str, default='./Finetuned/', help='Directory to store fine-tuned model, incl. processed data')
    parser.add_argument('--save-interval', type=int, default=1000, help='Number of steps between saving model')
    parser.add_argument('--dataset', type=str, default=None, help='Filename of dataset')
    parser.add_argument('--input-col', type=str, default='occ1', help='Column name of column with occupational descriptions')
    parser.add_argument('--target-cols', type=str, nargs='+', default=None, help='List of column names with labels')

    # Language (must specify none or one of below, cannot specify both)
    parser.add_argument('--language', type=str, default='unk', help='Occupational description language')
    parser.add_argument('--language-col', type=str, default=None, help='Optional column name in --dataset with language')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true', default=False, help='Enable distributed training across multiple GPUs')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (set automatically by torch.distributed.launch)')
    parser.add_argument('--world-size', type=int, default=1, help='Number of processes for distributed training')

    # Data settings
    parser.add_argument('--block-size', type=int, default=5, help='Maximum number of characters in target (e.g., this is 5 for the HISCO system)')
    parser.add_argument('--share-val', type=float, default=0.1, help='Share of data set aside for tracking model performance')
    parser.add_argument('--use-within-block-sep', action='store_true', default=False, help='Whether to use "," as a separator for tokens WITHIN a code. Useful for, e.g., PSTI')
    parser.add_argument('--drop-bad-labels', action='store_true', default=False, help='Omit all observations where labels not adhere to formatting rules.')
    parser.add_argument('--allow-codes-shorter-than-block-size', action='store_true', default=False, help='Allow for codes shorter than block size. If not specified, such codes will raise an error, as they may indicate that leading zeroes have accidently been dropped.')

    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=100, help='Number of steps between reporting training stats')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Number of steps between calculating and logging validation performance')
    parser.add_argument('--log-wandb', action='store_true', default=False, help='Whether to log validation performance using W&B')
    parser.add_argument('--wandb-project-name', type=str, default='histco-v2-mixer')

    # Data parameters
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--prefetch-factor', type=int, default=None, help='Number of batches loaded in advance by each worker (default: None uses PyTorch default of 2)')
    parser.add_argument('--pin-memory', action='store_true', default=False, help='Pin memory for faster data transfer to GPU')
    parser.add_argument('--persistent-workers', action='store_true', default=False, help='Keep workers alive between epochs (requires num_workers > 0)')

    # Model and optimizer parameters
    parser.add_argument('--learning-rate', type=float, default=2e-05)
    parser.add_argument('--seq2seq-weight', type=float, default=0.1)
    parser.add_argument('--warmup-pct', type=float, default=0.05, help='Warmup steps as percentage of total steps (default: 0.05 = 5%%)')

    # Model initialization
    parser.add_argument('--initial-checkpoint', type=str, default=None, help='Model weights to use for initialization. Discarded if resume state exists at --save-path')
    parser.add_argument('--only-encoder', action='store_true', default=False, help='Only attempt to load encoder part of --initial-checkpoint')

    # Freezing
    parser.add_argument('--freeze-encoder', action='store_true', default=False)
    
    # Mixed precision training
    parser.add_argument('--use-amp', action='store_true', default=False, help='Use Automatic Mixed Precision (AMP) for training')
    
    # Data preparation only
    parser.add_argument('--prepare-only', action='store_true', default=False, help='Only prepare data (write data_train.csv, data_val.csv, and key.csv) then exit before model/dataloader/training initialization')

    # Description augmentation
    parser.add_argument('--include-descriptions', action='store_true', default=False, help='Include code descriptions in the training input to help contextualize predictions')
    parser.add_argument('--description-prob', type=float, default=0.5, help='Probability of including description in input when --include-descriptions is enabled (default: 0.5)')
    parser.add_argument('--description-template', type=str, default='{occ} [DESC: {desc}]', help='Template for formatting input with description. Use {occ} for occupational text and {desc} for code description')
    parser.add_argument('--descriptions-file', type=str, default=None, help='Path to CSV file containing code-to-description mappings')
    parser.add_argument('--descriptions-code-col', type=str, default=None, help='Column name in --descriptions-file containing the codes')
    parser.add_argument('--descriptions-text-col', type=str, default=None, help='Column name in --descriptions-file containing the descriptions')
    parser.add_argument('--descriptions-lang-col', type=str, default=None, help='Column name in --descriptions-file containing language codes (for multilingual descriptions)')

    # All codes file for complete key generation
    parser.add_argument('--all-codes-file', type=str, default=None, help='Path to CSV file containing all unique codes for the target system. These codes will be included in the key even if they do not appear in the training data.')
    parser.add_argument('--all-codes-col', type=str, default=None, help='Column name in --all-codes-file containing the codes. If not specified, will try common column names (pst2_1, system_code, code) or use first column.')

    # Debugging: PST2 sample diagnostics
    parser.add_argument('--debug-pst2-samples', type=int, default=0, help='Number of deterministic PST2 rows to print diagnostics for (requires pst2_1 and pst2_2 in target cols).')
    parser.add_argument('--debug-pst2-seed', type=int, default=42, help='Random seed (int) for selecting PST2 debug rows.')
    parser.add_argument('--disallow-pad-inside-block', action='store_true', default=False, help='Disallow PAD during greedy decoding inside code blocks during eval/probing.')
    parser.add_argument('--disallow-zero-at-block-start', action='store_true', default=False, help='Disallow predicting token "0" at the start of each block during greedy decoding.')
    parser.add_argument('--min-double-steps', type=int, default=5000, help='Number of initial steps to enforce a minimum doubles quota per batch.')
    parser.add_argument('--min-double-ratio', type=float, default=0.1, help='Minimum doubles ratio per batch during warmup steps.')
    parser.add_argument('--gate-stabilize-metric', type=str, default='gating_f1', help='Metric name to monitor for gating stabilization.')
    parser.add_argument('--gate-stabilize-window', type=int, default=5, help='Number of eval points to check for gating stabilization.')
    parser.add_argument('--gate-stabilize-delta', type=float, default=0.02, help='Maximum allowed metric range within stabilization window.')
    parser.add_argument('--gate-stabilize-min', type=float, default=0.90, help='Minimum metric value required within stabilization window.')
    parser.add_argument('--late-grad-accum', type=int, default=1, help='Gradient accumulation steps after gating stabilization.')
    parser.add_argument('--late-lr-mult', type=float, default=1.0, help='Multiplier applied to LR when switching to late phase.')
    parser.add_argument('--late-warmup-steps', type=int, default=0, help='Warmup steps after switching to late phase.')
    parser.add_argument('--late-switch-once', action=argparse.BooleanOptionalAction, default=True, help='Only switch to late phase once when stabilization is detected.')
    parser.add_argument('--late-phase-start-step', type=int, default=None, help='Step to begin late-phase true batch scaling (global batch sizes).')
    parser.add_argument('--late-phase-batch-sizes', type=int, nargs='+', default=None, help='Global batch sizes for late-phase scaling (e.g., 512 1024 2048).')
    parser.add_argument('--late-phase-batch-steps', type=int, nargs='+', default=None, help='Absolute steps for each batch-size transition (length = len(batch_sizes)-1).')
    parser.add_argument('--late-phase-lr-mults', type=float, nargs='+', default=None, help='LR multipliers per batch-size transition (default: 0.7 per transition).')

    args = parser.parse_args()

    if args.language != 'unk' and args.language_col is not None:
        raise ValueError('Only specify one of --language and --language-col')

    if args.log_wandb and not has_wandb:
        raise ImportError('Specified --log-wandb, but wandb is not installed')

    if args.include_descriptions:
        if args.descriptions_file is not None:
            if args.descriptions_code_col is None or args.descriptions_text_col is None:
                raise ValueError(
                    'When using --descriptions-file, must specify both '
                    '--descriptions-code-col and --descriptions-text-col'
                )
        else:
            print('Warning: --include-descriptions enabled but no --descriptions-file provided. '
                  'No descriptions will be included in training.')

    return args


def _is_present_pst2(value: str | None) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return False
    value = str(value)
    return value.lower() not in {'', ' ', '?', 'nan', 'none', 'null'}


def print_pst2_sample_diagnostics(
        data_path: str,
        formatter: BlockyFormatter,
        sample_size: int,
        seed: int,
) -> None:
    if sample_size <= 0:
        return

    if 'pst2_1' not in formatter.target_cols or 'pst2_2' not in formatter.target_cols:
        print('PST2 debug sample requested, but target_cols do not include pst2_1 and pst2_2. Skipping.')
        return

    data = pd.read_csv(data_path, dtype=str)
    for col in ['occ1', 'pst2_1', 'pst2_2']:
        if col not in data.columns:
            print(f'PST2 debug sample requested, but {col} is missing from {data_path}. Skipping.')
            return

    has_both = data['pst2_1'].apply(_is_present_pst2) & data['pst2_2'].apply(_is_present_pst2)
    data = data[has_both]
    if data.empty:
        print('PST2 debug sample requested, but no rows with both pst2_1 and pst2_2 present were found.')
        return

    sample_size = min(sample_size, len(data))
    sample = data.sample(n=sample_size, random_state=seed)

    print(f'PST2 debug sample diagnostics (n={sample_size}, seed={seed}):')
    for idx, row in sample.iterrows():
        serialized = formatter.sanitize(row)
        labels = formatter.transform_label(row)
        labels_int = labels.astype(int).tolist()
        token_len = len(labels_int)
        last_tokens = labels_int[-20:]
        non_pad_tokens = int((labels != PAD_IDX).sum())
        block_boundaries = [
            1 + (block_idx + 1) * formatter.block_size
            for block_idx in range(formatter.max_num_codes - 1)
        ]
        eos_early = False
        for block_idx in range(formatter.max_num_codes):
            start = 1 + block_idx * formatter.block_size
            end = start + formatter.block_size
            block = labels_int[start:end]
            if EOS_IDX in block[:-1]:
                eos_early = True
                break
        overflow = False
        if serialized is not None:
            if formatter.sep_value:
                overflow = len(serialized.split(formatter.sep_value)) > formatter.max_num_codes
            else:
                overflow = formatter.max_num_codes < 1
        truncation = overflow or eos_early

        print(f'  row_index={idx}')
        print(f'    raw occ1={row["occ1"]!r}')
        print(f'    raw pst2_1={row["pst2_1"]!r}')
        print(f'    raw pst2_2={row["pst2_2"]!r}')
        print(f'    serialized target={serialized!r}')
        print(f'    target token length={token_len} last_20_tokens={last_tokens}')
        print(f'    non_padding_tokens={non_pad_tokens}')
        print(f'    separator_block_boundaries={block_boundaries}')
        print(f'    truncation={truncation} (overflow or EOS before block end)')


def check_if_data_prepared(save_path: str) -> dict[str, int] | None:
    if not os.path.isfile(os.path.join(save_path, 'data_train.csv')):
        return None

    if not os.path.isfile(os.path.join(save_path, 'data_val.csv')):
        return None

    if not os.path.isfile(os.path.join(save_path, 'key.csv')):
        return None

    mapping_df = pd.read_csv(
        os.path.join(save_path, 'key.csv'),
        dtype={'system_code': str, 'code': int},
        )
    mapping = dict(mapping_df.values)

    return mapping


def prepare_target_cols(
        data: pd.DataFrame,
        formatter: BlockyFormatter,
        drop_bad_rows: bool = False,
        allow_codes_shorter_than_block_size: bool = False,
) -> pd.DataFrame:
    # All cases of space (' ') are cast to NaN
    for i, target_col in enumerate(formatter.target_cols):
        # Some NaN values instead coded as spaces
        data[target_col] = data[target_col].replace(' ', None)
        data[target_col] = data[target_col].replace(
            {'nan': None, 'NaN': None, 'NAN': None, 'none': None, 'None': None, 'null': None, 'NULL': None}
        )

    # First colummn should not contain any NaN -> use the '?' token instead
    assert '?' in formatter.map_char_idx
    data[formatter.target_cols[0]] = data[formatter.target_cols[0]].fillna('?')

    # Send all through formatter and track whether that works
    passes_formatter: list[bool] = []

    # Track whether length shorter that block size, as that may indicate
    # that leading zeros have been dropped. We can do this by tracking
    # whether there are any EOS_IDX present in formatted code ASIDE from
    # as its last element
    len_less_than_block_size: list[int] = []

    for i in range(len(data)):
        try:
            formatted = formatter.transform_label(data.iloc[i])
            passes_formatter.append(True)

            if EOS_IDX in formatted[:-1]:
                if data.iloc[i][formatter.target_cols[0]] == '?' and not EOS_IDX in formatted[(formatter.block_size + 1):-1]:
                    # OK to have EOS_IDX in FIRST code if due to missing -> '?' cast (see above)
                    pass
                else:
                    len_less_than_block_size.append(i)
        except: # pylint: disable=W0702
            passes_formatter.append(False)

        if (i + 1) % 10_000 == 0:
            print(f'Scanned {i + 1:,} of {len(data):,} observations.')

    bad_cases = len(passes_formatter) - sum(passes_formatter)

    if bad_cases > 0:
        if drop_bad_rows:
            print(f'Dropping {bad_cases} cases of labels not fit for formatter.')
            data = data[passes_formatter]
        else:
            raise ValueError(f'{bad_cases} bad cases of labels (of {len(data)}). If you to omit these rows, specify --drop-bad-labels')

    if len(len_less_than_block_size) > 0:
        if allow_codes_shorter_than_block_size:
            print(f'{len(len_less_than_block_size):,} cases of labels shorter than block size. Assuming such codes are allowed since --allow-codes-shorter-than-block-size was specified.')
        else:
            raise ValueError(f'{len(len_less_than_block_size):,} cases of labels shorter than block size, which may indicate that leading zeroes have been dropped by accident. If these cases are exptected, specify --allow-codes-shorter-than-block-size \nExample rows: {data.iloc[len_less_than_block_size].head(10)}')

    return data


def prepare_data(
        dataset: str,
        input_col: str,
        formatter: BlockyFormatter,
        save_path: str,
        share_val: float,
        language: str = 'unk',
        language_col: str | None = None,
        drop_bad_rows: bool = False,
        allow_codes_shorter_than_block_size: bool = False,
        all_codes_file: str | None = None,
        all_codes_col: str | None = None,
) -> dict[str, int]:
    if not os.path.isdir(save_path):
        print(f'Creating fine-tuning directory {save_path}')
        os.makedirs(save_path, exist_ok=False)
    else:
        mapping = check_if_data_prepared(save_path)

        if mapping is not None:
            print(f'Prepared data exists at {save_path}, using that')
            return mapping

    # Load all codes from external file if provided
    all_external_codes = None
    if all_codes_file is not None:
        print(f'Loading all codes from external file: {all_codes_file}')
        all_codes_df = pd.read_csv(all_codes_file, dtype=str)
        
        # Determine which column contains the codes
        if all_codes_col is not None:
            code_col = all_codes_col
        else:
            # Try common column names first
            code_col = None
            for col in ['pst2_1', 'system_code', 'code']:
                if col in all_codes_df.columns:
                    code_col = col
                    break
            # Fall back to first column if no common name found
            if code_col is None:
                code_col = all_codes_df.columns[0]
        
        all_external_codes = set(all_codes_df[code_col].dropna().unique())
        print(f'Loaded {len(all_external_codes):,} unique codes from external file (column: {code_col})')

    # Load
    print(f'Loading data from {dataset}...')
    data: pd.DataFrame = pd.read_csv(dataset, dtype=str)
    print(f'Loaded {len(data):,} observations.')

    # Select columns
    if language_col is None:
        data['lang'] = language
    else:
        data['lang'] = data[language_col]

    data = data[[input_col, *formatter.target_cols, 'lang']]
    data = data.rename(columns={input_col: 'occ1'})

    # Value checks, subsetting, and changing some values
    print('Validating and preparing target columns...')
    data = prepare_target_cols(
        data=data,
        formatter=formatter,
        drop_bad_rows=drop_bad_rows,
        allow_codes_shorter_than_block_size=allow_codes_shorter_than_block_size,
    )
    print(f'Target column preparation complete. {len(data):,} observations remaining.')

    # Build code <-> label mapping
    print('Building code-to-label mapping...')
    unique_values = pd.unique(data[formatter.target_cols].values.ravel())
    training_codes = set(val for val in unique_values if pd.notna(val))
    
    # Merge with external codes if provided
    if all_external_codes is not None:
        combined_codes = training_codes.union(all_external_codes)
        codes_only_in_training = training_codes - all_external_codes
        codes_only_in_external = all_external_codes - training_codes
        
        print(f'  Codes in training data: {len(training_codes):,}')
        print(f'  Codes in external file: {len(all_external_codes):,}')
        print(f'  Codes only in training (not in external): {len(codes_only_in_training):,}')
        print(f'  Codes only in external (not in training): {len(codes_only_in_external):,}')
        print(f'  Combined unique codes: {len(combined_codes):,}')
        
        # Sort for consistent ordering
        all_codes = sorted(combined_codes)
    else:
        print(f'  Codes in training data: {len(training_codes):,}')
        print('  Note: No external all-codes file provided. Key will only contain training data codes.')
        all_codes = sorted(training_codes)

    mapping: dict[str, int] = {
        code: i for i, code in enumerate(all_codes)
        }
    mapping_df = pd.DataFrame(mapping.items(), columns=['system_code', 'code'])
    print(f'Created mapping for {len(mapping):,} unique codes.')

    # Split data into train and validation
    print(f'Splitting data into train and validation sets (validation share: {share_val:.1%})...')
    data_val = data.sample(int(len(data) * share_val), random_state=42)
    data_train = data.drop(data_val.index)
    print(f'Split complete: {len(data_train):,} training observations, {len(data_val):,} validation observations.')

    # Shuffle training data to avoid issues with sequential ordering of different data types
    print('Shuffling training data...')
    data_train = data_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save datasets & mapping in specified fine-tuning folder
    print(f'Saving prepared data to {save_path}...')
    data_train.to_csv(os.path.join(save_path, 'data_train.csv'), index=False)
    data_val.to_csv(os.path.join(save_path, 'data_val.csv'), index=False)
    mapping_df.to_csv(os.path.join(save_path, 'key.csv'), index=False)
    print('Data files saved successfully (data_train.csv, data_val.csv, key.csv).')

    return mapping


def setup_datasets(
        target_cols: list[str],
        save_path: str,
        formatter: BlockyFormatter,
        tokenizer: CanineTokenizer,
        num_classes_flat: int,
        map_code_label: dict[str, int],
        distributed: bool = False,
        include_descriptions: bool = False,
        description_prob: float = 0.5,
        description_template: str = '{occ} [DESC: {desc}]',
        descriptions_file: str = None,
        descriptions_code_col: str = None,
        descriptions_text_col: str = None,
        descriptions_lang_col: str = None,
) -> tuple[OccDatasetMixerInMemMultipleFiles, OccDatasetMixerInMemMultipleFiles]:
    dataset_train = OccDatasetMixerInMemMultipleFiles(
        fnames_data=[os.path.join(save_path, 'data_train.csv')],
        formatter=formatter,
        tokenizer=tokenizer,
        max_input_len=128,
        num_classes_flat=num_classes_flat,
        training=True,
        target_cols=target_cols,
        map_code_label=map_code_label,
        include_descriptions=include_descriptions,
        description_prob=description_prob,
        description_template=description_template,
        descriptions_file=descriptions_file,
        descriptions_code_col=descriptions_code_col,
        descriptions_text_col=descriptions_text_col,
        descriptions_lang_col=descriptions_lang_col,
    )

    dataset_val = OccDatasetMixerInMemMultipleFiles(
        fnames_data=[os.path.join(save_path, 'data_val.csv')],
        formatter=formatter,
        tokenizer=tokenizer,
        max_input_len=128,
        num_classes_flat=num_classes_flat,
        training=False,
        target_cols=target_cols,
        map_code_label=map_code_label,
        # Note: include_descriptions is False for validation to evaluate on clean inputs
    )

    return dataset_train, dataset_val


def main():
    # Arguments
    args = parse_args()
    
    # Distributed init & device selection
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # Set CUDA device before initializing process group to avoid NCCL warnings
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        # Barrier with device_id to avoid "devices used ... unknown" warnings
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
    
    def is_main_process() -> bool:
        return (not distributed) or dist.get_rank() == 0
    
    # perf tweaks
    cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Target-side tokenization
    formatter = construct_general_purpose_formatter(
        block_size=args.block_size,
        target_cols=args.target_cols,
        use_within_block_sep=args.use_within_block_sep,
    )

    # Input-side tokenization
    tokenizer = load_tokenizer(
        model_domain='Multilingual_CANINE',
    )

    # Data prep (only on main process to avoid race conditions)
    if is_main_process():
        map_code_label = prepare_data(
            dataset=args.dataset,
            input_col=args.input_col,
            formatter=formatter,
            save_path=args.save_path,
            share_val=args.share_val,
            language=args.language,
            language_col=args.language_col,
            drop_bad_rows=args.drop_bad_labels,
            allow_codes_shorter_than_block_size=args.allow_codes_shorter_than_block_size,
            all_codes_file=args.all_codes_file,
            all_codes_col=args.all_codes_col,
        )
        print("Data preparation stage completed.")
        print_pst2_sample_diagnostics(
            data_path=os.path.join(args.save_path, 'data_train.csv'),
            formatter=formatter,
            sample_size=args.debug_pst2_samples,
            seed=args.debug_pst2_seed,
        )
    
    # Wait for main process to finish data preparation
    if distributed:
        dist.barrier()
        
        # Load prepared data on non-main processes
        if not is_main_process():
            mapping = check_if_data_prepared(args.save_path)
            if mapping is None:
                raise RuntimeError("Data preparation failed on main process")
            map_code_label = mapping
    
    # Exit early if --prepare-only was specified
    if args.prepare_only:
        if is_main_process():
            print(f"Data preparation complete. Files written to {args.save_path}:")
            print(f"  - data_train.csv")
            print(f"  - data_val.csv")
            print(f"  - key.csv")
            print("Exiting due to --prepare-only flag (skipping model/dataloaders/training)")
        
        # Synchronize all processes before cleanup
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        
        return
    
    num_classes_flat = len(map_code_label)

    if args.log_wandb and is_main_process():
        wandb_init(
            output_dir=args.save_path,
            project=args.wandb_project_name,
            name=os.path.basename(args.save_path),
            resume='auto',
            config=args,
        )

    # Load datasets
    dataset_train, dataset_val = setup_datasets(
        target_cols=args.target_cols,
        save_path=args.save_path,
        formatter=formatter,
        tokenizer=tokenizer,
        num_classes_flat=num_classes_flat,
        map_code_label=map_code_label,
        distributed=distributed,
        include_descriptions=args.include_descriptions,
        description_prob=args.description_prob,
        description_template=args.description_template,
        descriptions_file=args.descriptions_file,
        descriptions_code_col=args.descriptions_code_col,
        descriptions_text_col=args.descriptions_text_col,
        descriptions_lang_col=args.descriptions_lang_col,
    )

    # Data loaders with distributed samplers if needed
    # Build DataLoader kwargs
    dataloader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': True if args.num_workers > 0 else False,
        'prefetch_factor': 4 if args.num_workers > 0 else None,
    }
    
    if distributed:
        train_sampler = DistributedSampler(
            dataset_train,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            dataset_val,
            shuffle=False,
        )
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            **dataloader_kwargs,
        )
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            **dataloader_kwargs,
        )
    else:
        train_sampler = None
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            **dataloader_kwargs,
        )
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            **dataloader_kwargs,
        )

    # Setup model, optimizer, scheduler
    model = Seq2SeqMixerOccCANINE(
        model_domain='Multilingual_CANINE',
        num_classes=formatter.num_classes,
        num_classes_flat=num_classes_flat,
    ).to(device)
    
    # Wrap model with DDP if distributed
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if args.freeze_encoder:
        # Handle DDP wrapper when accessing model parameters
        model_to_freeze = model.module if distributed else model
        for param in model_to_freeze.encoder.parameters():
            param.requires_grad = False

        optimizer = AdamW([param for name, param in model.named_parameters() if not name.startswith("encoder.") and not name.startswith("module.encoder.")], lr=args.learning_rate)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(data_loader_train) * args.num_epochs
    num_warmup_steps = int(total_steps * args.warmup_pct)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    # Setup mixed loss
    loss_fn_seq2seq = BlockOrderInvariantLoss(
        pad_idx=PAD_IDX,
        nb_blocks=formatter.max_num_codes,
        block_size=formatter.block_size,
    )
    loss_fn_linear = torch.nn.BCEWithLogitsLoss()
    loss_fn = LossMixer(
        loss_fn_seq2seq=loss_fn_seq2seq,
        loss_fn_linear=loss_fn_linear,
        seq2seq_weight=args.seq2seq_weight,
    ).to(device)

    # Load states (only the model without DDP wrapper should be passed to load_states)
    model_to_load = model.module if distributed else model
    current_step = load_states(
        save_dir=args.save_path,
        model=model_to_load,
        optimizer=optimizer,
        scheduler=scheduler,
        initial_checkpoint=args.initial_checkpoint,
        only_encoder=args.only_encoder,
    )

    # Save arguments (only from main process)
    if is_main_process():
        with open(os.path.join(args.save_path, 'args.yaml'), 'w', encoding='utf-8') as args_file:
            args_file.write(
                yaml.safe_dump(args.__dict__, default_flow_style=False)
            )

    train(
        model=model,
        data_loaders={
            'data_loader_train': data_loader_train,
            'data_loader_val': data_loader_val,
        },
        train_sampler=train_sampler,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        save_dir=args.save_path,
        total_steps=total_steps,
        current_step=current_step,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        log_wandb=args.log_wandb and is_main_process(),
        save_interval=args.save_interval,
        distributed=distributed,
        is_main_process=is_main_process(),
        use_amp=args.use_amp,
        disallow_pad_inside_block=args.disallow_pad_inside_block,
        disallow_zero_at_block_start=args.disallow_zero_at_block_start,
        min_double_steps=args.min_double_steps,
        min_double_ratio=args.min_double_ratio,
        gate_stabilize_metric=args.gate_stabilize_metric,
        gate_stabilize_window=args.gate_stabilize_window,
        gate_stabilize_delta=args.gate_stabilize_delta,
        gate_stabilize_min=args.gate_stabilize_min,
        late_grad_accum=args.late_grad_accum,
        late_lr_mult=args.late_lr_mult,
        late_warmup_steps=args.late_warmup_steps,
        late_switch_once=args.late_switch_once,
        batch_size=args.batch_size,
        late_phase_start_step=args.late_phase_start_step,
        late_phase_batch_sizes=args.late_phase_batch_sizes,
        late_phase_batch_steps=args.late_phase_batch_steps,
        late_phase_lr_mults=args.late_phase_lr_mults,
    )
    
    # Cleanup distributed training
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
