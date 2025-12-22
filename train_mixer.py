import argparse
import os
import yaml

import torch
import torch.distributed as dist

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
    hisco_blocky5,
    occ1950_blocky2,
    BlockyHISCOFormatter,
    BlockyOCC1950Formatter,
    PAD_IDX,
    BlockyFormatter,
    construct_general_purpose_formatter,
)
from histocc.utils import wandb_init, load_states

try:
    # want to do import to set has_wandb even if not used directly
    import wandb # pylint: disable=W0611
    has_wandb = True # pylint: disable=C0103
except ImportError:
    has_wandb = False # pylint: disable=C0103

# TODO torch.cudnn.benchmark

MAP_FORMATTER = {
    'hisco': hisco_blocky5,
    'occ1950': occ1950_blocky2,
    'gpf': construct_general_purpose_formatter,
}
MAP_NB_CLASSES = {
    'hisco': 1919,
    'occ1950': 1000, # FIMXE
    'gpf': 1919, # FIXME
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # File paths, data & model choices
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=5000, help='Number of steps between saving model')
    parser.add_argument('--save-each-epoch', action='store_true', default=False, help='Save model after each epoch instead of after a given number of steps')
    parser.add_argument('--initial-checkpoint', type=str, default=None, help='Model weights to use for initialization. Discarded if resume state exists at --save-dir')
    parser.add_argument('--only-encoder', action='store_true', default=False, help='Only attempt to load encoder part of --initial-checkpoint')

    parser.add_argument('--train-data', type=str, default=None, nargs='+')
    parser.add_argument('--val-data', type=str, default=None, nargs='+')
    parser.add_argument('--target-col-naming', type=str, default='hisco')

    # Args for general purpose formatter
    parser.add_argument('--target-cols', type=str, nargs='+', default=None, help='List of column names with labels')
    parser.add_argument('--block-size', type=int, default=5, help='Maximum number of characters in target (e.g., this is 5 for the HISCO system)')
    parser.add_argument('--use-within-block-sep', action='store_true', default=False, help='Whether to use "," as a separator for tokens WITHIN a code. Useful for, e.g., PSTI')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true', default=False, help='Enable distributed training across multiple GPUs')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (set automatically by torch.distributed.launch)')
    parser.add_argument('--world-size', type=int, default=1, help='Number of processes for distributed training')

    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=100, help='Number of steps between reporting training stats')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Number of steps between calculating and logging validation performance')
    parser.add_argument('--log-wandb', action='store_true', default=False, help='Whether to log validation performance using W&B')
    parser.add_argument('--wandb-project-name', type=str, default='histco-v2-mixer')

    # Data parameters
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pin-memory', action='store_true', default=False)

    # Model and optimizer parameters
    parser.add_argument('--learning-rate', type=float, default=2e-05)
    parser.add_argument('--warmup-pct', type=float, default=0.05, help='Warmup steps as percentage of total steps (default: 0.05 = 5%%)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Classifier dropout rate. Does not affect encoder.')
    parser.add_argument('--max-len', type=int, default=128, help='Max. number of characters for input')
    parser.add_argument('--decoder-dim-feedforward', type=int, default=None, help='Defaults to endoder hidden dim if not specified.')
    parser.add_argument('--seq2seq-weight', type=float, default=0.5)
    parser.add_argument('--formatter', type=str, default='hisco', choices=MAP_FORMATTER.keys(), help='Target-side tokenization')

    # Augmentation
    parser.add_argument('--num-transformations', type=int, default=3)
    parser.add_argument('--augmentation-prob', type=float, default=0.3)
    parser.add_argument('--unk-lang-prob', type=float, default=0.25)

    # Word frequency look-up for input augmentation
    parser.add_argument('--fn-word-freq', type=str, default=None, help='Filename for 2-column .csv-file ("word", "freq") to use for input string augmentation random word insertions.')

    args = parser.parse_args()

    if args.log_wandb and not has_wandb:
        raise ImportError('Specified --log-wandb, but wandb is not installed')

    # TODO add checks on file paths and directories

    return args


def setup_datasets(
        args: argparse.Namespace,
        formatter: BlockyHISCOFormatter | BlockyOCC1950Formatter | BlockyFormatter,
        tokenizer: CanineTokenizer,
        num_classes_flat: int,
) -> tuple[OccDatasetMixerInMemMultipleFiles, OccDatasetMixerInMemMultipleFiles]:
    if args.fn_word_freq is not None:
        word_freq_table = pd.read_csv(args.fn_word_freq, converters={'word': lambda x: x}) # Ensure 'nan' is not treated as NaN
    else:
        word_freq_table = None

    dataset_train = OccDatasetMixerInMemMultipleFiles(
        fnames_data=args.train_data,
        formatter=formatter,
        tokenizer=tokenizer,
        max_input_len=args.max_len,
        num_classes_flat=num_classes_flat,
        training=True,
        alt_prob=args.augmentation_prob,
        n_trans=args.num_transformations,
        unk_lang_prob=args.unk_lang_prob,
        target_cols=args.target_col_naming,
        word_freq_table=word_freq_table,
    )

    dataset_val = OccDatasetMixerInMemMultipleFiles(
        fnames_data=args.val_data,
        formatter=formatter,
        tokenizer=tokenizer,
        max_input_len=args.max_len,
        num_classes_flat=num_classes_flat,
        training=False,
        target_cols=args.target_col_naming,
    )

    return dataset_train, dataset_val


def setup_formatter(args: argparse.Namespace) -> BlockyFormatter | BlockyHISCOFormatter:
    formatter_fn = MAP_FORMATTER[args.formatter]

    if args.formatter == 'gpf':
        formatter = formatter_fn(
            block_size=args.block_size,
            target_cols=args.target_cols,
            use_within_block_sep=args.use_within_block_sep,
        )
    else:
        formatter = formatter_fn()

    return formatter


def main():
    args = parse_args()
    
    # Enable TF32 for improved performance on Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Setup distributed training
    local_rank = args.local_rank
    distributed = args.distributed
    
    # Initialize distributed training if enabled
    if distributed:
        if local_rank == -1:
            # If local_rank is not set, try to get it from environment variable
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        world_size = dist.get_world_size()
        
        # Only print from the main process
        is_main_process = local_rank == 0
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True
        world_size = 1

    if args.log_wandb and is_main_process:
        wandb_init(
            output_dir=args.save_dir,
            project=args.wandb_project_name,
            name=os.path.basename(args.save_dir),
            resume='auto',
            config=args,
        )

    # Target-side tokenization
    formatter = setup_formatter(args)
    num_classes_flat = MAP_NB_CLASSES[args.formatter]

    # Input-side tokenization
    tokenizer = load_tokenizer(
        model_domain='Multilingual_CANINE',
    )

    # Datasets
    dataset_train, dataset_val = setup_datasets(
        args=args,
        formatter=formatter,
        tokenizer=tokenizer,
        num_classes_flat=num_classes_flat,
    )

    # Data loaders with distributed samplers if needed
    if distributed:
        train_sampler = DistributedSampler(
            dataset_train,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            dataset_val,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False,
        )
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    else:
        train_sampler = None
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

    # Setup model, optimizer, scheduler
    model = Seq2SeqMixerOccCANINE(
        model_domain='Multilingual_CANINE',
        num_classes=formatter.num_classes, # FIXME potentially breaks to extract from formatter
        num_classes_flat=num_classes_flat,
        dropout_rate=args.dropout,
        decoder_dim_feedforward=args.decoder_dim_feedforward,
    ).to(device)
    
    # Wrap model with DDP if distributed
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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
        save_dir=args.save_dir,
        model=model_to_load,
        optimizer=optimizer,
        scheduler=scheduler,
        initial_checkpoint=args.initial_checkpoint,
        only_encoder=args.only_encoder,
    )

    # Save arguments (only from main process)
    if is_main_process:
        with open(os.path.join(args.save_dir, 'args.yaml'), 'w', encoding='utf-8') as args_file:
            args_file.write(
                yaml.safe_dump(args.__dict__, default_flow_style=False)
            )

    train(
        model=model,
        data_loaders={
            'data_loader_train': data_loader_train,
            'data_loader_val': data_loader_val,
        },
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        save_dir=args.save_dir,
        total_steps=total_steps,
        current_step=current_step,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_each_epoch=args.save_each_epoch,
        log_wandb=args.log_wandb and is_main_process,
        distributed=distributed,
        is_main_process=is_main_process,
    )
    
    # Cleanup distributed training
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
