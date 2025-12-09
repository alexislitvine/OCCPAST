import os
import time

import torch

from torch import nn

from .formatter import PAD_IDX
from .utils import (
    create_mask,
    Averager,
    order_invariant_accuracy,
    update_summary,
)


def train_one_epoch(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_step: int,
        log_interval: int = 100,
        eval_interval: int | None = None,
        save_interval: int | None = None,
        save_dir: str | None = None,
        data_loader_eval: torch.utils.data.DataLoader | None = None,
        log_wandb: bool = False,
        ) -> tuple[float, float]:
    model = model.train()

    last_step = len(data_loader) - 1
    losses = Averager()
    batch_time = Averager()
    batch_time_data = Averager()
    samples_per_sec = Averager()
    
    # Check GPU availability once
    has_cuda = torch.cuda.is_available()

    # Need to initialize first "end time", as this is
    # calculated at bottom of batch loop
    end = time.time()

    for batch_idx, batch in enumerate(data_loader):
        current_step += 1

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch['targets'].to(device)

        batch_time_data.update(time.time() - end)

        # Prepare target as input for seq2seq model
        target_input = targets[:, :-1]
        target_mask, target_padding_mask = create_mask(target_input, PAD_IDX, device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target_input,
            target_mask=target_mask,
            target_padding_mask=target_padding_mask,
        )

        loss = loss_fn(outputs, targets)
        losses.update(loss.item(), outputs.size(0))

        # Backward pass & step
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        elapsed = time.time() - end
        batch_time.update(elapsed)
        samples_per_sec.update(outputs.size(0) / elapsed)

        if batch_idx % log_interval == 0 or batch_idx == last_step:
            # Calculate ETA
            batches_remaining = len(data_loader) - (batch_idx + 1)
            eta_seconds = batches_remaining * batch_time.avg
            eta_str = f"{int(eta_seconds // 60)}m{int(eta_seconds % 60):02d}s"
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]
            
            print(f'Batch {batch_idx + 1}/{len(data_loader)} | '
                  f'Loss: {losses.avg:.4f} | '
                  f'LR: {current_lr:.2e} | '
                  f'Batch time: {batch_time.avg:.2f}s (data: {batch_time_data.avg:.2f}s) | '
                  f'Samples/sec: {samples_per_sec.avg:.1f} | '
                  f'ETA: {eta_str}')
            
            # Print GPU memory stats if using CUDA
            if has_cuda:
                print(f'  GPU Memory - Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB | '
                      f'Reserved: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB')

        if save_interval is not None and current_step % save_interval == 0:
            states = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': current_step,
            }
            torch.save(states, os.path.join(save_dir, f'{current_step}.bin'))
            torch.save(states, os.path.join(save_dir, 'last.bin'))

        if eval_interval is not None and current_step % eval_interval == 0:
            print('\n' + '='*80)
            print('Starting evaluation pass...')
            eval_loss, eval_seq_acc, eval_token_acc = evaluate(
                model=model,
                data_loader=data_loader_eval,
                loss_fn=loss_fn,
                device=device,
            )
            
            # Print evaluation summary
            print('='*80)
            print(f'EVALUATION RESULTS (Step {current_step})')
            print('='*80)
            print(f'Validation Loss     : {eval_loss:.4f}')
            print(f'Training Loss       : {losses.avg:.4f}')
            print(f'Sequence Accuracy   : {eval_seq_acc:.2f}%')
            print(f'Token Accuracy      : {eval_token_acc:.2f}%')
            print(f'Learning Rate       : {scheduler.get_last_lr()[0]:.2e}')
            print('='*80 + '\n')

            update_summary(
                current_step,
                metrics={
                    'batch_time': batch_time.avg,
                    'batch_time_data': batch_time_data.avg,
                    'train_loss': losses.avg,
                    'val_loss': eval_loss,
                    'seq_acc': eval_seq_acc,
                    'token_acc': eval_token_acc,
                    'lr': scheduler.get_last_lr()[0],
                },
                filename=os.path.join(save_dir, 'logs.csv'),
                log_wandb=log_wandb,
            )

        end = time.time()

    return current_step


@torch.no_grad
def evaluate(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        ):
    model = model.eval()

    losses = Averager()
    token_accs = Averager()
    seq_accs = Averager()

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch['targets'].to(device)

        # Prepare target as input for seq2seq model
        target_input = targets[:, :-1]
        target_mask, target_padding_mask = create_mask(target_input, PAD_IDX, device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target_input,
            target_mask=target_mask,
            target_padding_mask=target_padding_mask,
        )

        loss = loss_fn(outputs, targets)
        losses.update(loss.item(), outputs.size(0))

        seq_acc, token_acc = order_invariant_accuracy(
            output=outputs,
            target=targets[:, 1:],
            pad_idx=PAD_IDX,
            nb_blocks=loss_fn.nb_blocks,
            block_size=loss_fn.block_size,
        )
        seq_accs.update(seq_acc.item(), outputs.size(0))
        token_accs.update(token_acc.item(), outputs.size(0))

    return losses.avg, seq_accs.avg, token_accs.avg


def train(
        model: nn.Module,
        data_loaders: dict[str, torch.utils.data.DataLoader], # TODO split or use dataclass
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        save_dir: str,
        total_steps: int,
        current_step: int = 0,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 1000,
        log_wandb: bool = False,
        ):
    while current_step < total_steps:
        print('\n' + '='*80)
        print(f'Starting new epoch (Step {current_step}/{total_steps} - {100*current_step/total_steps:.1f}% complete)')
        print('='*80)

        current_step = train_one_epoch(
            model,
            data_loaders['data_loader_train'],
            loss_fn,
            optimizer,
            device,
            scheduler,
            current_step=current_step,
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            save_dir=save_dir,
            data_loader_eval=data_loaders['data_loader_val'],
            log_wandb=log_wandb,
        )
    
    # Save model at the end of training to ensure latest version is always saved
    states = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': current_step,
    }
    torch.save(states, os.path.join(save_dir, f'{current_step}.bin'))
    torch.save(states, os.path.join(save_dir, 'last.bin'))
    print('\n' + '='*80)
    print(f'TRAINING COMPLETE - Final model saved at step {current_step}')
    print('='*80)
