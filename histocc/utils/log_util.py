import os
import csv
import argparse
import hashlib

from collections import OrderedDict

try:
    import wandb
except ImportError:
    pass


def update_summary(step: int, metrics, filename, log_wandb=False):
    write_header = not os.path.isfile(filename)

    rowd = OrderedDict(step=step)
    rowd.update(metrics)

    with open(filename, mode='a') as summary:
        writer = csv.DictWriter(summary, fieldnames=rowd.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(rowd)

    if log_wandb:
        wandb.log(rowd)


def pathhash(output_dir: str, target_len: int = 20):
    output_dir = os.path.normpath(output_dir).split(os.path.sep)
    output_dir = ''.join(output_dir)

    abbrev = hashlib.sha256(output_dir.encode()).hexdigest()

    if len(abbrev) < target_len:
        abbrev += '0' * (target_len - len(abbrev))
    elif len(abbrev) > target_len:
        abbrev = abbrev[:target_len]

    return abbrev


def wandb_init(
        output_dir: str,
        project: str,
        name: str,
        resume: str,
        config: argparse.Namespace,
        mode: str = 'online',
        ):
    _dir = os.path.join('./wandb', pathhash(output_dir))

    if not os.path.exists(_dir):
        os.makedirs(_dir)

    init_timeout = int(os.getenv("WANDB_INIT_TIMEOUT", "120"))
    settings = wandb.Settings(init_timeout=init_timeout)
    fallback_mode = os.getenv("WANDB_FALLBACK_MODE", "offline")
    allow_fail = os.getenv("WANDB_ALLOW_INIT_FAIL", "0") == "1"

    try:
        wandb.init(
            project=project,
            name=name,
            dir=_dir,
            resume=resume,
            config=config,
            mode=mode,
            settings=settings,
            )
    except wandb.errors.CommError as exc:
        if allow_fail:
            print(f"W&B init failed: {exc}. Continuing without W&B.")
            return
        if fallback_mode:
            print(f"W&B init failed: {exc}. Falling back to mode={fallback_mode}.")
            wandb.init(
                project=project,
                name=name,
                dir=_dir,
                resume=resume,
                config=config,
                mode=fallback_mode,
                settings=settings,
                )
        else:
            raise
