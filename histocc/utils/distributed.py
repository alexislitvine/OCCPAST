import os
import re


def _resolve_slurm_nodelist(nodelist: str) -> str:
    if not nodelist:
        return ""

    if "[" not in nodelist or "]" not in nodelist:
        return nodelist.split(",")[0]

    prefix, rest = nodelist.split("[", 1)
    ranges, _ = rest.split("]", 1)
    first_range = ranges.split(",")[0]
    if "-" in first_range:
        first_range = first_range.split("-", 1)[0]
    first_range = re.sub(r"\D", "", first_range)
    return f"{prefix}{first_range}" if first_range else prefix


def configure_slurm_env(default_port: str = "29500") -> None:
    if "SLURM_PROCID" in os.environ and "RANK" not in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "SLURM_LOCALID" in os.environ and "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "SLURM_NTASKS" in os.environ and "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = default_port

    if "MASTER_ADDR" not in os.environ and "SLURM_NODELIST" in os.environ:
        resolved = _resolve_slurm_nodelist(os.environ["SLURM_NODELIST"])
        if resolved:
            os.environ["MASTER_ADDR"] = resolved

