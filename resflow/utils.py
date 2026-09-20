from __future__ import annotations

from pathlib import Path
import os
import random
import shutil

import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_setup() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def pad_to_multiple(image: Tensor, multiple: int) -> tuple[Tensor, tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (height, width)
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(image, (0, pad_w, 0, pad_h), mode=mode), (height, width)


def crop_original(image: Tensor, shape: tuple[int, int]) -> Tensor:
    return image[..., : shape[0], : shape[1]]


def checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: object,
    step: int,
    config: dict,
) -> dict:
    return {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def atomic_torch_save(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def atomic_link_or_copy(source: str | Path, destination: str | Path) -> None:
    """Atomically update ``destination`` without duplicating storage when possible."""
    source, destination = Path(source), Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def prune_checkpoints(directory: str | Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(Path(directory).glob("step_*.pt"))
    for path in checkpoints[:-keep_last]:
        path.unlink()
