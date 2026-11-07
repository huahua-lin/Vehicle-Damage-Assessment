"""General reproducibility and checkpoint helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed random generators and request deterministic operations."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic Python and NumPy seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    class_names: list[str],
    config: dict,
    output_path: Path,
) -> None:
    """Save model state, optimizer state and inference metadata."""
    checkpoint = {
        "epoch": epoch,
        "validation_loss": validation_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "class_names": class_names,
        "training_config": config,
    }
    torch.save(checkpoint, output_path)
