"""Training and validation loops for Mask R-CNN."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train for one epoch and return the mean total loss."""
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch:0{len(str(total_epochs))}d}/{total_epochs}")
    for images, targets in progress_bar:
        images = [image.to(device) for image in images]
        targets = [{name: value.to(device) for name, value in target.items()} for target in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss_parts = model(images, targets)
            loss = sum(loss_parts.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    return running_loss / max(len(data_loader), 1)


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, mixed_precision: bool) -> dict[str, float]:
    """Return mean validation losses without updating model weights."""
    model.train()
    loss_totals = {}

    for images, targets in tqdm(data_loader, desc="Validation"):
        images = [image.to(device) for image in images]
        targets = [{name: value.to(device) for name, value in target.items()} for target in targets]

        with torch.amp.autocast("cuda", enabled=mixed_precision):
            loss_parts = model(images, targets)

        for name, value in loss_parts.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + value.item()

    batch_count = max(len(data_loader), 1)
    mean_losses = {name: total / batch_count for name, total in loss_totals.items()}
    mean_losses["loss"] = sum(mean_losses.values())
    return mean_losses
