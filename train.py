"""Train Mask R-CNN on the CarDD dataset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src import (
    CarDDDataset,
    build_model,
    build_training_transforms,
    create_data_loader,
    evaluate_loss,
    save_checkpoint,
    seed_everything,
    setup_logger,
    train_one_epoch,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_config.yaml"
LOG_SEPARATOR = "=" * 60


def build_dataloaders(config: dict, dataset_root: Path):
    """Create the training and validation DataLoaders."""
    train_dataset = CarDDDataset(dataset_root, split="train", transforms=build_training_transforms(config["augmentation"]))
    validation_dataset = CarDDDataset(dataset_root, split="val")

    train_loader = create_data_loader(
        train_dataset, config["data"]["batch_size"], config["data"]["num_workers"], shuffle=True, seed=config["training"]["seed"]
    )
    validation_loader = create_data_loader(
        validation_dataset, config["data"]["batch_size"], config["data"]["num_workers"], shuffle=False, seed=config["training"]["seed"]
    )
    return train_loader, validation_loader, train_dataset.class_names


def main() -> int:
    """Load the configuration and orchestrate model training."""
    # Load the experiment settings.
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on CarDD")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    with args.config.expanduser().resolve().open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Set up reproducibility, output logging and TensorBoard.
    seed_everything(config["training"]["seed"])
    output_dir = PROJECT_ROOT / config["experiment"]["exp_dir"]
    logger = setup_logger(output_dir)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # Record the resolved runtime settings in checkpoints and logs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = config["training"]["mixed_precision"] and device.type == "cuda"
    config["runtime"] = {"device": str(device), "mixed_precision": mixed_precision}

    # Save the exact configuration used by this run.
    with (output_dir / "runtime_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    # Log the effective configuration in compact sections.
    logger.info(LOG_SEPARATOR)
    logger.info("Starting Training Experiment")
    logger.info(LOG_SEPARATOR)
    for section, values in config.items():
        compact_values = yaml.safe_dump(values, default_flow_style=True, sort_keys=False).strip()
        logger.info("Config - %s: %s", section, compact_values)
    logger.info(LOG_SEPARATOR)

    # Build the datasets, model and optimisation components.
    train_loader, validation_loader, class_names = build_dataloaders(config, PROJECT_ROOT / config["data"]["dataset_root"])

    model = build_model(config["model"]["num_classes"], config["model"]["pretrained"]).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        momentum=config["optimizer"]["momentum"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"])
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    # Track progress and format epoch numbers to the configured maximum width.
    total_epochs = config["training"]["epochs"]
    epoch_width = len(str(total_epochs))
    best_validation_loss = float("inf")
    best_epoch = 0
    training_start = time.perf_counter()

    try:
        for epoch in range(1, total_epochs + 1):
            # Train and validate one complete epoch.
            epoch_start = time.perf_counter()
            training_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, total_epochs)
            validation_losses = evaluate_loss(model, validation_loader, device, mixed_precision)
            validation_loss = validation_losses["loss"]
            epoch_duration = time.perf_counter() - epoch_start

            # Record metrics before advancing the learning-rate schedule.
            writer.add_scalar("train/loss", training_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            for name, value in validation_losses.items():
                writer.add_scalar(f"val/{name}", value, epoch)
            writer.flush()
            scheduler.step()

            # Replace the last checkpoint after every completed epoch.
            save_checkpoint(model, optimizer, epoch, validation_loss, class_names, config, output_dir / "checkpoint_last.pt")

            logger.info(
                "Epoch [%s] Train: %.4f | Val: %.4f | Time: %.2fs",
                f"{epoch:0{epoch_width}d}/{total_epochs}",
                training_loss,
                validation_loss,
                epoch_duration,
            )

            # Replace the best checkpoint only when validation improves.
            if validation_loss < best_validation_loss:
                previous_best = best_validation_loss
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_path = output_dir / "checkpoint_best.pt"
                save_checkpoint(model, optimizer, epoch, validation_loss, class_names, config, best_path)
                logger.info("--> Val loss improved from %.4f to %.4f. Saved checkpoint_best.pt", previous_best, validation_loss)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 130
    except Exception as error:
        logger.exception("Training failed: %s", error)
        raise
    finally:
        # Close TensorBoard and report the best completed epoch.
        writer.close()
        total_minutes = (time.perf_counter() - training_start) / 60
        if best_epoch:
            logger.info("Training complete in %.2f min. Best Val Loss: %.4f (Epoch %d)", total_minutes, best_validation_loss, best_epoch)
        else:
            logger.info("Training stopped after %.2f min before an epoch completed.", total_minutes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
