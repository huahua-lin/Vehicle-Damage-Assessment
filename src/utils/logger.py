"""Experiment logging helpers."""

import logging
import sys
from pathlib import Path


def setup_logger(output_dir: Path) -> logging.Logger:
    """Log training messages to both stdout and training.log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cardd_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate messages when main() is called more than once.
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "training.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
