"""Training augmentation helpers."""

from __future__ import annotations

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


def build_training_transforms(config: dict) -> v2.Compose:
    """Create mask-aware TorchVision v2 training augmentation."""
    jitter = config["color_jitter"]
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomApply([v2.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter)], p=config["color_jitter_probability"]),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=config["blur_sigma"])], p=config["blur_probability"]),
            v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=config["noise_sigma"])], p=config["noise_probability"]),
            v2.RandomApply(
                [
                    v2.RandomAffine(
                        degrees=config["rotation_degrees"], scale=config["scale"], fill={tv_tensors.Image: 0.45, tv_tensors.Mask: 0}
                    )
                ],
                p=config["affine_probability"],
            ),
            v2.RandomHorizontalFlip(p=config["flip_probability"]),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes(),
        ]
    )
