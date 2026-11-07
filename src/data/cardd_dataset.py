"""PyTorch dataset loader for CarDD COCO instance-segmentation annotations."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors

from src.utils.general import seed_worker

VALID_SPLITS = ("train", "val", "test")
Target = dict[str, torch.Tensor]
Transform = Callable[[torch.Tensor, Target], tuple[torch.Tensor, Target]]


class CarDDDataset(Dataset):
    """Load one official CarDD split from its COCO annotations."""

    def __init__(self, dataset_root: str | Path, split: str = "train", transforms: Transform | None = None) -> None:
        """Load image metadata and group annotations by image ID."""
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.transforms = transforms
        self.image_dir = self.dataset_root / f"{split}2017"
        annotation_path = self.dataset_root / "annotations" / f"instances_{split}2017.json"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        with annotation_path.open(encoding="utf-8") as file:
            data = json.load(file)

        # Map original COCO category IDs to contiguous model labels starting at 1.
        self.images = sorted(data["images"], key=lambda item: item["id"])
        categories = sorted(data["categories"], key=lambda item: item["id"])
        self.category_id_to_label = {category["id"]: label for label, category in enumerate(categories, start=1)}
        self.class_names = ["background"] + [category["name"] for category in categories]

        # Group annotations once so each sample does not scan the full dataset.
        self.annotations_by_image = defaultdict(list)
        for annotation in data["annotations"]:
            self.annotations_by_image[annotation["image_id"]].append(annotation)

    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Target]:
        """Return an image tensor and its detection/segmentation targets."""
        image_info = self.images[index]
        image_path = self.image_dir / image_info["file_name"]

        with Image.open(image_path) as file:
            image = file.convert("RGB")

        annotations = self.annotations_by_image[image_info["id"]]
        target = self._build_target(image, image_info["id"], annotations)

        # TVTensors let torchvision v2 transform images, boxes and masks together.
        image_array = np.asarray(image).copy()
        image_tensor = tv_tensors.Image(torch.from_numpy(image_array).permute(2, 0, 1))

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)
            # Sanitation may remove instances, so refresh their dependent fields.
            target["area"] = target["masks"].flatten(1).sum(1).float()
            target["iscrowd"] = torch.zeros(len(target["labels"]), dtype=torch.int64)
        else:
            image_tensor = image_tensor.to(dtype=torch.float32) / 255.0

        return image_tensor, target

    def _build_target(self, image: Image.Image, image_id: int, annotations: list[dict[str, Any]]) -> Target:
        """Convert COCO boxes and polygons into PyTorch target tensors."""
        width, height = image.size
        boxes = []
        labels = []
        masks = []
        areas = []
        crowd_flags = []

        for annotation in annotations:
            # Convert COCO [x, y, width, height] to [x1, y1, x2, y2].
            x, y, box_width, box_height = annotation["bbox"]
            boxes.append([x, y, x + box_width, y + box_height])
            labels.append(self.category_id_to_label[annotation["category_id"]])
            areas.append(annotation["area"])
            crowd_flags.append(annotation.get("iscrowd", 0))

            # Rasterise all polygons belonging to this damage instance.
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            for polygon in annotation["segmentation"]:
                points = list(zip(polygon[::2], polygon[1::2]))
                draw.polygon(points, fill=1)
            masks.append(torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy()))

        # Preserve the expected tensor shapes for images without annotations.
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            mask_tensor = torch.stack(masks)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            mask_tensor = torch.zeros((0, height, width), dtype=torch.uint8)

        return {
            "boxes": tv_tensors.BoundingBoxes(box_tensor, format="XYXY", canvas_size=(height, width)),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": tv_tensors.Mask(mask_tensor),
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowd_flags, dtype=torch.int64),
        }

    def get_height_and_width(self, index: int) -> tuple[int, int]:
        """Return stored dimensions without loading the image file."""
        image_info = self.images[index]
        return image_info["height"], image_info["width"]


def collate_fn(batch: list[tuple[torch.Tensor, Target]]) -> tuple[tuple[torch.Tensor, ...], tuple[Target, ...]]:
    """Group variable-sized images and targets for a PyTorch DataLoader."""
    return tuple(zip(*batch))


def create_data_loader(dataset: Dataset, batch_size: int, workers: int, shuffle: bool, seed: int) -> DataLoader:
    """Create a loader that supports different image and target sizes."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
