"""Visualise one annotated CarDD image for each damage class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "CarDD_release" / "CarDD_COCO"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "cardd_damage_examples.png"

DAMAGE_COLOURS = {
    "dent": "#E41A1C",
    "scratch": "#FF7F00",
    "crack": "#984EA3",
    "glass shatter": "#377EB8",
    "lamp broken": "#4DAF4A",
    "tire flat": "#A65628",
}


def parse_args() -> argparse.Namespace:
    """Parse dataset, output and display options."""
    parser = argparse.ArgumentParser(description="Visualise CarDD annotations")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true", help="Open the figure after saving")
    return parser.parse_args()


def main() -> None:
    """Create and save the six-class annotation figure."""
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    annotation_path = dataset_root / "annotations" / "instances_train2017.json"

    # Load the training annotations and index image details by ID.
    with annotation_path.open(encoding="utf-8") as file:
        data = json.load(file)

    categories = {item["id"]: item["name"] for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Use the first original annotation available for each damage class.
    for axis, (category_id, class_name) in zip(axes.flat, categories.items()):
        candidates = [annotation for annotation in data["annotations"] if annotation["category_id"] == category_id]
        annotation = candidates[0]
        image_info = images[annotation["image_id"]]
        image_path = dataset_root / "train2017" / image_info["file_name"]
        colour = DAMAGE_COLOURS[class_name]

        axis.imshow(Image.open(image_path).convert("RGB"))

        # Draw the segmentation polygons and matching bounding box.
        for points in annotation["segmentation"]:
            polygon = list(zip(points[::2], points[1::2]))
            axis.add_patch(Polygon(polygon, facecolor=colour, edgecolor=colour, alpha=0.4))

        x, y, width, height = annotation["bbox"]
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=colour,
                linewidth=2.5,
            )
        )
        axis.set_title(class_name, color=colour, fontweight="bold")
        axis.axis("off")

    fig.suptitle("CarDD damage annotation examples", fontsize=16)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {args.output.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
