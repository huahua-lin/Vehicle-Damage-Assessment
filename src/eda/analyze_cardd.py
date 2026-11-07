"""Basic validation and statistics for the CarDD COCO dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "CarDD_release" / "CarDD_COCO"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "cardd_audit.json"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the CarDD dataset")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_annotations(path: Path, errors: list[str]) -> dict[str, Any] | None:
    """Load a COCO annotation file and record any structural errors."""
    if not path.exists():
        errors.append(f"Missing annotation file: {path}")
        return None

    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"Could not read {path}: {error}")
        return None

    required_keys = ("images", "annotations", "categories")
    if not all(isinstance(data.get(key), list) for key in required_keys):
        errors.append(f"Invalid COCO structure in {path}")
        return None

    return data


def bbox_is_valid(bbox: object) -> bool:
    """Return whether a bounding box has valid COCO coordinates."""
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
        and bbox[0] >= 0
        and bbox[1] >= 0
        and bbox[2] > 0
        and bbox[3] > 0
    )


def segmentation_is_valid(segmentation: object) -> bool:
    """Return whether a segmentation contains valid polygon shapes."""
    if not isinstance(segmentation, list) or not segmentation:
        return False
    return all(isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0 for polygon in segmentation)


def audit_dataset(dataset_root: Path) -> dict[str, Any]:
    """Validate all dataset splits and return summary statistics."""
    dataset_root = dataset_root.expanduser().resolve()
    errors = []
    split_reports = {}
    split_file_names = {}
    expected_categories = None
    total_class_counts = Counter()

    if not dataset_root.is_dir():
        errors.append(f"Dataset root does not exist: {dataset_root}")

    # Audit each official split independently.
    for split in SPLITS:
        annotation_path = dataset_root / "annotations" / f"instances_{split}2017.json"
        image_dir = dataset_root / f"{split}2017"
        data = load_annotations(annotation_path, errors)

        if data is None:
            split_reports[split] = {
                "images": 0,
                "annotations": 0,
                "multi_class_images": 0,
                "class_counts": {},
                "missing_images": 0,
                "invalid_annotations": 0,
            }
            split_file_names[split] = set()
            continue

        # Confirm that all splits use the same category mapping.
        categories = {item["id"]: item["name"] for item in data["categories"] if "id" in item and "name" in item}
        if expected_categories is None:
            expected_categories = categories
        elif categories != expected_categories:
            errors.append(f"{split}: categories do not match the train split")

        # Index images and verify that every referenced file exists.
        images_by_id = {image.get("id"): image for image in data["images"]}
        file_names = {image.get("file_name") for image in data["images"] if image.get("file_name")}
        split_file_names[split] = file_names

        missing_images = [file_name for file_name in file_names if not (image_dir / file_name).is_file()]
        for file_name in missing_images:
            errors.append(f"{split}: missing image {file_name}")

        # Validate annotations while counting classes per image.
        class_counts = Counter()
        classes_by_image = defaultdict(set)
        invalid_annotations = 0

        for annotation in data["annotations"]:
            annotation_id = annotation.get("id")
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            problems = []

            if image_id not in images_by_id:
                problems.append("unknown image_id")
            if category_id not in categories:
                problems.append("unknown category_id")
            if not bbox_is_valid(annotation.get("bbox")):
                problems.append("invalid bbox")
            if not segmentation_is_valid(annotation.get("segmentation")):
                problems.append("invalid segmentation")

            if problems:
                invalid_annotations += 1
                errors.append(f"{split}: annotation {annotation_id}: {', '.join(problems)}")
                continue

            class_name = categories[category_id]
            class_counts[class_name] += 1
            classes_by_image[image_id].add(class_name)

        # Store the completed statistics for this split.
        total_class_counts.update(class_counts)
        split_reports[split] = {
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
            "multi_class_images": sum(len(names) > 1 for names in classes_by_image.values()),
            "class_counts": dict(sorted(class_counts.items())),
            "missing_images": len(missing_images),
            "invalid_annotations": invalid_annotations,
        }

    # A repeated filename across splits may indicate data leakage.
    for index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[index + 1 :]:
            overlap = split_file_names.get(first_split, set()) & split_file_names.get(second_split, set())
            if overlap:
                errors.append(f"{first_split}/{second_split}: {len(overlap)} overlapping filenames")

    # Combine split results into the final JSON-ready report.
    return {
        "dataset_root": str(dataset_root),
        "categories": expected_categories or {},
        "splits": split_reports,
        "totals": {
            "images": sum(split["images"] for split in split_reports.values()),
            "annotations": sum(split["annotations"] for split in split_reports.values()),
            "class_counts": dict(sorted(total_class_counts.items())),
        },
        "validation": {"passed": not errors, "error_count": len(errors), "errors": errors},
    }


def print_report(report: dict[str, Any]) -> None:
    """Print the main dataset statistics and validation result."""
    print(f"Dataset: {report['dataset_root']}")
    print(f"Categories: {', '.join(report['categories'].values())}\n")
    print(f"{'Split':<8} {'Images':>8} {'Annotations':>12} {'Multi-class':>12}")

    for name, split in report["splits"].items():
        print(f"{name:<8} {split['images']:>8,} {split['annotations']:>12,} {split['multi_class_images']:>12,}")

    totals = report["totals"]
    print(f"{'Total':<8} {totals['images']:>8,} {totals['annotations']:>12,}")
    print("\nInstances by class:")
    for name, count in sorted(totals["class_counts"].items(), key=lambda item: item[1], reverse=True):
        print(f"  {name:<16} {count:>5,}")

    validation = report["validation"]
    status = "PASSED" if validation["passed"] else "FAILED"
    print(f"\nValidation: {status} ({validation['error_count']} errors)")
    for error in validation["errors"][:20]:
        print(f"  - {error}")


def save_report(report: dict[str, Any], output_path: Path) -> None:
    """Save the audit results as a JSON report."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"\nJSON report: {output_path}")


def main() -> int:
    args = parse_args()
    report = audit_dataset(args.dataset_root)
    print_report(report)
    save_report(report, args.output)
    return 0 if report["validation"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
