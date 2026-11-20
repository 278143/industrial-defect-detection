"""Inspect local MVTec AD images and masks without changing the dataset."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Decode through NumPy so Windows paths containing Chinese work correctly."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def inspect_category(category_dir: Path) -> tuple[dict, list[dict]]:
    """Check image decoding, layout, and pixel mask correspondence."""
    errors: list[str] = []
    counts: dict[str, dict[str, int]] = {"train": {}, "test": {}}
    sizes: Counter[str] = Counter()
    expected_masks: set[Path] = set()
    samples: list[dict] = []
    for required in ("train/good", "test/good", "ground_truth"):
        if not (category_dir / required).is_dir():
            errors.append(f"Missing directory: {required}")
    for split in ("train", "test"):
        split_dir = category_dir / split
        if not split_dir.is_dir():
            continue
        for label_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            images = sorted(label_dir.glob("*.png"))
            counts[split][label_dir.name] = len(images)
            if not images:
                errors.append(f"No PNG images in {split}/{label_dir.name}")
            if split == "train" and label_dir.name != "good":
                errors.append(f"Unexpected non-normal training directory: {label_dir.name}")
            for image_path in images:
                mask_path = None
                if split == "test" and label_dir.name != "good":
                    mask_path = (
                        category_dir
                        / "ground_truth"
                        / label_dir.name
                        / f"{image_path.stem}_mask.png"
                    )
                    expected_masks.add(mask_path)
                samples.append(
                    {
                        "split": split,
                        "label": label_dir.name,
                        "image": image_path,
                        "mask": mask_path,
                    }
                )
                try:
                    image = read_image(image_path)
                    height, width = image.shape[:2]
                    sizes[f"{width}x{height}"] += 1
                    if mask_path is not None:
                        if not mask_path.is_file():
                            raise ValueError(f"Missing mask: {mask_path}")
                        mask = read_image(mask_path, cv2.IMREAD_UNCHANGED)
                        if mask.ndim != 2 or mask.shape != image.shape[:2]:
                            raise ValueError(f"Mask shape mismatch: {mask_path}")
                        if not np.isin(mask, [0, 255]).all():
                            raise ValueError(f"Mask must contain only 0 and 255: {mask_path}")
                        if not np.any(mask):
                            raise ValueError(f"Empty defect mask: {mask_path}")
                except (OSError, ValueError, cv2.error) as exc:
                    errors.append(str(exc))
    actual_masks = set((category_dir / "ground_truth").rglob("*.png"))
    for orphan in sorted(actual_masks - expected_masks):
        errors.append(f"Mask without a matching test image: {orphan}")
    report = {
        "category": category_dir.name,
        "dataset_path": str(category_dir),
        "counts": counts,
        "train_total": sum(counts["train"].values()),
        "test_total": sum(counts["test"].values()),
        "test_normal": counts["test"].get("good", 0),
        "test_anomalous": sum(n for label, n in counts["test"].items() if label != "good"),
        "mask_total": len(actual_masks),
        "image_sizes_width_height": dict(sorted(sizes.items())),
        "errors": errors,
        "passed": not errors,
    }
    if report["train_total"] == 0 or report["test_total"] == 0:
        errors.append("Training and test images must both be present")
        report["passed"] = False
    return report, samples


def save_preview(samples: list[dict], output_path: Path) -> None:
    """Show one test image per label alongside its ground-truth mask and overlay."""
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected: dict[str, dict] = {}
    for sample in samples:
        if sample["split"] == "test":
            selected.setdefault(sample["label"], sample)
    figure, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)), squeeze=False)
    try:
        for row, (label, sample) in enumerate(selected.items()):
            image = cv2.cvtColor(read_image(sample["image"]), cv2.COLOR_BGR2RGB)
            mask = (
                read_image(sample["mask"], cv2.IMREAD_GRAYSCALE)
                if sample["mask"] is not None
                else np.zeros(image.shape[:2], dtype=np.uint8)
            )
            overlay = image.copy()
            overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(
                np.uint8
            )
            axes[row, 0].imshow(image)
            axes[row, 0].set_title(f"{label} / {sample['image'].name}")
            axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
            axes[row, 1].set_title("Ground-truth mask" if sample["mask"] else "Normal: empty mask")
            axes[row, 2].imshow(overlay)
            axes[row, 2].set_title("Ground-truth overlay")
            for axis in axes[row]:
                axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_path, dpi=130)
    finally:
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/dataset.yaml")
    parser.add_argument("--category", help="Override the category in the configuration")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("Configuration must contain a YAML mapping")
    category = args.category or config["category"]
    if Path(category).name != category or category in (".", ".."):
        parser.error("Category must be a single directory name")
    category_dir = PROJECT_ROOT / config["root"] / category
    output_dir = PROJECT_ROOT / config["output_dir"]
    if not category_dir.is_dir():
        parser.error(f"Dataset directory not found: {category_dir}")
    report, samples = inspect_category(category_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{category}_inspection.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Category: {category}")
    print(f"Train: {report['train_total']} | Test: {report['test_total']}")
    print(f"Normal test: {report['test_normal']} | Anomalous test: {report['test_anomalous']}")
    print(f"Masks: {report['mask_total']} | Errors: {len(report['errors'])}")
    print(f"Report: {report_path}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        return 1
    preview_path = output_dir / f"{category}_preview.png"
    save_preview(samples, preview_path)
    print(f"Preview: {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
