"""Validate data-checking failures using synthetic images, without real data."""

import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest

inspect_category = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/prepare_data.py")
)["inspect_category"]


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    assert success
    encoded.tofile(path)


@pytest.fixture
def category_dir(tmp_path):
    root = tmp_path / "metal_nut"
    image = np.full((16, 16, 3), 100, dtype=np.uint8)
    for relative in ("train/good/000.png", "test/good/000.png", "test/scratch/000.png"):
        write_png(root / relative, image)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    write_png(root / "ground_truth/scratch/000_mask.png", mask)
    return root


def test_valid_dataset(category_dir):
    report, samples = inspect_category(category_dir)
    assert report["passed"]
    assert (report["train_total"], report["test_total"], report["mask_total"]) == (1, 2, 1)
    assert len(samples) == 3


@pytest.mark.parametrize("problem", ["missing", "shape", "nonbinary", "empty"])
def test_invalid_mask(category_dir, problem):
    path = category_dir / "ground_truth/scratch/000_mask.png"
    if problem == "missing":
        path.unlink()
    elif problem == "shape":
        write_png(path, np.full((8, 8), 255, dtype=np.uint8))
    elif problem == "nonbinary":
        write_png(path, np.full((16, 16), 128, dtype=np.uint8))
    else:
        write_png(path, np.zeros((16, 16), dtype=np.uint8))
    report, _ = inspect_category(category_dir)
    assert not report["passed"]
    assert len(report["errors"]) == 1


def test_unmatched_mask(category_dir):
    write_png(
        category_dir / "ground_truth/scratch/999_mask.png", np.full((16, 16), 255, dtype=np.uint8)
    )
    report, _ = inspect_category(category_dir)
    assert not report["passed"]
    assert any("without a matching test image" in error for error in report["errors"])
