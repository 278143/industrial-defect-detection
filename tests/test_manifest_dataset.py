"""Check manifest isolation, stale-data detection, preprocessing and calibration."""

import json

import cv2
import numpy as np
import pytest

from defect_detection.data.dataset import ManifestDataset, validate_manifest
from defect_detection.data.splitting import build_split
from defect_detection.evaluation.metrics import normal_threshold


@pytest.fixture
def source(tmp_path):
    samples = []
    for index in range(8):
        split = "train" if index < 6 else "test"
        label = "scratch" if index == 7 else "good"
        path = tmp_path / "metal_nut" / split / label / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((16, 16, 3), index * 20, dtype=np.uint8)
        _, encoded = cv2.imencode(".png", image)
        encoded.tofile(path)
        mask_path = None
        if label == "scratch":
            mask_path = tmp_path / "metal_nut/ground_truth/scratch/7_mask.png"
            mask_path.parent.mkdir(parents=True)
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:8, 4:8] = 255
            _, encoded = cv2.imencode(".png", mask)
            encoded.tofile(mask_path)
        samples.append({"image": path, "split": split, "label": label, "mask": mask_path})
    manifest, _ = build_split(samples, tmp_path, "metal_nut")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, path, manifest


def test_manifest_isolation_and_mask(source):
    root, path, expected = source
    manifest = validate_manifest(root, path)
    train = ManifestDataset(root, manifest, "train", 32)
    validation = ManifestDataset(root, manifest, "validation", 32)
    test = ManifestDataset(root, manifest, "test", 32)
    assert len(train) + len(validation) == 6
    assert {r["image"] for r in train.records}.isdisjoint(r["image"] for r in validation.records)
    assert [r["image"] for r in test.records] == [r["image"] for r in expected["test"]]
    item = next(test[i] for i in range(len(test)) if test[i]["label"])
    assert item["image"].shape == (3, 32, 32)
    assert item["mask"].sum() == 64
    assert not train[0]["mask"].any()


def test_changed_data_rejected(source):
    root, path, manifest = source
    (root / manifest["train"][0]).write_bytes(b"invalid PNG")
    with pytest.raises(ValueError):
        validate_manifest(root, path)


def test_tampered_manifest_rejected(source):
    root, path, manifest = source
    manifest["validation"][0] = manifest["train"][0]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        validate_manifest(root, path)


def test_calibration_and_invalid_scores():
    assert normal_threshold(np.arange(44), 0.95) == 41
    with pytest.raises(ValueError):
        normal_threshold([0.1, np.nan], 0.95)
