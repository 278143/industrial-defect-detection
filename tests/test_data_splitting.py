"""Verify reproducibility and prevent content leakage across partitions."""

import cv2
import numpy as np
import pytest

from defect_detection.data.splitting import build_split


def make_samples(tmp_path, train_values=tuple(range(20, 30)), test_value=200):
    samples = []
    for split, values in (("train", train_values), ("test", (test_value,))):
        for index, value in enumerate(values):
            path = tmp_path / "metal_nut" / split / "good" / f"{index:03}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            success, encoded = cv2.imencode(".png", np.full((8, 8, 3), value, dtype=np.uint8))
            assert success
            encoded.tofile(path)
            samples.append({"image": path, "mask": None, "split": split, "label": "good"})
    return samples


def test_reproducible_and_disjoint(tmp_path):
    samples = make_samples(tmp_path)
    first, audit = build_split(samples, tmp_path, "metal_nut")
    repeated, _ = build_split(list(reversed(samples)), tmp_path, "metal_nut")
    assert first == repeated
    assert audit["passed"]
    assert (len(first["train"]), len(first["validation"]), len(first["test"])) == (8, 2, 1)
    assert not set(first["train"]) & set(first["validation"])
    assert all("/train/good/" in path for path in first["validation"])
    assert len(set(first["train"] + first["validation"])) == 10


def test_training_duplicates_stay_together(tmp_path):
    samples = make_samples(tmp_path, train_values=(20, 20, 21, 22, 23))
    manifest, audit = build_split(samples, tmp_path, "metal_nut", validation_ratio=0.4)
    pair = {"metal_nut/train/good/000.png", "metal_nut/train/good/001.png"}
    assert pair <= set(manifest["train"]) or pair <= set(manifest["validation"])
    assert len(audit["pixel_duplicate_groups"]) == 1
    assert not audit["cross_partition_pixel_overlap"]


def test_train_test_duplicate_blocks_split(tmp_path):
    samples = make_samples(tmp_path, test_value=20)
    manifest, audit = build_split(samples, tmp_path, "metal_nut")
    assert manifest is None
    assert not audit["passed"]
    assert len(audit["source_train_test_leakage"]) == 1


def test_reencoded_duplicate_is_still_detected(tmp_path):
    samples = make_samples(tmp_path, test_value=20)
    test_path = samples[-1]["image"]
    _, encoded = cv2.imencode(
        ".png", np.full((8, 8, 3), 20, dtype=np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    encoded.tofile(test_path)
    assert samples[0]["image"].read_bytes() != test_path.read_bytes()
    manifest, audit = build_split(samples, tmp_path, "metal_nut")
    assert manifest is None
    assert not audit["file_duplicate_groups"]
    assert len(audit["pixel_duplicate_groups"]) == 1


def test_large_duplicate_groups_keep_partitions_nonempty(tmp_path):
    samples = make_samples(tmp_path, train_values=(20,) + (21,) * 9)
    for seed in range(4):
        manifest, _ = build_split(samples, tmp_path, "metal_nut", validation_ratio=0.9, seed=seed)
        assert manifest["train"] and manifest["validation"]


@pytest.mark.parametrize("ratio", [-0.1, 0, 1, 1.1])
def test_invalid_ratio(tmp_path, ratio):
    with pytest.raises(ValueError, match="validation_ratio"):
        build_split(make_samples(tmp_path), tmp_path, "metal_nut", validation_ratio=ratio)


def test_identical_training_pool_cannot_be_split(tmp_path):
    with pytest.raises(ValueError, match="distinct normal image groups"):
        build_split(make_samples(tmp_path, train_values=(20, 20, 20)), tmp_path, "metal_nut")
