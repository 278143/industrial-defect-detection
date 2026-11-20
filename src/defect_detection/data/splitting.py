"""Create reproducible normal-only validation splits and audit exact duplicates."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _duplicate_groups(records, key):
    groups = defaultdict(list)
    for record in records:
        groups[record[key]].append(record)
    return [group for group in groups.values() if len(group) > 1]


def build_split(samples, data_root: Path, category: str, validation_ratio=0.2, seed=42):
    """Return (manifest, audit); manifest is None when source data leak or conflict."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be strictly between 0 and 1")
    data_root = data_root.resolve()
    records = []
    for sample in sorted(samples, key=lambda item: str(item["image"])):
        image_path = sample["image"]
        raw = image_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        pixel_digest = hashlib.sha256()
        pixel_digest.update(f"{image.shape}|{image.dtype}|".encode("ascii"))
        pixel_digest.update(image.tobytes())
        mask_path = sample["mask"]
        records.append(
            {
                "image": image_path.resolve().relative_to(data_root).as_posix(),
                "source_split": sample["split"],
                "label": sample["label"],
                "mask": mask_path.resolve().relative_to(data_root).as_posix()
                if mask_path
                else None,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "pixel_sha256": pixel_digest.hexdigest(),
                "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest()
                if mask_path
                else None,
            }
        )
    train_source = [r for r in records if r["source_split"] == "train"]
    test_source = [r for r in records if r["source_split"] == "test"]
    if not train_source or not test_source:
        raise ValueError("Training and test data must both be present")
    if any(r["label"] != "good" for r in train_source):
        raise ValueError("Only normal images may be used for training and validation")
    file_groups = _duplicate_groups(records, "file_sha256")
    pixel_groups = _duplicate_groups(records, "pixel_sha256")
    leakage = [g for g in pixel_groups if len({r["source_split"] for r in g}) > 1]
    conflicts = [g for g in pixel_groups if len({r["label"] for r in g}) > 1]

    def describe(groups):
        return [
            [
                {"image": r["image"], "source_split": r["source_split"], "label": r["label"]}
                for r in group
            ]
            for group in groups
        ]

    inventory_digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit = {
        "category": category,
        "seed": seed,
        "requested_validation_ratio": validation_ratio,
        "source_train_count": len(train_source),
        "source_test_count": len(test_source),
        "source_inventory_sha256": inventory_digest,
        "file_duplicate_groups": describe(file_groups),
        "pixel_duplicate_groups": describe(pixel_groups),
        "source_train_test_leakage": describe(leakage),
        "conflicting_labels": describe(conflicts),
        "scope": "Exact file bytes and decoded pixels; near-duplicates are not assessed.",
        "passed": not leakage and not conflicts,
    }
    if not audit["passed"]:
        return None, audit
    grouped_train = defaultdict(list)
    for record in train_source:
        grouped_train[record["pixel_sha256"]].append(record)
    groups = sorted(grouped_train.values(), key=lambda group: group[0]["image"])
    if len(groups) < 2:
        raise ValueError("At least two distinct normal image groups are needed for a holdout")
    random.Random(seed).shuffle(groups)
    target = max(1, min(len(train_source) - 1, int(len(train_source) * validation_ratio + 0.5)))
    validation = []
    training = []
    for group in groups:
        if abs(len(validation) + len(group) - target) < abs(len(validation) - target):
            validation.extend(group)
        else:
            training.extend(group)
    if not validation:
        smallest = min(groups, key=lambda group: (len(group), group[0]["image"]))
        validation = list(smallest)
        chosen = {r["image"] for r in validation}
        training = [r for r in train_source if r["image"] not in chosen]
    if not training:
        # A large group can overshoot the target; return one whole group to training.
        returned = min(
            groups, key=lambda g: (abs(len(validation) - len(g) - target), g[0]["image"])
        )
        training = list(returned)
        returned_paths = {r["image"] for r in returned}
        validation = [r for r in validation if r["image"] not in returned_paths]
    hash_sets = [
        {r["pixel_sha256"] for r in partition} for partition in (training, validation, test_source)
    ]
    cross_partition_overlap = any(
        hash_sets[i] & hash_sets[j] for i in range(3) for j in range(i + 1, 3)
    )
    if cross_partition_overlap:
        raise ValueError("Unexpected duplicate overlap between output partitions")
    audit.update(
        {
            "train_count": len(training),
            "validation_count": len(validation),
            "test_count": len(test_source),
            "actual_validation_ratio": len(validation) / len(train_source),
            "cross_partition_pixel_overlap": False,
        }
    )
    manifest = {
        "schema_version": 1,
        "category": category,
        "path_base": "dataset root from configs/dataset.yaml",
        "seed": seed,
        "requested_validation_ratio": validation_ratio,
        "source_inventory_sha256": inventory_digest,
        "algorithm": "Seeded shuffle of sorted exact-pixel groups; keep groups together.",
        "train": sorted(r["image"] for r in training),
        "validation": sorted(r["image"] for r in validation),
        "test": [
            {"image": r["image"], "label": r["label"], "mask": r["mask"]}
            for r in sorted(test_source, key=lambda r: r["image"])
        ],
    }
    return manifest, audit
