"""Load only manifest-selected images; validate the manifest against current data."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .splitting import build_split


def validate_manifest(data_root, manifest_path):
    data_root = Path(data_root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    category = manifest["category"]
    if Path(category).name != category or category in (".", ".."):
        raise ValueError("Invalid category")
    source = []
    category_dir = data_root / category
    for split in ("train", "test"):
        for path in sorted((category_dir / split).glob("*/*.png")):
            mask = None
            if split == "test" and path.parent.name != "good":
                mask = category_dir / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
            source.append({"image": path, "mask": mask, "split": split, "label": path.parent.name})
    current, audit = build_split(
        source, data_root, category, manifest["requested_validation_ratio"], manifest["seed"]
    )
    if not audit["passed"] or current != manifest:
        raise ValueError("Data or split manifest changed; rerun split_data.py and review the audit")
    return manifest


class ManifestDataset(Dataset):
    """RGB images use ImageNet normalization; masks use nearest-neighbor resizing."""

    def __init__(self, data_root, manifest, split, image_size=224):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.root = Path(data_root).resolve()
        self.size = image_size
        if split == "test":
            self.records = manifest[split]
        else:
            self.records = [
                {"image": path, "label": "good", "mask": None} for path in manifest[split]
            ]
        if not self.records:
            raise ValueError(f"Empty split: {split}")
        for record in self.records:
            self.resolve(record["image"])
            if record["mask"]:
                self.resolve(record["mask"])

    def resolve(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Invalid dataset path: {relative}")
        return path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imdecode(
            np.fromfile(self.resolve(record["image"]), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError(f"Cannot decode: {record['image']}")
        image = cv2.cvtColor(
            cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255)
        tensor = (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        if record["mask"]:
            original = cv2.imdecode(
                np.fromfile(self.resolve(record["mask"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if original is None:
                raise ValueError(f"Cannot decode: {record['mask']}")
            mask = cv2.resize(original, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return {
            "image": tensor,
            "mask": torch.from_numpy(mask > 0),
            "label": record["label"] != "good",
            "path": record["image"],
        }
