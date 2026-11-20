"""Split normal MVTec AD training data and audit exact duplicate leakage."""

import argparse
import json
from pathlib import Path

import yaml
from prepare_data import inspect_category

from defect_detection.data.splitting import build_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/dataset.yaml")
    parser.add_argument("--category")
    parser.add_argument(
        "--force", action="store_true", help="Replace a different existing manifest"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    category = args.category or config["category"]
    if Path(category).name != category or category in (".", ".."):
        parser.error("Category must be a single directory name")
    data_root = PROJECT_ROOT / config["root"]
    category_dir = data_root / category
    manifest_path = PROJECT_ROOT / config["split_dir"] / f"{category}.json"
    audit_path = PROJECT_ROOT / config["output_dir"] / f"{category}_split_audit.json"
    if not category_dir.is_dir():
        parser.error(f"Dataset directory not found: {category_dir}")
    inspection, samples = inspect_category(category_dir)
    if inspection["errors"]:
        write_json(audit_path, {"passed": False, "inspection_errors": inspection["errors"]})
        print(f"Data inspection failed. Audit: {audit_path}")
        return 1
    try:
        manifest, audit = build_split(
            samples,
            data_root,
            category,
            validation_ratio=float(config["validation_ratio"]),
            seed=int(config["seed"]),
        )
    except ValueError as exc:
        write_json(audit_path, {"passed": False, "error": str(exc)})
        print(f"Split failed: {exc}")
        return 1
    if manifest is None:
        write_json(audit_path, audit)
        print(f"Source leakage or conflicting labels found. Audit: {audit_path}")
        print("No new manifest written. Any older manifest must not be used for this dataset.")
        return 1
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest and not args.force:
            audit["passed"] = False
            audit["error"] = (
                "Existing manifest differs; inspect data/config changes before using --force"
            )
            write_json(audit_path, audit)
            print(audit["error"])
            return 1
        if existing == manifest:
            print("Existing manifest matches; keeping it unchanged.")
        else:
            write_json(manifest_path, manifest)
    else:
        write_json(manifest_path, manifest)
    write_json(audit_path, audit)
    print(
        f"Train: {audit['train_count']} | Validation: {audit['validation_count']}"
        f" | Test: {audit['test_count']}"
    )
    print(f"File duplicate groups: {len(audit['file_duplicate_groups'])}")
    print(f"Pixel duplicate groups: {len(audit['pixel_duplicate_groups'])}")
    print("Cross-partition exact duplicates: 0")
    print(f"Manifest: {manifest_path}")
    print(f"Audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
