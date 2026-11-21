"""Run the complete manifest-based PatchCore baseline, without test-set tuning."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from defect_detection.data.dataset import ManifestDataset, validate_manifest
from defect_detection.evaluation.metrics import evaluate, normal_threshold
from defect_detection.models.patchcore import create_model

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def predict(model, dataset, config):
    scores, maps, masks, labels, paths = [], [], [], [], []
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["image"].to(config["device"]))
            scores.extend(prediction.pred_score.cpu().tolist())
            maps.extend(prediction.anomaly_map[:, 0].cpu().numpy())
            masks.extend(batch["mask"].numpy())
            labels.extend(batch["label"].tolist())
            paths.extend(batch["path"])
    return (
        np.array(scores),
        np.array(maps),
        np.array(masks),
        labels,
        paths,
        time.perf_counter() - start,
    )


def save_previews(dataset, maps, paths, threshold, output):
    import matplotlib

    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = set()
    for index, path in enumerate(paths):
        label = Path(path).parent.name
        if label in selected:
            continue
        selected.add(label)
        rgb = cv2.cvtColor(
            cv2.imdecode(np.fromfile(dataset.resolve(path), dtype=np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        rgb = cv2.resize(rgb, (dataset.size, dataset.size), interpolation=cv2.INTER_AREA)
        heat = cv2.cvtColor(
            cv2.applyColorMap(
                np.uint8(np.clip(maps[index] / max(threshold * 1.5, 1e-8), 0, 1) * 255),
                cv2.COLORMAP_JET,
            ),
            cv2.COLOR_BGR2RGB,
        )
        overlay = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)
        figure, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for axis, image, title in zip(
            axes,
            (rgb, heat, overlay),
            ("Original", "Predicted anomaly heatmap", "Overlay"),
            strict=True,
        ):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(path)
        figure.tight_layout()
        figure.savefig(output / f"{label}_{Path(path).stem}.png", dpi=120)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/patchcore.yaml")
    parser.add_argument(
        "--checkpoint", type=Path, help="Evaluate an existing trusted local checkpoint"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_config = yaml.safe_load((ROOT / "configs/dataset.yaml").read_text(encoding="utf-8"))
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device unavailable")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    manifest_path = ROOT / data_config["split_dir"] / f"{data_config['category']}.json"
    print("Validating manifest and current dataset...", flush=True)
    manifest = validate_manifest(ROOT / data_config["root"], manifest_path)
    datasets = {
        split: ManifestDataset(ROOT / data_config["root"], manifest, split, config["image_size"])
        for split in ("train", "validation", "test")
    }
    print({key: len(value) for key, value in datasets.items()}, flush=True)
    output = ROOT / config["output_dir"]
    if args.checkpoint:
        output = output / "reload_check"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists() and args.checkpoint is None:
        raise FileExistsError("Completed run exists; choose a new output_dir to avoid overwriting")
    start = time.perf_counter()
    model = create_model(config, ROOT / "outputs/cache/torch", pretrained=args.checkpoint is None)
    model.to(config["device"])
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if (
            checkpoint["config"] != config
            or checkpoint["inventory"] != manifest["source_inventory_sha256"]
        ):
            raise ValueError("Checkpoint config/data mismatch")
        model.load_state_dict(checkpoint["state_dict"])
        model.to(config["device"])
        thresholds = checkpoint["thresholds"]
    else:
        model.train()
        model.feature_extractor.eval()
        loader = DataLoader(
            datasets["train"], batch_size=config["batch_size"], shuffle=False, num_workers=0
        )
        with torch.no_grad():
            for index, batch in enumerate(loader):
                model(batch["image"].to(config["device"]))
                if index % 10 == 0:
                    print(f"Extracting normal features: {index + 1}/{len(loader)}", flush=True)
            print("Selecting normal feature coreset...", flush=True)
            model.subsample_embedding(config["coreset_sampling_ratio"])
        print(f"Memory bank: {tuple(model.memory_bank.shape)}", flush=True)
        validation = predict(model, datasets["validation"], config)
        scores, maps = validation[:2]
        thresholds = {
            "image": normal_threshold(scores, config["normal_quantile"]),
            "pixel": normal_threshold(maps.max(axis=(1, 2)), config["normal_quantile"]),
            "method": "normal-only higher empirical quantile; strict greater-than decision",
            "quantile": config["normal_quantile"],
            "validation_count": len(scores),
            "validation_empirical_fpr": None,
        }
        thresholds["validation_empirical_fpr"] = float((scores > thresholds["image"]).mean())
        save_json(output / "thresholds.json", thresholds)
        np.savez_compressed(output / "validation_scores.npz", scores=scores, paths=validation[4])
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": config,
                "inventory": manifest["source_inventory_sha256"],
                "thresholds": thresholds,
            },
            output / "model.pt",
        )
    print("Thresholds frozen. Evaluating held-out test data...", flush=True)
    scores, maps, masks, labels, paths, elapsed = predict(model, datasets["test"], config)
    metrics = evaluate(labels, scores, maps, masks, thresholds["image"], thresholds["pixel"])
    metrics.update(
        {
            "config": config,
            "counts": {key: len(value) for key, value in datasets.items()},
            "execution_mode": "checkpoint_evaluation" if args.checkpoint else "fit_and_evaluate",
            "memory_bank_shape": list(model.memory_bank.shape),
            "test_pipeline_seconds": elapsed,
            "total_seconds": time.perf_counter() - start,
            "timing_scope": "test image loading, preprocessing, inference and CPU transfer; no warmup",
            "gpu": torch.cuda.get_device_name(0) if config["device"] == "cuda" else None,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "anomalib", "torchvision")
            },
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "data_inventory_sha256": manifest["source_inventory_sha256"],
            "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "limitations": [
                "ResNet-18/224/1% coreset is not the paper default configuration",
                "Normal-only validation; small sample calibration does not guarantee deployment FPR",
                "Pixel AUROC uses resized masks at model resolution, not original 700x700",
                "Raw anomaly scores are not probabilities; no test-based threshold optimization",
            ],
        }
    )
    np.savez_compressed(
        output / "test_predictions.npz",
        scores=scores,
        anomaly_maps=maps,
        masks=masks,
        labels=labels,
        paths=paths,
    )
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "is_anomalous", "raw_score", "image_threshold", "prediction"])
        writer.writerows(
            zip(
                paths,
                labels,
                scores.tolist(),
                [thresholds["image"]] * len(scores),
                (scores > thresholds["image"]).tolist(),
                strict=True,
            )
        )
    save_previews(datasets["test"], maps, paths, thresholds["pixel"], output)
    save_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
