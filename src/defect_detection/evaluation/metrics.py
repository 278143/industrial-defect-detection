"""Calibrate only on normal validation data; evaluate without test-time tuning."""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def normal_threshold(values, quantile):
    values = np.asarray(values)
    if not 0 < quantile < 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Need finite normal validation scores and a quantile in (0, 1)")
    return float(np.quantile(values, quantile, method="higher"))


def evaluate(labels, scores, maps, masks, image_threshold, pixel_threshold):
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(scores) > image_threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[False, True]).ravel()
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "pixel_auroc_at_model_resolution": float(roc_auc_score(masks.ravel(), maps.ravel())),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "pixel_rule": "raw anomaly map > normal-validation maximum-map quantile",
    }
