"""Record reproducible Step 25 LSTM baseline results.

This script consumes the artifacts produced by Steps 22--24 under one detector
run directory. It reports thresholded precision/recall/F1 and false positives,
average-precision PR-AUC, and probability-calibration metrics for each frozen
split. The PR-AUC field uses average precision (the step-function area under
the precision-recall curve), which is recorded explicitly in the manifest.

No fitting, threshold selection, or model inference happens here.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    from calibrate_detector import calibration_metrics, load_predictions
    from select_threshold import binary_threshold_metrics
except ImportError:  # pragma: no cover - supports package-style imports
    from research.scripts.calibrate_detector import calibration_metrics, load_predictions
    from research.scripts.select_threshold import binary_threshold_metrics


SCHEMA_VERSION = "1.0"
SPLITS = ("train", "validation", "test")


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute average precision without requiring scikit-learn."""

    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be one-dimensional arrays with equal length")
    if labels.size == 0:
        raise ValueError("average_precision requires at least one record")
    positive_count = int(np.sum(labels == 1))
    if positive_count == 0:
        return 0.0
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, labels.size + 1)
    precision_at_rank = true_positives / ranks
    return float(np.sum(precision_at_rank[sorted_labels == 1]) / positive_count)


def _open_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _load_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield record


def _thresholded_metrics(path: Path, threshold: float) -> dict[str, Any]:
    labels: list[int] = []
    probabilities: list[float] = []
    for record in _load_jsonl_gz(path):
        labels.append(int(record["label_id"]))
        probabilities.append(float(record["probability"]))
    label_array = np.asarray(labels, dtype=np.int32)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    metrics = binary_threshold_metrics(label_array, probability_array, threshold)
    metrics["pr_auc"] = average_precision(label_array, probability_array)
    metrics["false_positives_per_1000_units"] = metrics["false_positives"] / metrics["count"] * 1000
    return metrics


def _probability_metrics(path: Path, ece_bins: int) -> dict[str, Any]:
    probabilities, labels, _ = load_predictions(path, show_progress=False)
    metrics = calibration_metrics(labels, probabilities, bins=ece_bins)
    metrics["pr_auc"] = average_precision(labels, probabilities)
    return metrics


def record_baseline(
    run_dir: Path,
    output_path: Path,
    ece_bins: int = 10,
) -> dict[str, Any]:
    run_manifest_path = run_dir / "run_manifest_seed_13.json"
    calibration_manifest_path = run_dir / "calibrated" / "calibration_manifest.json"
    threshold_manifest_path = run_dir / "thresholded" / "threshold_manifest.json"
    required = [run_manifest_path, calibration_manifest_path, threshold_manifest_path]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    run_manifest = _open_json(run_manifest_path)
    calibration_manifest = _open_json(calibration_manifest_path)
    threshold_manifest = _open_json(threshold_manifest_path)
    selected_threshold = float(threshold_manifest["selected_threshold"])
    temperature = float(calibration_manifest["temperature"])

    metrics: dict[str, Any] = {}
    for split in SPLITS:
        raw_path = run_dir / f"predictions_{split}.jsonl.gz"
        calibrated_path = run_dir / "calibrated" / f"calibrated_predictions_{split}.jsonl.gz"
        thresholded_path = run_dir / "thresholded" / f"thresholded_predictions_{split}.jsonl.gz"
        for path in (raw_path, calibrated_path, thresholded_path):
            if not path.exists():
                raise FileNotFoundError(path)
        raw_probability, raw_labels, _ = load_predictions(raw_path, show_progress=False)
        calibrated_probability, calibrated_labels, _ = load_predictions(calibrated_path, show_progress=False)
        if not np.array_equal(raw_labels, calibrated_labels):
            raise ValueError(f"Raw and calibrated labels differ for split {split}")
        raw_probability_metrics = calibration_metrics(raw_labels, raw_probability, bins=ece_bins)
        raw_probability_metrics["pr_auc"] = average_precision(raw_labels, raw_probability)
        calibrated_probability_metrics = calibration_metrics(
            calibrated_labels, calibrated_probability, bins=ece_bins
        )
        calibrated_probability_metrics["pr_auc"] = average_precision(calibrated_labels, calibrated_probability)
        metrics[split] = {
            "raw_probability": raw_probability_metrics,
            "calibrated_probability": calibrated_probability_metrics,
            "thresholded": _thresholded_metrics(thresholded_path, selected_threshold),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "step": 25,
        "detector_id": "loginsight_lstm",
        "architecture": run_manifest.get("architecture", "loginsight_lstm"),
        "dataset": "HDFS",
        "seed": int(run_manifest.get("seed", 13)),
        "run_directory": str(run_dir),
        "checkpoint_paths": run_manifest.get("checkpoints", {}),
        "pr_auc_definition": "average_precision_step_function",
        "calibration_method": calibration_manifest.get("method", "temperature_scaling"),
        "calibration_temperature": temperature,
        "selected_threshold": selected_threshold,
        "threshold_source": threshold_manifest.get("fit_split", "validation"),
        "test_labels_used_for_threshold_selection": threshold_manifest.get(
            "test_labels_used_for_selection", False
        ),
        "ece_bins": ece_bins,
        "metrics_by_split": metrics,
        "source_manifests": {
            "run": str(run_manifest_path),
            "calibration": str(calibration_manifest_path),
            "threshold": str(threshold_manifest_path),
        },
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ece-bins", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = record_baseline(args.run_dir, args.output, ece_bins=args.ece_bins)
    print(json.dumps({
        "output": str(args.output),
        "detector_id": result["detector_id"],
        "dataset": result["dataset"],
        "seed": result["seed"],
        "selected_threshold": result["selected_threshold"],
        "test": result["metrics_by_split"]["test"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
