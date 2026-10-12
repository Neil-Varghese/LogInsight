"""Select and apply an anomaly threshold using validation predictions only.

The default registered rule maximizes validation F1. Ties are resolved by
choosing the threshold closest to the provisional 0.5 reference and then the
lower threshold. The selected threshold is frozen before it is applied to the
test split; test labels are reported for evaluation only and never influence
selection.

The input files may be raw prediction files from ``train_loginsight_lstm.py``
or calibrated prediction files from ``calibrate_detector.py``. The selected
probability field defaults to ``probability``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


SCHEMA_VERSION = "1.0"
DEFAULT_REFERENCE_THRESHOLD = 0.5


class ProgressBar:
    """Dependency-free progress display for long prediction-file scans."""

    def __init__(self, label: str, total: int | None = None, enabled: bool = True) -> None:
        self.label = label
        self.total = total
        self.enabled = enabled
        self.last_update = 0.0

    def update(self, current: int, suffix: str = "", force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_update < 0.15:
            return
        self.last_update = now
        width = 30
        if self.total is None:
            filled = (current // 100) % (width + 1)
            bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
            message = f"{self.label} [{bar}] records={current:,} {suffix}".rstrip()
        else:
            fraction = min(max(current, 0) / max(self.total, 1), 1.0)
            filled = int(width * fraction)
            bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
            message = f"{self.label} [{bar}] {fraction * 100:6.2f}% {suffix}".rstrip()
        print(f"\r{message}", end="", file=sys.stderr, flush=True)
        if self.total is not None and current >= self.total:
            print(file=sys.stderr)

    def finish(self, current: int, suffix: str = "") -> None:
        self.update(current, suffix=suffix, force=True)
        if self.total is None:
            print(file=sys.stderr)


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def iter_prediction_records(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if "label_id" not in record or "probability" not in record:
                raise ValueError(f"Prediction record {path}:{line_number} needs label_id and probability")
            try:
                label_id = int(record["label_id"])
                probability = float(record["probability"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Prediction record {path}:{line_number} has invalid label/probability") from exc
            if label_id not in (0, 1):
                raise ValueError(f"Prediction record {path}:{line_number} label_id is not binary")
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Prediction record {path}:{line_number} probability is outside [0, 1]")
            yield record


def load_predictions(path: Path, show_progress: bool = True) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    probabilities: list[float] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    progress = ProgressBar(f"Reading {path.name}", enabled=show_progress)
    for record in iter_prediction_records(path):
        probabilities.append(float(record["probability"]))
        labels.append(int(record["label_id"]))
        records.append(record)
        progress.update(len(records), suffix=f"records={len(records):,}")
    progress.finish(len(records), suffix=f"records={len(records):,}")
    if not records:
        raise ValueError(f"Prediction file is empty: {path}")
    return np.asarray(probabilities, dtype=np.float64), np.asarray(labels, dtype=np.int32), records


def binary_threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    predictions = (probabilities >= threshold).astype(np.int32)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": int(labels.size),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "threshold": threshold,
        "accuracy": float(np.mean(predictions == labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    """Return thresholds covering every distinct predicted-positive set."""

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    unique = np.unique(np.clip(probabilities, 0, 1))
    candidates = {0.0, 1.0}
    candidates.update(float(value) for value in unique)
    for lower, upper in zip(unique[:-1], unique[1:]):
        candidates.add(float((lower + upper) / 2.0))
    return np.asarray(sorted(candidates), dtype=np.float64)


def select_validation_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    reference_threshold: float = DEFAULT_REFERENCE_THRESHOLD,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Select the validation-F1 threshold with deterministic tie-breaking."""

    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be one-dimensional arrays with equal length")
    if labels.size < 2 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Validation predictions must contain both binary classes")
    if not 0 <= reference_threshold <= 1:
        raise ValueError("reference_threshold must be between 0 and 1")
    candidates = threshold_candidates(probabilities)
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        results.append(binary_threshold_metrics(labels, probabilities, float(candidate)))
    best = min(
        results,
        key=lambda metrics: (
            -metrics["f1"],
            abs(metrics["threshold"] - reference_threshold),
            metrics["threshold"],
        ),
    )
    return float(best["threshold"]), best, results


def threshold_records(
    records: Iterable[dict[str, Any]],
    threshold: float,
    threshold_source: str = "validation",
) -> list[dict[str, Any]]:
    output_records: list[dict[str, Any]] = []
    for record in records:
        output = dict(record)
        output["prediction"] = int(float(record["probability"]) >= threshold)
        output["threshold"] = threshold
        output["threshold_source"] = threshold_source
        output["confidence"] = max(float(record["probability"]), 1.0 - float(record["probability"]))
        output_records.append(output)
    return output_records


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def select_and_apply_threshold(
    validation_path: Path,
    split_paths: dict[str, Path],
    output_dir: Path,
    reference_threshold: float = DEFAULT_REFERENCE_THRESHOLD,
    show_progress: bool = True,
) -> dict[str, Any]:
    validation_probabilities, validation_labels, validation_records = load_predictions(
        validation_path, show_progress=show_progress
    )
    selected_threshold, validation_metrics, candidates = select_validation_threshold(
        validation_labels,
        validation_probabilities,
        reference_threshold=reference_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    split_metrics: dict[str, Any] = {}
    for split, input_path in split_paths.items():
        probabilities, labels, records = load_predictions(input_path, show_progress=show_progress)
        split_metrics[split] = binary_threshold_metrics(labels, probabilities, selected_threshold)
        output_path = output_dir / f"thresholded_predictions_{split}.jsonl.gz"
        write_records(output_path, threshold_records(records, selected_threshold))
        output_paths[split] = str(output_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_metric": "f1",
        "selection_rule": "maximize_validation_f1; tie_break_closest_to_reference_then_lower_threshold",
        "fit_split": "validation",
        "test_labels_used_for_selection": False,
        "reference_threshold": reference_threshold,
        "selected_threshold": selected_threshold,
        "validation_input": str(validation_path),
        "input_predictions": {split: str(path) for split, path in split_paths.items()},
        "output_predictions": output_paths,
        "validation_metrics": validation_metrics,
        "metrics_by_split": split_metrics,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    manifest_path = output_dir / "threshold_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-threshold", type=float, default=DEFAULT_REFERENCE_THRESHOLD)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = select_and_apply_threshold(
        validation_path=args.validation_predictions,
        split_paths={
            "train": args.train_predictions,
            "validation": args.validation_predictions,
            "test": args.test_predictions,
        },
        output_dir=args.output_dir,
        reference_threshold=args.reference_threshold,
        show_progress=not args.no_progress,
    )
    print(json.dumps({
        "manifest": str(args.output_dir / "threshold_manifest.json"),
        "fit_split": manifest["fit_split"],
        "selection_metric": manifest["selection_metric"],
        "selected_threshold": manifest["selected_threshold"],
        "validation_metrics": manifest["validation_metrics"],
        "test_metrics": manifest["metrics_by_split"]["test"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
