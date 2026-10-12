"""Calibrate detector probabilities using validation-only temperature scaling.

The detector's raw sigmoid probability is converted to a logit, divided by a
single positive temperature fitted on the validation split, and converted back
to a probability.  This preserves ranking while making probability values
more comparable across detector architectures and seeds.  Test labels are
never used to fit the temperature.

Input files are the ``predictions_*.jsonl.gz`` artifacts emitted by
``train_loginsight_lstm.py``.  The script does not select an alert threshold;
that is the next registered implementation step.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


SCHEMA_VERSION = "1.0"
DEFAULT_ECE_BINS = 10
EPSILON = 1e-7


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
            if "probability" not in record:
                raise ValueError(f"Prediction record {path}:{line_number} has no probability")
            if "label_id" not in record:
                raise ValueError(f"Prediction record {path}:{line_number} has no label_id")
            try:
                probability = float(record["probability"])
                label_id = int(record["label_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Prediction record {path}:{line_number} has invalid probability/label") from exc
            if not 0 <= probability <= 1:
                raise ValueError(f"Prediction record {path}:{line_number} probability is outside [0, 1]")
            if label_id not in (0, 1):
                raise ValueError(f"Prediction record {path}:{line_number} label_id is not binary")
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


def _validate_fit_inputs(labels: np.ndarray, probabilities: np.ndarray) -> None:
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be one-dimensional arrays with equal length")
    if labels.size < 2:
        raise ValueError("At least two validation predictions are required")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Validation predictions must contain both binary classes")
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")


def _logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    negative_values = np.exp(values[~positive])
    result[~positive] = negative_values / (1 + negative_values)
    return result


def _nll(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))


def calibrate_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return _sigmoid(_logits(np.asarray(probabilities, dtype=np.float64)) / temperature)


def fit_temperature(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    grid_size: int = 200,
) -> float:
    """Fit one temperature by validation negative log-likelihood only."""

    _validate_fit_inputs(labels, probabilities)
    if not 0 < min_temperature < max_temperature:
        raise ValueError("temperature bounds must satisfy 0 < min < max")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    logits = _logits(probabilities)

    def objective(log_temperature: float) -> float:
        return _nll(labels, _sigmoid(logits / math.exp(log_temperature)))

    grid = np.linspace(math.log(min_temperature), math.log(max_temperature), grid_size)
    values = np.asarray([objective(value) for value in grid])
    best_index = int(np.argmin(values))
    left_index = max(0, best_index - 1)
    right_index = min(grid_size - 1, best_index + 1)
    left = float(grid[left_index])
    right = float(grid[right_index])
    if left == right:
        return math.exp(left)

    # Golden-section refinement in log-temperature space.  The one-dimensional
    # objective is smooth and this avoids an extra SciPy dependency.
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = objective(x2)
    return float(math.exp((left + right) / 2.0))


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int = DEFAULT_ECE_BINS) -> dict[str, Any]:
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional arrays with equal length")
    if bins < 1:
        raise ValueError("bins must be positive")
    clipped = np.clip(probabilities, 0, 1)
    bin_indices = np.minimum((clipped * bins).astype(int), bins - 1)
    ece = 0.0
    bin_stats: list[dict[str, Any]] = []
    for index in range(bins):
        mask = bin_indices == index
        count = int(np.sum(mask))
        if count:
            mean_probability = float(np.mean(clipped[mask]))
            observed_rate = float(np.mean(labels[mask]))
            ece += count / labels.size * abs(mean_probability - observed_rate)
        else:
            mean_probability = None
            observed_rate = None
        bin_stats.append({
            "bin": index,
            "lower_inclusive": index / bins,
            "upper_exclusive": (index + 1) / bins,
            "count": count,
            "mean_probability": mean_probability,
            "observed_positive_rate": observed_rate,
        })
    return {
        "count": int(labels.size),
        "nll": _nll(labels, clipped),
        "brier": float(np.mean((clipped - labels) ** 2)),
        "ece": float(ece),
        "ece_bins": bins,
        "bin_stats": bin_stats,
    }


def calibrate_records(
    records: Iterable[dict[str, Any]],
    temperature: float,
) -> list[dict[str, Any]]:
    calibrated: list[dict[str, Any]] = []
    for record in records:
        raw_probability = float(record["probability"])
        calibrated_probability = float(calibrate_probabilities(np.asarray([raw_probability]), temperature)[0])
        output = dict(record)
        output.pop("prediction", None)
        output.pop("threshold", None)
        output["raw_probability"] = raw_probability
        output["probability"] = calibrated_probability
        output["calibrated_probability"] = calibrated_probability
        output["confidence"] = max(calibrated_probability, 1.0 - calibrated_probability)
        output["calibration_method"] = "temperature_scaling"
        output["temperature"] = temperature
        calibrated.append(output)
    return calibrated


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def calibrate_files(
    validation_path: Path,
    split_paths: dict[str, Path],
    output_dir: Path,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    ece_bins: int = DEFAULT_ECE_BINS,
    show_progress: bool = True,
) -> dict[str, Any]:
    validation_probabilities, validation_labels, validation_records = load_predictions(
        validation_path, show_progress=show_progress
    )
    temperature = fit_temperature(
        validation_labels,
        validation_probabilities,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "validation_before": calibration_metrics(validation_labels, validation_probabilities, bins=ece_bins),
        "validation_after": calibration_metrics(
            validation_labels,
            calibrate_probabilities(validation_probabilities, temperature),
            bins=ece_bins,
        ),
    }
    output_paths: dict[str, str] = {}
    for split, input_path in split_paths.items():
        probabilities, labels, records = load_predictions(input_path, show_progress=show_progress)
        calibrated = calibrate_probabilities(probabilities, temperature)
        metrics[f"{split}_before"] = calibration_metrics(labels, probabilities, bins=ece_bins)
        metrics[f"{split}_after"] = calibration_metrics(labels, calibrated, bins=ece_bins)
        output_path = output_dir / f"calibrated_predictions_{split}.jsonl.gz"
        write_records(output_path, calibrate_records(records, temperature))
        output_paths[split] = str(output_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": "temperature_scaling",
        "fit_split": "validation",
        "validation_input": str(validation_path),
        "input_predictions": {split: str(path) for split, path in split_paths.items()},
        "output_predictions": output_paths,
        "temperature": temperature,
        "temperature_bounds": {
            "min": min_temperature,
            "max": max_temperature,
        },
        "ece_bins": ece_bins,
        "threshold_selection": "deferred_to_step_23",
        "metrics": metrics,
    }
    manifest_path = output_dir / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-temperature", type=float, default=0.05)
    parser.add_argument("--max-temperature", type=float, default=20.0)
    parser.add_argument("--ece-bins", type=int, default=DEFAULT_ECE_BINS)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = calibrate_files(
        validation_path=args.validation_predictions,
        split_paths={
            "train": args.train_predictions,
            "validation": args.validation_predictions,
            "test": args.test_predictions,
        },
        output_dir=args.output_dir,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
        ece_bins=args.ece_bins,
        show_progress=not args.no_progress,
    )
    print(json.dumps({
        "manifest": str(args.output_dir / "calibration_manifest.json"),
        "method": manifest["method"],
        "fit_split": manifest["fit_split"],
        "temperature": manifest["temperature"],
        "validation_before": manifest["metrics"]["validation_before"],
        "validation_after": manifest["metrics"]["validation_after"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
