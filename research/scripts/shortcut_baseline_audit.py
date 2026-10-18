"""Run R2 shortcut baselines and compare them with detector predictions.

The event-count baseline is a training-only bag-of-event-template-count
logistic regression. The length-only baseline uses only the number of events
in a block. Both baselines use the full unpadded research ``Sequence`` and a
fixed 0.5 decision threshold; no test labels are used for fitting or tuning.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


SCRIPT_SCHEMA_VERSION = "1.0"
DEFAULT_THRESHOLD = 0.5


def _open_jsonl(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with _open_jsonl(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"record in {path} at line {line_number} must be an object")
            yield record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence(record: dict[str, Any], path: Path) -> list[str]:
    sequence = record.get("Sequence")
    if not isinstance(sequence, list) or any(not isinstance(event_id, str) for event_id in sequence):
        raise ValueError(f"{path} contains a record with an invalid Sequence list")
    return sequence


def _label(record: dict[str, Any], path: Path) -> int:
    try:
        label = int(record["LabelId"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} contains a record without a valid LabelId") from error
    if label not in (0, 1):
        raise ValueError(f"{path} contains a non-binary LabelId")
    return label


def _fit_vocabulary(path: Path) -> dict[str, int]:
    event_ids: set[str] = set()
    for record in iter_records(path):
        event_ids.update(_sequence(record, path))
    if not event_ids:
        raise ValueError(f"{path} contains no event IDs")
    return {event_id: index for index, event_id in enumerate(sorted(event_ids))}


def _vectorize(path: Path, vocabulary: dict[str, int]) -> tuple[csr_matrix, np.ndarray, np.ndarray, list[str]]:
    row_indices = array("i")
    column_indices = array("i")
    values = array("d")
    labels: list[int] = []
    lengths: list[int] = []
    unit_ids: list[str] = []
    row_count = 0
    for record in iter_records(path):
        sequence = _sequence(record, path)
        labels.append(_label(record, path))
        lengths.append(len(sequence))
        unit_id = record.get("BlockId", record.get("unit_id"))
        if unit_id in (None, ""):
            raise ValueError(f"{path} contains a record without BlockId/unit_id")
        unit_ids.append(str(unit_id))
        counts = Counter(event_id for event_id in sequence if event_id in vocabulary)
        for event_id, count in counts.items():
            row_indices.append(row_count)
            column_indices.append(vocabulary[event_id])
            values.append(float(count))
        row_count += 1
    if row_count == 0:
        raise ValueError(f"{path} is empty")
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError(f"{path} contains duplicate prediction units")
    matrix = csr_matrix(
        (np.asarray(values), (np.asarray(row_indices), np.asarray(column_indices))),
        shape=(row_count, len(vocabulary)),
        dtype=np.float64,
    )
    return matrix, np.asarray(labels, dtype=np.int32), np.asarray(lengths, dtype=np.float64), unit_ids


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
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
        "average_precision": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _fit_and_evaluate(
    name: str,
    train_features: np.ndarray | csr_matrix,
    validation_features: np.ndarray | csr_matrix,
    test_features: np.ndarray | csr_matrix,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, Any]:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=300,
        random_state=2026,
        solver="liblinear",
    )
    model.fit(train_features, train_labels)
    results = {}
    for split, features, labels in (
        ("train", train_features, train_labels),
        ("validation", validation_features, validation_labels),
        ("test", test_features, test_labels),
    ):
        probabilities = model.predict_proba(features)[:, 1]
        results[split] = _metrics(labels, probabilities)
    return {
        "name": name,
        "fit_split": "train",
        "class_weight": "balanced",
        "threshold_policy": "fixed_0.5_no_test_tuning",
        "metrics": results,
        "coefficients": int(model.coef_.shape[1]),
    }


def _detector_metrics(path: Path, expected_test_units: set[str]) -> dict[str, Any]:
    labels: list[int] = []
    probabilities: list[float] = []
    unit_ids: set[str] = set()
    architecture = None
    seed = None
    threshold = DEFAULT_THRESHOLD
    for record in iter_records(path):
        unit_id = record.get("unit_id")
        if unit_id in (None, ""):
            raise ValueError(f"{path} contains a prediction without unit_id")
        unit_key = str(unit_id)
        if unit_key in unit_ids:
            raise ValueError(f"{path} contains duplicate unit_id {unit_key!r}")
        unit_ids.add(unit_key)
        if unit_key not in expected_test_units:
            raise ValueError(f"{path} contains unit {unit_key!r} outside the test evidence")
        label = int(record["label_id"])
        probability = float(record["probability"])
        if label not in (0, 1) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{path} contains an invalid label or probability")
        labels.append(label)
        probabilities.append(probability)
        architecture = architecture or record.get("architecture")
        seed = seed if seed is not None else record.get("seed")
        threshold = float(record.get("threshold", threshold))
    if unit_ids != expected_test_units:
        raise ValueError(f"{path} does not contain exactly the test evidence units")
    metric = _metrics(np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float64), threshold)
    return {
        "name": "detector",
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "architecture": architecture,
        "seed": seed,
        "metrics": {"test": metric},
    }


def build_audit(
    *,
    train_evidence: Path,
    validation_evidence: Path,
    test_evidence: Path,
    prediction_files: list[Path],
    output: Path,
) -> dict[str, Any]:
    vocabulary = _fit_vocabulary(train_evidence)
    train_matrix, train_labels, train_lengths, train_units = _vectorize(train_evidence, vocabulary)
    validation_matrix, validation_labels, validation_lengths, validation_units = _vectorize(validation_evidence, vocabulary)
    test_matrix, test_labels, test_lengths, test_units = _vectorize(test_evidence, vocabulary)
    test_unit_set = set(test_units)
    baselines = [
        _fit_and_evaluate(
            "event_count_logistic_regression",
            train_matrix,
            validation_matrix,
            test_matrix,
            train_labels,
            validation_labels,
            test_labels,
        ),
        _fit_and_evaluate(
            "length_only_logistic_regression",
            train_lengths.reshape(-1, 1),
            validation_lengths.reshape(-1, 1),
            test_lengths.reshape(-1, 1),
            train_labels,
            validation_labels,
            test_labels,
        ),
    ]
    detectors = [_detector_metrics(path, test_unit_set) for path in prediction_files]
    report = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "audit": "shortcut_baselines",
        "feature_definition": {
            "event_count": "training-vocabulary bag-of-event-ID counts from the full unpadded Sequence",
            "length_only": "number of events in the full unpadded Sequence",
            "unknown_test_events": "ignored because vocabulary is fitted on training only",
        },
        "fit_policy": {
            "fit_split": "train",
            "class_weight": "balanced",
            "threshold": DEFAULT_THRESHOLD,
            "test_labels_used_for_fit_or_tuning": False,
        },
        "inputs": {
            "train": {"path": str(train_evidence.resolve()), "sha256": sha256(train_evidence), "unit_count": len(train_units)},
            "validation": {"path": str(validation_evidence.resolve()), "sha256": sha256(validation_evidence), "unit_count": len(validation_units)},
            "test": {"path": str(test_evidence.resolve()), "sha256": sha256(test_evidence), "unit_count": len(test_units)},
        },
        "training_event_vocabulary_size": len(vocabulary),
        "baselines": baselines,
        "detectors": detectors,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-evidence", type=Path, default=Path("research/data/processed/hdfs_block_evidence_train.jsonl.gz"))
    parser.add_argument("--validation-evidence", type=Path, default=Path("research/data/processed/hdfs_block_evidence_validation.jsonl.gz"))
    parser.add_argument("--test-evidence", type=Path, default=Path("research/data/processed/hdfs_block_evidence_test.jsonl.gz"))
    parser.add_argument("--prediction-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/shortcut_baseline_audit.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_audit(
        train_evidence=args.train_evidence,
        validation_evidence=args.validation_evidence,
        test_evidence=args.test_evidence,
        prediction_files=args.prediction_file,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
