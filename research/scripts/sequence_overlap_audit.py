"""Audit exact event-sequence overlap and novel-sequence detector metrics.

R1 checks whether identical event-ID sequences occur across the registered
HDFS train and test splits. It also reports detector metrics on test units
whose exact sequence was not present in training and counts unique sequences
represented by the annotation pool.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


SCRIPT_SCHEMA_VERSION = "1.0"


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


def sequence_key(record: dict[str, Any], path: Path) -> str:
    sequence = record.get("Sequence")
    if not isinstance(sequence, list):
        raise ValueError(f"{path} record {record.get('BlockId', record.get('unit_id'))!r} lacks a Sequence list")
    if any(not isinstance(event_id, str) for event_id in sequence):
        raise ValueError(f"{path} contains a non-string event ID in Sequence")
    canonical = json.dumps(sequence, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sequence_summary(path: Path) -> tuple[set[str], dict[str, str], int]:
    sequence_keys: set[str] = set()
    unit_to_sequence: dict[str, str] = {}
    record_count = 0
    for record in iter_records(path):
        record_count += 1
        unit_id = record.get("BlockId", record.get("unit_id"))
        if unit_id in (None, ""):
            raise ValueError(f"{path} contains a record without BlockId/unit_id")
        unit_key = str(unit_id)
        if unit_key in unit_to_sequence:
            raise ValueError(f"{path} contains duplicate unit {unit_key!r}")
        key = sequence_key(record, path)
        unit_to_sequence[unit_key] = key
        sequence_keys.add(key)
    if record_count == 0:
        raise ValueError(f"{path} is empty")
    return sequence_keys, unit_to_sequence, record_count


def _load_annotation_units(path: Path) -> list[str]:
    units: list[str] = []
    seen_items: set[str] = set()
    for record in iter_records(path):
        unit_id = record.get("unit_id")
        if unit_id in (None, ""):
            raise ValueError(f"{path} contains an annotation record without unit_id")
        unit_key = str(unit_id)
        item_id = record.get("annotation_item_id")
        if item_id not in (None, ""):
            item_key = str(item_id)
            if item_key in seen_items:
                raise ValueError(f"{path} contains duplicate annotation_item_id {item_key!r}")
            seen_items.add(item_key)
        units.append(unit_key)
    if not units:
        raise ValueError(f"{path} is empty")
    return units


def _binary_metrics(labels: list[int], predictions: list[int], probabilities: list[float]) -> dict[str, Any]:
    if not labels:
        raise ValueError("cannot calculate metrics on an empty subset")
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    positives = sum(labels)
    average_precision = None
    if positives:
        order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
        seen_positive = 0
        precision_sum = 0.0
        for rank, index in enumerate(order, start=1):
            if labels[index] == 1:
                seen_positive += 1
                precision_sum += seen_positive / rank
        average_precision = precision_sum / positives
    return {
        "count": len(labels),
        "positive_count": positives,
        "negative_count": len(labels) - positives,
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _prediction_metrics(path: Path, novel_units: set[str]) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    architecture = None
    seed = None
    total_count = 0
    for record in iter_records(path):
        total_count += 1
        if record.get("split") not in ("test", "test_seed_13", None):
            raise ValueError(f"{path} contains a non-test prediction record")
        unit_id = record.get("unit_id")
        if unit_id in (None, ""):
            raise ValueError(f"{path} contains a prediction without unit_id")
        if str(unit_id) not in novel_units:
            continue
        try:
            label = int(record["label_id"])
            prediction = int(record["prediction"])
            probability = float(record["probability"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} contains an invalid prediction record") from error
        if label not in (0, 1) or prediction not in (0, 1) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{path} contains an invalid binary prediction or probability")
        labels.append(label)
        predictions.append(prediction)
        probabilities.append(probability)
        architecture = architecture or record.get("architecture")
        seed = seed if seed is not None else record.get("seed")
    if total_count == 0:
        raise ValueError(f"{path} is empty")
    if not labels:
        raise ValueError(f"{path} has no predictions on the novel-sequence subset")
    result = _binary_metrics(labels, predictions, probabilities)
    result.update({
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "architecture": architecture,
        "seed": seed,
        "total_prediction_count": total_count,
        "novel_sequence_prediction_count": len(labels),
    })
    return result


def build_audit(
    *,
    train_evidence: Path,
    test_evidence: Path,
    annotation_pool: Path,
    prediction_files: Iterable[Path],
    output: Path,
    validation_evidence: Path | None = None,
) -> dict[str, Any]:
    train_sequences, _, train_count = _sequence_summary(train_evidence)
    test_sequences, test_units, test_count = _sequence_summary(test_evidence)
    shared_sequences = train_sequences & test_sequences
    test_unit_sequence_overlap = sum(key in train_sequences for key in test_units.values())
    novel_units = {unit_id for unit_id, key in test_units.items() if key not in train_sequences}

    annotation_units = _load_annotation_units(annotation_pool)
    annotation_sequence_map: dict[str, str] = {}
    evidence_paths = [train_evidence, test_evidence]
    if validation_evidence is not None:
        evidence_paths.append(validation_evidence)
    for evidence_path in evidence_paths:
        for record in iter_records(evidence_path):
            unit_id = str(record.get("BlockId", record.get("unit_id", "")))
            if unit_id in annotation_units:
                annotation_sequence_map[unit_id] = sequence_key(record, evidence_path)
    missing_annotation_units = sorted(set(annotation_units) - set(annotation_sequence_map))
    if missing_annotation_units:
        raise ValueError(f"annotation units missing from supplied evidence: {missing_annotation_units[:5]}")
    annotation_sequences = set(annotation_sequence_map.values())

    prediction_results = [_prediction_metrics(path, novel_units) for path in prediction_files]
    manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "audit": "hdfs_exact_event_sequence_overlap",
        "sequence_definition": "exact ordered Sequence event-ID list, without padding",
        "train": {
            "path": str(train_evidence.resolve()),
            "sha256": sha256(train_evidence),
            "unit_count": train_count,
            "unique_sequence_count": len(train_sequences),
        },
        "test": {
            "path": str(test_evidence.resolve()),
            "sha256": sha256(test_evidence),
            "unit_count": test_count,
            "unique_sequence_count": len(test_sequences),
            "units_with_sequence_seen_in_train": test_unit_sequence_overlap,
            "units_with_novel_sequence": len(novel_units),
            "novel_unit_fraction": len(novel_units) / test_count,
            "shared_unique_sequence_count": len(shared_sequences),
            "shared_unique_sequence_fraction_of_test_sequences": len(shared_sequences) / len(test_sequences),
        },
        "annotation_pool": {
            "path": str(annotation_pool.resolve()),
            "sha256": sha256(annotation_pool),
            "annotation_item_count": len(annotation_units),
            "unique_unit_count": len(set(annotation_units)),
            "unique_sequence_count": len(annotation_sequences),
            "duplicate_sequence_item_count": len(annotation_units) - len(annotation_sequences),
        },
        "novel_sequence_metrics": prediction_results,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-evidence", type=Path, default=Path("research/data/processed/hdfs_block_evidence_train.jsonl.gz"))
    parser.add_argument("--test-evidence", type=Path, default=Path("research/data/processed/hdfs_block_evidence_test.jsonl.gz"))
    parser.add_argument("--validation-evidence", type=Path, default=None)
    parser.add_argument("--annotation-pool", type=Path, default=Path("research/experiments/pilot/annotation_pool.jsonl.gz"))
    parser.add_argument("--prediction-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/sequence_overlap_audit.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_audit(
        train_evidence=args.train_evidence,
        test_evidence=args.test_evidence,
        validation_evidence=args.validation_evidence,
        annotation_pool=args.annotation_pool,
        prediction_files=args.prediction_file,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
