"""Reconcile the frozen annotation pool before human labeling.

The pool is sampled at the dataset/unit/architecture level.  Therefore a
prediction unit may appear once for each detector architecture without being
an accidental duplicate annotation item.  This script records that decision,
checks the registered manifest, and records whether the current HDFS-only
composition is accepted for the pilot.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def reconcile(pool_path: Path, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(pool_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("annotation pool manifest must be an object")

    by_unit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    annotation_item_ids: list[str] = []
    for row in rows:
        dataset = str(row.get("dataset", ""))
        unit_id = str(row.get("unit_id", ""))
        architecture = str(row.get("architecture", ""))
        item_id = str(row.get("annotation_item_id", ""))
        if not dataset or not unit_id or not architecture or not item_id:
            raise ValueError("every pool row must contain dataset, unit_id, architecture, and annotation_item_id")
        expected_item_id = f"{dataset}:{unit_id}:{architecture}"
        if item_id != expected_item_id:
            raise ValueError(f"annotation_item_id does not match its sampling key: {item_id!r}")
        annotation_item_ids.append(item_id)
        by_unit[(dataset, unit_id)].append(row)

    duplicate_units = []
    for (dataset, unit_id), items in sorted(by_unit.items()):
        if len(items) <= 1:
            continue
        architectures = sorted(str(item["architecture"]) for item in items)
        if len(architectures) != len(set(architectures)):
            raise ValueError(f"duplicate annotation item for {dataset}:{unit_id}")
        duplicate_units.append(
            {
                "dataset": dataset,
                "unit_id": unit_id,
                "row_count": len(items),
                "architectures": architectures,
                "decision": "retain",
                "reason": "one annotation item per detector architecture is the registered sampling unit",
            }
        )

    dataset_counts = dict(sorted(Counter(str(row["dataset"]) for row in rows).items()))
    label_counts = dict(sorted(Counter(str(row["label"]) for row in rows).items()))
    architecture_counts = dict(sorted(Counter(str(row["architecture"]) for row in rows).items()))
    split_counts = dict(sorted(Counter(str(row["split"]) for row in rows).items()))
    unique_prediction_units = len(by_unit)
    duplicate_row_count = len(rows) - unique_prediction_units

    manifest_selected_counts = manifest.get("selected_counts", {})
    checks = {
        "row_count_matches_manifest": len(rows) == manifest.get("selected_item_count"),
        "unique_prediction_unit_count_matches_manifest": unique_prediction_units == manifest.get("selected_unique_prediction_unit_count"),
        "annotation_item_ids_unique": len(annotation_item_ids) == len(set(annotation_item_ids)),
        "dataset_counts_match_manifest": dataset_counts == manifest_selected_counts.get("dataset", {}),
        "label_counts_match_manifest": label_counts == manifest_selected_counts.get("label", {}),
        "architecture_counts_match_manifest": architecture_counts == manifest_selected_counts.get("architecture", {}),
        "split_is_test_only": split_counts == {"test": len(rows)},
        "hdfs_only_pilot_composition_accepted": dataset_counts == {"HDFS": len(rows)},
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"annotation pool reconciliation failed: {failed}")

    report = {
        "schema_version": "1.0",
        "status": "reconciled",
        "reconciled_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pool": {
            "path": str(pool_path),
            "sha256": sha256(pool_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        },
        "counts": {
            "row_count": len(rows),
            "unique_annotation_item_count": len(set(annotation_item_ids)),
            "unique_prediction_unit_count": unique_prediction_units,
            "duplicate_prediction_unit_row_count": duplicate_row_count,
            "dataset": dataset_counts,
            "label": label_counts,
            "architecture": architecture_counts,
            "split": split_counts,
        },
        "duplicate_prediction_units": duplicate_units,
        "decisions": {
            "duplicate_prediction_units": {
                "decision": "retain",
                "reason": "The registered sampling unit is dataset + prediction unit + architecture. The 239 unique HDFS blocks produce 240 distinct annotation items because one block is intentionally represented once for each architecture.",
            },
            "hdfs_only_composition": {
                "decision": "accept_for_current_pilot",
                "composition": {"Normal": label_counts.get("Normal", 0), "Anomaly": label_counts.get("Anomaly", 0)},
                "reason": "The current registered pilot pool is HDFS-only. BGL is prepared and validated separately but is not part of this pilot pool; no replacement sampling is made.",
            },
            "sampling_change": "none",
        },
        "checks": checks,
        "ready_for_step57": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("research/experiments/pilot/annotation_pool.jsonl.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("research/experiments/pilot/annotation_pool_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("research/experiments/pilot/annotation_pool_reconciliation.json"))
    args = parser.parse_args()
    report = reconcile(args.pool, args.manifest, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output), "ready_for_step57": report["ready_for_step57"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
