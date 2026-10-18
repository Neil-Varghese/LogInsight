"""Build annotation-pool candidates from detector prediction metadata.

The annotation sampler deliberately consumes a small, architecture-neutral
candidate schema. This builder converts the shared detector prediction files
for the held-out split into that schema while preserving the detector seed and
probability as provenance fields. One prediction file per architecture is
expected; duplicate dataset/unit/architecture records are rejected so that
multiple seeds cannot be mixed accidentally.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DATASETS = {"HDFS", "BGL"}


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield record


def normalize_prediction(
    record: dict[str, Any], line_number: int, eligible_split: str
) -> dict[str, Any]:
    required = {
        "dataset": record.get("dataset"),
        "unit_id": record.get("unit_id"),
        "split": record.get("split"),
        "label": record.get("label"),
        "sequence_length": record.get("sequence_length"),
        "confidence": record.get("confidence"),
        "architecture": record.get("architecture"),
        "seed": record.get("seed"),
    }
    missing = [name for name, value in required.items() if value is None or value == ""]
    if missing:
        raise ValueError(f"Prediction line {line_number} is missing: {', '.join(missing)}")
    if required["dataset"] not in DATASETS:
        raise ValueError(f"Prediction line {line_number} has unsupported dataset")
    if required["split"] != eligible_split:
        raise ValueError(
            f"Prediction line {line_number} uses split {required['split']!r}; "
            f"expected {eligible_split!r}"
        )
    if required["label"] not in {"Normal", "Anomaly"}:
        raise ValueError(f"Prediction line {line_number} has an invalid label")
    try:
        sequence_length = int(required["sequence_length"])
        confidence = float(required["confidence"])
        probability = float(record["probability"])
        seed = int(required["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Prediction line {line_number} has invalid numeric metadata") from exc
    if sequence_length < 1:
        raise ValueError(f"Prediction line {line_number} has non-positive sequence length")
    if not 0 <= confidence <= 1 or not 0 <= probability <= 1:
        raise ValueError(f"Prediction line {line_number} has probability/confidence outside [0, 1]")
    if seed < 0:
        raise ValueError(f"Prediction line {line_number} has a negative seed")
    return {
        "dataset": required["dataset"],
        "unit_id": str(required["unit_id"]),
        "split": required["split"],
        "label": required["label"],
        "sequence_length": sequence_length,
        "confidence": confidence,
        "architecture": str(required["architecture"]),
        "detector_probability": probability,
        "detector_prediction": int(record.get("prediction", probability >= 0.5)),
        "detector_seed": seed,
    }


def build_candidates(prediction_files: Iterable[Path], eligible_split: str = "test") -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in prediction_files:
        for line_number, record in enumerate(iter_records(path), start=1):
            candidate = normalize_prediction(record, line_number, eligible_split)
            key = (candidate["dataset"], candidate["unit_id"], candidate["architecture"])
            if key in seen:
                raise ValueError(f"Duplicate dataset/unit/architecture candidate: {key}")
            seen.add(key)
            candidates.append(candidate)
    candidates.sort(key=lambda item: (item["dataset"], item["unit_id"], item["architecture"]))
    return candidates


def write_candidates(path: Path, candidates: Iterable[dict[str, Any]], force: bool = False) -> int:
    if path.exists() and not force:
        raise FileExistsError(f"Candidate metadata already exists: {path}; use --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        count = 0
        for candidate in candidates:
            handle.write(json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-file",
        type=Path,
        action="append",
        required=True,
        help="Detector prediction JSONL/JSONL.GZ file; repeat once per architecture",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/experiments/pilot/annotation_candidates.jsonl"),
    )
    parser.add_argument("--eligible-split", default="test")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for path in args.prediction_file:
        if not path.is_file():
            raise FileNotFoundError(path)
    candidates = build_candidates(args.prediction_file, args.eligible_split)
    count = write_candidates(args.output, candidates, force=args.force)
    print(json.dumps({
        "output": str(args.output),
        "candidate_count": count,
        "architectures": dict(sorted(Counter(item["architecture"] for item in candidates).items())),
        "datasets": dict(sorted(Counter(item["dataset"] for item in candidates).items())),
        "detector_seeds": dict(sorted(Counter(item["detector_seed"] for item in candidates).items())),
        "eligible_split": args.eligible_split,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
