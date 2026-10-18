"""Verify transferred Colab detector artifacts against local inference.

R5 hashes the local copies of checkpoints and stored predictions, reruns the
same frozen detector locally, and compares probabilities by prediction-unit
ID. The verifier never overwrites the stored prediction files. The registered
numeric tolerance is read from ``experiment_config.json`` unless overridden
explicitly for a diagnostic run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from train_loginsight_lstm import load_run_config, load_split
    from detector_interface import DetectorContract, KerasDetectorAdapter, predict_probabilities
    from event_vocabulary import load_vocabulary
    from train_logrobust_attention_bilstm import build_model as build_logrobust_model, TemporalAttention
    from train_loginsight_lstm import build_model as build_lstm_model
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.train_loginsight_lstm import load_run_config, load_split
    from research.scripts.detector_interface import DetectorContract, KerasDetectorAdapter, predict_probabilities
    from research.scripts.event_vocabulary import load_vocabulary
    from research.scripts.train_logrobust_attention_bilstm import build_model as build_logrobust_model, TemporalAttention
    from research.scripts.train_loginsight_lstm import build_model as build_lstm_model


SCHEMA_VERSION = "1.0"
DEFAULT_TOLERANCE = 1e-6
TARGETS = {
    "loginsight_lstm": {
        "config": Path("research/experiments/lstm_training_config.json"),
        "seed": 13,
        "checkpoint": Path("research/models/checkpoints/loginsight_lstm/best_seed_13.keras"),
        "prediction_dir": Path("research/models/checkpoints/loginsight_lstm"),
    },
    "logrobust_attention_bilstm": {
        "config": Path("research/experiments/logrobust_attention_bilstm_training_config.json"),
        "seed": 202,
        "checkpoint": Path("research/models/checkpoints/logrobust_attention_bilstm/best_seed_202.keras"),
        "prediction_dir": Path("research/models/checkpoints/logrobust_attention_bilstm"),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_jsonl(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def load_prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    with _open_jsonl(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("unit_id"):
                raise ValueError(f"{path}:{line_number} must contain a prediction object with unit_id")
            unit_id = str(row["unit_id"])
            if unit_id in rows:
                raise ValueError(f"{path} contains duplicate unit_id {unit_id!r}")
            probability = row.get("probability")
            if not isinstance(probability, (int, float)) or not math.isfinite(float(probability)):
                raise ValueError(f"{path}:{line_number} has a non-finite probability")
            rows[unit_id] = row
    if not rows:
        raise ValueError(f"{path} contains no prediction rows")
    return rows


def compare_predictions(
    stored: dict[str, dict[str, Any]],
    local_probabilities: Iterable[float],
    metadata: Iterable[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    metadata = list(metadata)
    probabilities = np.asarray(list(local_probabilities), dtype=np.float64)
    if probabilities.shape != (len(metadata),):
        raise ValueError("local probabilities do not match metadata count")
    local_ids = [str(record["unit_id"]) for record in metadata]
    if len(set(local_ids)) != len(local_ids):
        raise ValueError("local metadata contains duplicate unit_id values")
    missing = [unit_id for unit_id in local_ids if unit_id not in stored]
    extra = [unit_id for unit_id in stored if unit_id not in set(local_ids)]
    if missing or extra:
        raise ValueError(f"prediction-unit mismatch: missing={len(missing)} extra={len(extra)}")
    deltas = np.asarray(
        [abs(float(stored[unit_id]["probability"]) - float(probability)) for unit_id, probability in zip(local_ids, probabilities)],
        dtype=np.float64,
    )
    mismatched_metadata = 0
    for record in metadata:
        row = stored[str(record["unit_id"])]
        if row.get("architecture") not in (None, record.get("architecture")):
            mismatched_metadata += 1
        if row.get("label_id") is not None and int(row["label_id"]) != int(record["label_id"]):
            mismatched_metadata += 1
    return {
        "stored_count": len(stored),
        "local_count": len(metadata),
        "max_absolute_probability_delta": float(np.max(deltas)),
        "mean_absolute_probability_delta": float(np.mean(deltas)),
        "within_tolerance_count": int(np.sum(deltas <= tolerance)),
        "outside_tolerance_count": int(np.sum(deltas > tolerance)),
        "metadata_mismatch_count": mismatched_metadata,
        "tolerance": tolerance,
        "status": "pass" if bool(np.all(deltas <= tolerance)) and mismatched_metadata == 0 else "fail",
    }


def _load_registered_tolerance(repo_root: Path) -> float:
    path = repo_root / "research/experiments/experiment_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    tolerance = payload.get("attribution_validation", {}).get("numeric_tolerance", DEFAULT_TOLERANCE)
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError("registered numeric tolerance must be positive")
    return float(tolerance)


def _source_environment(manifest_path: Path) -> dict[str, Any]:
    source: dict[str, Any] = {"manifest": str(manifest_path.resolve())}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        source["manifest_paths_are_colab"] = any(
            "/content/" in str(value)
            for value in (payload.get("checkpoints", {}), payload.get("inputs", {}), payload.get("output_dir", ""))
        )
        source["source_manifest_sha256"] = sha256_file(manifest_path)
    return source


def _predict_with_progress(model: Any, inputs: np.ndarray, batch_size: int, label: str) -> np.ndarray:
    """Predict in chunks while showing rows, rate, and ETA on stderr."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    total = int(inputs.shape[0])
    chunks: list[np.ndarray] = []
    completed = 0
    started = time.monotonic()
    last_update = 0.0
    for start in range(0, total, batch_size):
        batch = inputs[start : start + batch_size]
        chunks.append(predict_probabilities(model, batch, len(batch), verbose=0))
        completed += len(batch)
        now = time.monotonic()
        if completed == total or now - last_update >= 0.5:
            elapsed = max(now - started, 1e-9)
            rate = completed / elapsed
            eta = (total - completed) / rate if rate else float("inf")
            print(
                f"\rR5 {label}: {completed:,}/{total:,} rows "
                f"({completed / max(total, 1) * 100:6.2f}%) "
                f"{rate:,.0f} rows/s ETA {eta:,.0f}s",
                end="",
                file=sys.stderr,
                flush=True,
            )
            last_update = now
    print(file=sys.stderr)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)


def verify_target(repo_root: Path, architecture: str, tolerance: float, splits: tuple[str, ...], batch_size: int | None = None) -> dict[str, Any]:
    target = TARGETS[architecture]
    config_path = repo_root / target["config"]
    dataset, config = load_run_config(config_path)
    dataset = replace(dataset, seed=int(target["seed"]), architecture=architecture)
    vocabulary = load_vocabulary(repo_root / dataset.vocabulary)
    checkpoint = repo_root / target["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if architecture == "logrobust_attention_bilstm":
        adapter = KerasDetectorAdapter(
            DetectorContract(architecture),
            build_logrobust_model,
            custom_objects={"TemporalAttention": TemporalAttention, "_TemporalAttention": TemporalAttention},
        )
    else:
        adapter = KerasDetectorAdapter(DetectorContract(architecture), build_lstm_model)
    model = adapter.load_model(checkpoint)
    split_reports: dict[str, Any] = {}
    for split in splits:
        data_path = repo_root / getattr(dataset, split)
        inputs, labels, metadata = load_split(data_path, vocabulary, config, show_progress=False)
        stored_path = repo_root / target["prediction_dir"] / f"predictions_{split}.jsonl.gz"
        stored = load_prediction_rows(stored_path)
        effective_batch_size = config.prediction_batch_size if batch_size is None else batch_size
        probabilities = _predict_with_progress(model, inputs, effective_batch_size, f"{architecture}/{split}")
        split_reports[split] = {
            "input": {"path": str(data_path.resolve()), "sha256": sha256_file(data_path), "count": len(metadata)},
            "stored_predictions": {"path": str(stored_path.resolve()), "sha256": sha256_file(stored_path)},
            "comparison": compare_predictions(stored, probabilities, metadata, tolerance=tolerance),
        }
    manifest_path = repo_root / target["prediction_dir"] / f"run_manifest_seed_{target['seed']}.json"
    return {
        "architecture": architecture,
        "seed": int(target["seed"]),
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "source": _source_environment(manifest_path),
        "splits": split_reports,
        "status": "pass" if all(item["comparison"]["status"] == "pass" for item in split_reports.values()) else "fail",
    }


def verify_transfer(repo_root: Path, architectures: tuple[str, ...], splits: tuple[str, ...], tolerance: float | None = None, batch_size: int | None = None) -> dict[str, Any]:
    registered_tolerance = _load_registered_tolerance(repo_root)
    effective_tolerance = registered_tolerance if tolerance is None else float(tolerance)
    if effective_tolerance <= 0:
        raise ValueError("tolerance must be positive")
    targets = [verify_target(repo_root, architecture, effective_tolerance, splits, batch_size) for architecture in architectures]
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": "r5_colab_transfer_verification",
        "registered_tolerance": registered_tolerance,
        "effective_tolerance": effective_tolerance,
        "tolerance_source": "experiment_config.attribution_validation.numeric_tolerance",
        "architectures": targets,
        "status": "pass" if all(target["status"] == "pass" for target in targets) else "fail",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--architecture", choices=tuple(TARGETS) + ("both",), default="both")
    parser.add_argument("--split", choices=("train", "validation", "test"), nargs="+", default=("train", "validation", "test"))
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--batch-size", type=int, help="Inference batch size; defaults to each registered detector config")
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/r5_colab_transfer_verification.json"))
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    architectures = tuple(TARGETS) if args.architecture == "both" else (args.architecture,)
    report = verify_transfer(repo_root, architectures, tuple(args.split), args.tolerance, args.batch_size)
    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "tolerance": report["effective_tolerance"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
