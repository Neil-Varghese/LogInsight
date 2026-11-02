"""Run the bounded Step 75 ROAR-style remove-and-retrain diagnostic.

This is a limited OOD diagnostic for the primary LogInsight LSTM. For each
registered removal fraction, the script replaces the highest-absolute
Integrated-Gradients events with the learned unknown-event token, retrains a
fresh model on the modified train/validation data, and evaluates the modified
test data. A deterministic random-event variant receives the same per-unit
removal budget. The unmodified model is trained with the same configuration
and seed family as the comparison runs.

The script intentionally does not claim to reproduce the full ROAR benchmark.
It requires attribution artifacts for every bounded train, validation, and
test unit so that both training and evaluation distributions receive the same
replacement operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from detector_interface import binary_classification_metrics, predict_probabilities
    from event_vocabulary import load_vocabulary
    from seed_utils import derive_seed, make_numpy_rng, set_global_seed
    from train_loginsight_lstm import (
        TrainingConfig,
        _resolve_path,
        build_model,
        class_weights,
        load_run_config,
        load_split,
    )
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.detector_interface import binary_classification_metrics, predict_probabilities
    from research.scripts.event_vocabulary import load_vocabulary
    from research.scripts.seed_utils import derive_seed, make_numpy_rng, set_global_seed
    from research.scripts.train_loginsight_lstm import (
        TrainingConfig,
        _resolve_path,
        build_model,
        class_weights,
        load_run_config,
        load_split,
    )


SCRIPT_SCHEMA_VERSION = "1.0"
DEFAULT_FRACTIONS = (0.0, 0.10, 0.30, 0.50)
DEFAULT_MAX_TRAIN_RECORDS = 256
DEFAULT_MAX_VALIDATION_RECORDS = 128
DEFAULT_MAX_TEST_RECORDS = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_jsonl(path: Path):
    import gzip

    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def _read_attributions(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate one IG record per prediction unit."""

    if not path.exists():
        raise FileNotFoundError(path)
    records: dict[str, dict[str, Any]] = {}
    with _open_jsonl(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid attribution JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Attribution record at {path}:{line_number} is not an object")
            if record.get("method") != "integrated_gradients":
                raise ValueError(f"Attribution record at {path}:{line_number} is not Integrated Gradients")
            unit_id = str(record.get("unit_id", ""))
            if not unit_id:
                raise ValueError(f"Attribution record at {path}:{line_number} has no unit_id")
            if unit_id in records:
                raise ValueError(f"Duplicate attribution unit_id {unit_id!r} in {path}")
            events = record.get("events")
            if not isinstance(events, list) or not events:
                raise ValueError(f"Attribution record {unit_id!r} has no events")
            active: list[dict[str, Any]] = []
            for event in events:
                if not isinstance(event, dict):
                    raise ValueError(f"Attribution event for {unit_id!r} is not an object")
                if event.get("is_padding", False):
                    continue
                try:
                    position = int(event["input_position"])
                    event_id = int(event["event_id"])
                    score = float(event["absolute_attribution"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Attribution event for {unit_id!r} is missing aligned fields") from error
                if position < 0 or event_id < 0 or not math.isfinite(score):
                    raise ValueError(f"Attribution event for {unit_id!r} has invalid values")
                active.append({"input_position": position, "event_id": event_id, "score": score})
            active.sort(key=lambda item: (-item["score"], item["input_position"]))
            records[unit_id] = {
                "unit_id": unit_id,
                "sequence_length": int(record.get("sequence_length", len(active))),
                "padded_length": int(record.get("padded_length", len(events))),
                "events": active,
                "source_output_space": record.get("output_space", "probability"),
            }
    if not records:
        raise ValueError(f"Attribution file is empty: {path}")
    return records


def _validate_fraction(fraction: float) -> float:
    value = float(fraction)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("removal fractions must be finite values in [0, 1]")
    return value


def _removed_count(active_count: int, fraction: float) -> int:
    if active_count < 1:
        raise ValueError("each unit must contain at least one active event")
    return min(active_count, int(math.ceil(active_count * fraction))) if fraction > 0 else 0


def select_removed_positions(
    attribution: Mapping[str, Any],
    *,
    fraction: float,
    strategy: str,
    seed: int,
) -> list[dict[str, int]]:
    """Select aligned event positions for IG or equal-budget random removal."""

    fraction = _validate_fraction(fraction)
    if strategy not in {"integrated_gradients", "random_event"}:
        raise ValueError("strategy must be integrated_gradients or random_event")
    active = list(attribution["events"])
    count = _removed_count(len(active), fraction)
    if count == 0:
        return []
    if strategy == "integrated_gradients":
        chosen = active[:count]
    else:
        rng = make_numpy_rng(seed, "roar_random_event", attribution["unit_id"], fraction)
        indices = sorted(int(value) for value in rng.choice(len(active), size=count, replace=False))
        chosen = [active[index] for index in indices]
    return [{"input_position": int(item["input_position"]), "event_id": int(item["event_id"])} for item in chosen]


def replace_top_events(
    inputs: np.ndarray,
    metadata: list[Mapping[str, Any]],
    attributions: Mapping[str, Mapping[str, Any]],
    *,
    fraction: float,
    strategy: str,
    neutral_id: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Replace selected active positions while preserving sequence length."""

    inputs = np.asarray(inputs)
    if inputs.ndim != 2:
        raise ValueError("inputs must be a two-dimensional array")
    if neutral_id < 0:
        raise ValueError("neutral_id must be non-negative")
    if len(metadata) != inputs.shape[0]:
        raise ValueError("metadata and inputs must have the same row count")
    output = inputs.copy()
    changes: list[dict[str, Any]] = []
    for row_index, item in enumerate(metadata):
        unit_id = str(item["unit_id"])
        if unit_id not in attributions:
            raise ValueError(f"missing attribution for unit {unit_id!r}")
        attribution = attributions[unit_id]
        if int(attribution["padded_length"]) != inputs.shape[1]:
            raise ValueError(f"padded length mismatch for unit {unit_id!r}")
        selected = select_removed_positions(
            attribution,
            fraction=fraction,
            strategy=strategy,
            seed=seed,
        )
        positions: list[int] = []
        for selected_event in selected:
            position = int(selected_event["input_position"])
            if position >= inputs.shape[1]:
                raise ValueError(f"attribution position {position} is outside the input for {unit_id!r}")
            if int(inputs[row_index, position]) != int(selected_event["event_id"]):
                raise ValueError(f"attribution/input event mismatch at {unit_id!r}:{position}")
            output[row_index, position] = neutral_id
            positions.append(position)
        changes.append({
            "unit_id": unit_id,
            "row_index": row_index,
            "removed_positions": positions,
            "removed_count": len(positions),
            "active_event_count": len(attribution["events"]),
        })
    return output, changes


def _subset(
    inputs: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if limit is not None and limit < 1:
        raise ValueError("subset limits must be positive or null")
    if limit is None or limit >= len(metadata):
        return inputs, labels, metadata
    return inputs[:limit], labels[:limit], metadata[:limit]


def _train_and_score(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    *,
    vocabulary_size: int,
    config: TrainingConfig,
    seed: int,
    show_progress: bool,
) -> dict[str, Any]:
    """Train one fresh bounded model and return split metrics."""

    set_global_seed(seed, deterministic=config.deterministic)
    model = build_model(vocabulary_size, config)
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required for the ROAR retraining check") from error
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1 if show_progress else 0,
        )
    ]
    fit_kwargs: dict[str, Any] = {
        "x": train_inputs,
        "y": train_labels,
        "validation_data": (validation_inputs, validation_labels),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "callbacks": callbacks,
        "verbose": 1 if show_progress else 0,
    }
    if config.use_class_weights:
        fit_kwargs["class_weight"] = class_weights(train_labels)
    history = model.fit(**fit_kwargs)
    result: dict[str, Any] = {
        "seed": seed,
        "epochs_ran": len(history.history.get("loss", [])),
        "metrics": {},
    }
    for split, inputs, labels in (
        ("train", train_inputs, train_labels),
        ("validation", validation_inputs, validation_labels),
        ("test", test_inputs, test_labels),
    ):
        probabilities = predict_probabilities(model, inputs, config.prediction_batch_size, verbose=0)
        result["metrics"][split] = binary_classification_metrics(labels, probabilities, config.threshold)
    return result


def _parse_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(sorted({_validate_fraction(float(item.strip())) for item in value.split(",") if item.strip()}))
    if not fractions or fractions[0] != 0.0:
        raise ValueError("fractions must include 0 for the unmodified baseline")
    return fractions


def run_roar(
    config_path: Path,
    train_attribution_path: Path,
    validation_attribution_path: Path,
    test_attribution_path: Path,
    *,
    repo_root: Path,
    output_path: Path,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    max_train_records: int | None = DEFAULT_MAX_TRAIN_RECORDS,
    max_validation_records: int | None = DEFAULT_MAX_VALIDATION_RECORDS,
    max_test_records: int | None = DEFAULT_MAX_TEST_RECORDS,
    epochs: int | None = None,
    batch_size: int | None = None,
    seed_override: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Execute and serialize the bounded ROAR diagnostic."""

    fractions = tuple(sorted({_validate_fraction(value) for value in fractions}))
    if not fractions or fractions[0] != 0.0:
        raise ValueError("fractions must include 0 for the unmodified baseline")
    config_path = _resolve_path(config_path, repo_root)
    dataset, base_config = load_run_config(config_path)
    if dataset.architecture != "loginsight_lstm":
        raise ValueError("Step 75 currently targets the registered primary loginsight_lstm detector")
    if seed_override is not None:
        dataset = replace(dataset, seed=int(seed_override))
    config = base_config
    overrides: dict[str, Any] = {}
    if epochs is not None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        overrides["epochs"] = epochs
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        overrides["batch_size"] = batch_size
    if overrides:
        config = replace(config, **overrides)
        config.validate()

    vocabulary = load_vocabulary(_resolve_path(dataset.vocabulary, repo_root))
    paths = {name: _resolve_path(getattr(dataset, name), repo_root) for name in ("train", "validation", "test")}
    loaded = {}
    for split in ("train", "validation", "test"):
        loaded[split] = load_split(paths[split], vocabulary, config, show_progress=show_progress)
    train_inputs, train_labels, train_metadata = _subset(*loaded["train"], max_train_records)
    validation_inputs, validation_labels, validation_metadata = _subset(*loaded["validation"], max_validation_records)
    test_inputs, test_labels, test_metadata = _subset(*loaded["test"], max_test_records)
    attributions = {
        "train": _read_attributions(_resolve_path(train_attribution_path, repo_root)),
        "validation": _read_attributions(_resolve_path(validation_attribution_path, repo_root)),
        "test": _read_attributions(_resolve_path(test_attribution_path, repo_root)),
    }
    for split, metadata in (("train", train_metadata), ("validation", validation_metadata), ("test", test_metadata)):
        missing = sorted({str(item["unit_id"]) for item in metadata} - set(attributions[split]))
        if missing:
            raise ValueError(f"missing {split} attributions for units: {missing[:10]}")

    master_seed = int(dataset.seed)
    report: dict[str, Any] = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "step": 75,
        "status": "completed",
        "diagnostic": "bounded_roar_style_remove_and_retrain",
        "architecture": dataset.architecture,
        "seed": master_seed,
        "neutral_representation": {"type": "unknown_event_token", "id": vocabulary.unknown_id, "token": vocabulary.unknown_token},
        "fractions": list(fractions),
        "strategies": ["integrated_gradients", "random_event"],
        "subset_limits": {"train": max_train_records, "validation": max_validation_records, "test": max_test_records},
        "subset_counts": {"train": len(train_metadata), "validation": len(validation_metadata), "test": len(test_metadata)},
        "inputs": {
            "config": str(_resolve_path(config_path, repo_root)),
            "config_sha256": _sha256(_resolve_path(config_path, repo_root)),
            "train_attribution": str(_resolve_path(train_attribution_path, repo_root)),
            "validation_attribution": str(_resolve_path(validation_attribution_path, repo_root)),
            "test_attribution": str(_resolve_path(test_attribution_path, repo_root)),
        },
        "runs": [],
        "interpretation": "limited OOD diagnostic; not a full ROAR benchmark",
    }
    baseline = _train_and_score(
        train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels,
        vocabulary_size=vocabulary.size, config=config, seed=derive_seed(master_seed, "roar", "unmodified"), show_progress=show_progress,
    )
    baseline_f1 = float(baseline["metrics"]["test"]["f1"])
    report["runs"].append({"strategy": "unmodified", "fraction": 0.0, **baseline, "test_f1_delta_vs_unmodified": 0.0})

    for strategy in ("integrated_gradients", "random_event"):
        for fraction in fractions:
            if fraction == 0.0:
                continue
            variant_inputs: dict[str, np.ndarray] = {}
            changes: dict[str, list[dict[str, Any]]] = {}
            for split, inputs, metadata in (
                ("train", train_inputs, train_metadata),
                ("validation", validation_inputs, validation_metadata),
                ("test", test_inputs, test_metadata),
            ):
                variant_inputs[split], changes[split] = replace_top_events(
                    inputs, metadata, attributions[split], fraction=fraction, strategy=strategy,
                    neutral_id=vocabulary.unknown_id, seed=master_seed,
                )
            run_seed = derive_seed(master_seed, "roar", strategy, f"{fraction:.8f}")
            if show_progress:
                print(f"ROAR {strategy} fraction={fraction:.2f}: training", file=sys.stderr)
            trained = _train_and_score(
                variant_inputs["train"], train_labels,
                variant_inputs["validation"], validation_labels,
                variant_inputs["test"], test_labels,
                vocabulary_size=vocabulary.size, config=config, seed=run_seed, show_progress=show_progress,
            )
            test_f1 = float(trained["metrics"]["test"]["f1"])
            report["runs"].append({
                "strategy": strategy,
                "fraction": fraction,
                "seed": run_seed,
                "metrics": trained["metrics"],
                "epochs_ran": trained["epochs_ran"],
                "test_f1_delta_vs_unmodified": test_f1 - baseline_f1,
                "removed_event_counts": {
                    split: {
                        "total": int(sum(item["removed_count"] for item in values)),
                        "mean": float(np.mean([item["removed_count"] for item in values])) if values else 0.0,
                    }
                    for split, values in changes.items()
                },
            })

    output_path = _resolve_path(output_path, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("research/experiments/lstm_training_config.json"))
    parser.add_argument("--train-attribution", type=Path, required=True)
    parser.add_argument("--validation-attribution", type=Path, required=True)
    parser.add_argument("--test-attribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/step75_roar_check.json"))
    parser.add_argument("--fractions", default=",".join(str(value) for value in DEFAULT_FRACTIONS))
    parser.add_argument("--max-train-records", type=int, default=DEFAULT_MAX_TRAIN_RECORDS)
    parser.add_argument("--max-validation-records", type=int, default=DEFAULT_MAX_VALIDATION_RECORDS)
    parser.add_argument("--max-test-records", type=int, default=DEFAULT_MAX_TEST_RECORDS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    limits = [args.max_train_records, args.max_validation_records, args.max_test_records]
    if any(value == 0 or value < -1 for value in limits):
        raise SystemExit("subset limits must be positive; use -1 for all records")
    limits = [None if value == -1 else value for value in limits]
    report = run_roar(
        args.config,
        args.train_attribution,
        args.validation_attribution,
        args.test_attribution,
        repo_root=args.repo_root.resolve(),
        output_path=args.output,
        fractions=_parse_fractions(args.fractions),
        max_train_records=limits[0],
        max_validation_records=limits[1],
        max_test_records=limits[2],
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed_override=args.seed,
        show_progress=not args.no_progress,
    )
    print(json.dumps({"status": report["status"], "output": str(_resolve_path(args.output, args.repo_root.resolve())), "runs": len(report["runs"])}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
