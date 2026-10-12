"""Train and evaluate the leakage-free LogInsight research LSTM.

The script consumes split-specific JSONL records with a ``Sequence`` field and
``LabelId`` (or ``Label``) field.  It fits no vocabulary and never combines
splits: the vocabulary must be supplied as a manifest created from the
training split only.  The default architecture mirrors the repository's
current LSTM notebook while making all experiment-affecting choices explicit.

Training is intentionally an explicit command-line operation.  This module is
safe to import in lightweight test environments because TensorFlow is imported
only when a model is built or loaded.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

try:
    from detector_interface import (
        DetectorContract,
        KerasDetectorAdapter,
        binary_classification_metrics,
        make_detector_batch,
        prediction_records,
        predict_probabilities,
        write_prediction_records,
    )
    from event_vocabulary import EventVocabulary, load_vocabulary
    from seed_utils import set_global_seed
except ImportError:  # pragma: no cover - supports package-style imports
    from research.scripts.detector_interface import (
        DetectorContract,
        KerasDetectorAdapter,
        binary_classification_metrics,
        make_detector_batch,
        prediction_records,
        predict_probabilities,
        write_prediction_records,
    )
    from research.scripts.event_vocabulary import EventVocabulary, load_vocabulary
    from research.scripts.seed_utils import set_global_seed


SCRIPT_SCHEMA_VERSION = "1.0"
DEFAULT_MAX_SEQUENCE_LENGTH = 50
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    """All choices that affect a detector run, excluding input records."""

    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    padding: str = "pre"
    truncating: str = "pre"
    embedding_dim: int = 128
    first_lstm_units: int = 128
    second_lstm_units: int = 64
    dense_units: int = 32
    first_lstm_dropout: float = 0.3
    second_lstm_dropout: float = 0.3
    dense_dropout: float = 0.2
    mask_zero: bool = False
    learning_rate: float = 0.001
    batch_size: int = 128
    prediction_batch_size: int = 256
    epochs: int = 20
    early_stopping_patience: int = 3
    reduce_lr_patience: int = 2
    threshold: float = DEFAULT_THRESHOLD
    deterministic: bool = True
    use_class_weights: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown training configuration fields: {', '.join(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if self.padding not in {"pre", "post"}:
            raise ValueError("padding must be 'pre' or 'post'")
        if self.truncating not in {"pre", "post"}:
            raise ValueError("truncating must be 'pre' or 'post'")
        for name in (
            "embedding_dim",
            "first_lstm_units",
            "second_lstm_units",
            "dense_units",
            "batch_size",
            "prediction_batch_size",
            "epochs",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "first_lstm_dropout",
            "second_lstm_dropout",
            "dense_dropout",
        ):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.early_stopping_patience < 0 or self.reduce_lr_patience < 0:
            raise ValueError("callback patience values must be non-negative")


@dataclass(frozen=True)
class DatasetConfig:
    train: Path
    validation: Path
    test: Path
    vocabulary: Path
    output_dir: Path
    seed: int = 2026
    architecture: str = "loginsight_lstm"


class ProgressBar:
    """Small dependency-free progress display for long record scans."""

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


def _unit_id(record: dict[str, Any], line_number: int) -> str:
    value = record.get("unit_id", record.get("BlockId", record.get("WindowId")))
    if value is None or value == "":
        raise ValueError(f"Record {line_number} has no unit_id, BlockId, or WindowId")
    return str(value)


def _label_id(record: dict[str, Any], line_number: int) -> int:
    value = record.get("LabelId")
    if value is None:
        label = record.get("Label")
        if label == "Normal":
            value = 0
        elif label == "Anomaly":
            value = 1
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Record {line_number} has no valid binary LabelId/Label") from exc
    if value not in (0, 1):
        raise ValueError(f"Record {line_number} has non-binary label: {value}")
    return value


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield record


def count_records(path: Path) -> int:
    return sum(1 for _ in iter_records(path))


def encode_and_pad(
    sequence: list[Any],
    vocabulary: EventVocabulary,
    max_length: int,
    padding: str = "pre",
    truncating: str = "pre",
) -> tuple[np.ndarray, int]:
    """Encode one event sequence and return its fixed input plus original length."""

    if max_length < 1:
        raise ValueError("max_length must be positive")
    if padding not in {"pre", "post"} or truncating not in {"pre", "post"}:
        raise ValueError("padding and truncating must be 'pre' or 'post'")
    original_length = len(sequence)
    encoded = vocabulary.encode([str(event) for event in sequence])
    if len(encoded) > max_length:
        encoded = encoded[-max_length:] if truncating == "pre" else encoded[:max_length]
    result = np.full(max_length, vocabulary.pad_id, dtype=np.int32)
    if encoded:
        values = np.asarray(encoded, dtype=np.int32)
        if padding == "pre":
            result[-len(values):] = values
        else:
            result[: len(values)] = values
    return result, original_length


def load_split(
    path: Path,
    vocabulary: EventVocabulary,
    config: TrainingConfig,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load one split while retaining IDs and original sequence lengths."""

    total = count_records(path)
    inputs: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    progress = ProgressBar(f"Loading {path.name}", total=total, enabled=show_progress)
    for line_number, record in enumerate(iter_records(path), start=1):
        unit_id = _unit_id(record, line_number)
        if unit_id in seen_ids:
            raise ValueError(f"Duplicate prediction unit {unit_id!r} in {path}")
        seen_ids.add(unit_id)
        sequence = record.get("Sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"Record {line_number} in {path} has no non-empty Sequence list")
        padded, original_length = encode_and_pad(
            sequence,
            vocabulary,
            config.max_sequence_length,
            config.padding,
            config.truncating,
        )
        inputs.append(padded)
        labels.append(_label_id(record, line_number))
        metadata.append({
            "unit_id": unit_id,
            "label": "Anomaly" if labels[-1] else "Normal",
            "label_id": labels[-1],
            "sequence_length": original_length,
        })
        progress.update(len(inputs), suffix=f"units={len(inputs):,}")
    progress.finish(len(inputs), suffix=f"units={len(inputs):,}")
    if not inputs:
        raise ValueError(f"Split is empty: {path}")
    return np.stack(inputs).astype(np.int32), np.asarray(labels, dtype=np.int32), metadata


def assert_disjoint_splits(
    split_metadata: dict[str, list[dict[str, Any]]],
) -> None:
    """Refuse accidental unit overlap before training or evaluation."""

    seen: dict[str, str] = {}
    for split, records in split_metadata.items():
        for record in records:
            unit_id = record["unit_id"]
            prior = seen.get(unit_id)
            if prior is not None:
                raise ValueError(f"Prediction unit {unit_id!r} occurs in both {prior} and {split}")
            seen[unit_id] = split


def build_model(vocabulary_size: int, config: TrainingConfig):
    """Create the repository-faithful research LSTM."""

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("TensorFlow is required for model training/evaluation") from exc
    config.validate()
    model = models.Sequential([
        layers.Input(shape=(config.max_sequence_length,), dtype="int32"),
        layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=config.embedding_dim,
            mask_zero=config.mask_zero,
        ),
        layers.LSTM(config.first_lstm_units, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(config.first_lstm_dropout),
        layers.LSTM(config.second_lstm_units),
        layers.BatchNormalization(),
        layers.Dropout(config.second_lstm_dropout),
        layers.Dense(config.dense_units, activation="relu"),
        layers.Dropout(config.dense_dropout),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(labels.astype(np.int32), minlength=2)
    if np.any(counts == 0):
        return {int(index): 1.0 for index in range(2) if counts[index] > 0}
    total = int(counts.sum())
    return {index: total / (2.0 * int(counts[index])) for index in range(2)}


def _binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    return binary_classification_metrics(labels, probabilities, threshold)


def _predict(model, inputs: np.ndarray, batch_size: int) -> np.ndarray:
    return predict_probabilities(model, inputs, batch_size)


def write_predictions(
    path: Path,
    metadata: list[dict[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    architecture: str,
    seed: int,
) -> None:
    split = path.name.split("predictions_", 1)[-1].split(".jsonl", 1)[0]
    labels = np.asarray([int(record["label_id"]) for record in metadata], dtype=np.int32)
    batch = make_detector_batch(
        split,
        np.zeros((len(metadata), 1), dtype=np.int32),
        labels,
        metadata,
    )
    contract = DetectorContract(architecture)
    records = prediction_records(contract, batch, probabilities, threshold, seed)
    write_prediction_records(path, records)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_run_config(path: Path) -> tuple[DatasetConfig, TrainingConfig]:
    payload = _load_json(path)
    try:
        data = payload["data"]
        training = payload.get("training", {})
    except KeyError as exc:
        raise ValueError(f"Training configuration is missing {exc.args[0]!r}") from exc
    required = ("train", "validation", "test", "vocabulary", "output_dir")
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"Training configuration is missing: {', '.join(missing)}")
    dataset = DatasetConfig(
        train=Path(data["train"]),
        validation=Path(data["validation"]),
        test=Path(data["test"]),
        vocabulary=Path(data["vocabulary"]),
        output_dir=Path(data["output_dir"]),
        seed=int(data.get("seed", 2026)),
        architecture=str(data.get("architecture", "loginsight_lstm")),
    )
    return dataset, TrainingConfig.from_dict(training)


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def run_training(
    dataset: DatasetConfig,
    config: TrainingConfig,
    repo_root: Path,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train once, save artifacts, and return the run manifest."""

    config.validate()
    seed_report = set_global_seed(dataset.seed, deterministic=config.deterministic)
    vocabulary = load_vocabulary(_resolve_path(dataset.vocabulary, repo_root))
    paths = {
        name: _resolve_path(getattr(dataset, name), repo_root)
        for name in ("train", "validation", "test", "output_dir")
    }
    for name in ("train", "validation", "test"):
        if not paths[name].exists():
            raise FileNotFoundError(paths[name])
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    loaded: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    for split in ("train", "validation", "test"):
        loaded[split] = load_split(paths[split], vocabulary, config, show_progress=show_progress)
    metadata = {split: value[2] for split, value in loaded.items()}
    assert_disjoint_splits(metadata)

    adapter = KerasDetectorAdapter(DetectorContract(dataset.architecture), build_model)
    model = adapter.build_model(vocabulary.size, config)
    checkpoint_path = paths["output_dir"] / f"best_seed_{dataset.seed}.keras"
    final_model_path = paths["output_dir"] / f"final_seed_{dataset.seed}.keras"
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("TensorFlow is required for model training/evaluation") from exc
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=config.reduce_lr_patience,
            verbose=1,
        ),
    ]
    fit_kwargs: dict[str, Any] = {
        "x": loaded["train"][0],
        "y": loaded["train"][1],
        "validation_data": (loaded["validation"][0], loaded["validation"][1]),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "callbacks": callbacks,
        "verbose": 1,
    }
    if config.use_class_weights:
        fit_kwargs["class_weight"] = class_weights(loaded["train"][1])
    history = model.fit(**fit_kwargs)
    model.save(final_model_path)

    metrics: dict[str, Any] = {}
    for split, (inputs, labels, split_metadata) in loaded.items():
        probabilities = _predict(model, inputs, config.prediction_batch_size)
        metrics[split] = binary_metrics(labels, probabilities, config.threshold)
        write_predictions(
            paths["output_dir"] / f"predictions_{split}.jsonl.gz",
            split_metadata,
            probabilities,
            config.threshold,
            dataset.architecture,
            dataset.seed,
        )

    history_path = paths["output_dir"] / f"history_seed_{dataset.seed}.json"
    history_path.write_text(json.dumps(history.history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "architecture": dataset.architecture,
        "seed": dataset.seed,
        "seed_report": seed_report,
        "vocabulary": vocabulary.to_dict(),
        "inputs": {name: str(paths[name]) for name in ("train", "validation", "test")},
        "output_dir": str(paths["output_dir"]),
        "checkpoints": {"best": str(checkpoint_path), "final": str(final_model_path)},
        "training_config": asdict(config),
        "split_counts": {split: len(records) for split, records in metadata.items()},
        "metrics": metrics,
        "history_path": str(history_path),
    }
    manifest_path = paths["output_dir"] / f"run_manifest_seed_{dataset.seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_evaluation(
    dataset: DatasetConfig,
    config: TrainingConfig,
    checkpoint: Path,
    repo_root: Path,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Evaluate an existing Keras checkpoint on all three frozen splits."""

    config.validate()
    seed_report = set_global_seed(dataset.seed, deterministic=config.deterministic)
    vocabulary = load_vocabulary(_resolve_path(dataset.vocabulary, repo_root))
    paths = {
        name: _resolve_path(getattr(dataset, name), repo_root)
        for name in ("train", "validation", "test", "output_dir")
    }
    checkpoint_path = _resolve_path(checkpoint, repo_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    loaded: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    for split in ("train", "validation", "test"):
        loaded[split] = load_split(paths[split], vocabulary, config, show_progress=show_progress)
    metadata = {split: value[2] for split, value in loaded.items()}
    assert_disjoint_splits(metadata)

    adapter = KerasDetectorAdapter(DetectorContract(dataset.architecture), build_model)
    model = adapter.load_model(checkpoint_path)
    metrics: dict[str, Any] = {}
    for split, (inputs, labels, split_metadata) in loaded.items():
        probabilities = _predict(model, inputs, config.prediction_batch_size)
        metrics[split] = binary_metrics(labels, probabilities, config.threshold)
        write_predictions(
            paths["output_dir"] / f"predictions_{split}.jsonl.gz",
            split_metadata,
            probabilities,
            config.threshold,
            dataset.architecture,
            dataset.seed,
        )
    manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "mode": "evaluation_only",
        "architecture": dataset.architecture,
        "seed": dataset.seed,
        "seed_report": seed_report,
        "vocabulary": vocabulary.to_dict(),
        "inputs": {name: str(paths[name]) for name in ("train", "validation", "test")},
        "checkpoint": str(checkpoint_path),
        "output_dir": str(paths["output_dir"]),
        "training_config": asdict(config),
        "split_counts": {split: len(records) for split, records in metadata.items()},
        "metrics": metrics,
    }
    manifest_path = paths["output_dir"] / f"evaluation_manifest_seed_{dataset.seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON training configuration")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, help="Override the seed from the configuration")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count")
    parser.add_argument("--batch-size", type=int, help="Override the configured training batch size")
    parser.add_argument("--max-sequence-length", type=int, help="Override the configured sequence length")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate --checkpoint without training")
    parser.add_argument("--checkpoint", type=Path, help="Keras checkpoint for --evaluate-only")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    dataset, config = load_run_config(_resolve_path(args.config, repo_root))
    if args.seed is not None:
        dataset = replace(dataset, seed=args.seed)
    overrides = {
        key: value for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_sequence_length": args.max_sequence_length,
        }.items() if value is not None
    }
    if overrides:
        config = replace(config, **overrides)
        config.validate()
    if args.evaluate_only:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required with --evaluate-only")
        manifest = run_evaluation(
            dataset,
            config,
            args.checkpoint,
            repo_root,
            show_progress=not args.no_progress,
        )
        manifest_path = _resolve_path(dataset.output_dir, repo_root) / f"evaluation_manifest_seed_{dataset.seed}.json"
    else:
        manifest = run_training(dataset, config, repo_root, show_progress=not args.no_progress)
        manifest_path = _resolve_path(dataset.output_dir, repo_root) / f"run_manifest_seed_{dataset.seed}.json"
    print(json.dumps({
        "manifest": str(manifest_path),
        "architecture": manifest["architecture"],
        "seed": manifest["seed"],
        "split_counts": manifest["split_counts"],
        "test_metrics": manifest["metrics"]["test"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
