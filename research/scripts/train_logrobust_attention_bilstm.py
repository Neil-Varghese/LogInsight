"""Train and evaluate the LogRobust-style attention Bi-LSTM.

This is the second detector for LogInsight-FV.  It follows the published
LogRobust detector shape: event representations are encoded by a
bidirectional LSTM, an additive temporal-attention layer pools the sequence,
and a sigmoid classifier predicts the block/window label.

The research protocol intentionally reuses the project's training-only event
vocabulary and fixed prediction-unit splits.  The published method uses
external semantic embeddings; this implementation uses a trainable embedding
over the frozen event vocabulary so both detectors receive the same leakage-
controlled inputs.  That protocol deviation is recorded for Step 27 rather
than hidden in the implementation.

Training is an explicit command-line operation.  TensorFlow is imported only
when a model is built or loaded, so data and configuration tests stay light.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from detector_interface import DetectorContract, KerasDetectorAdapter
    from event_vocabulary import load_vocabulary
    from seed_utils import set_global_seed
    from train_loginsight_lstm import (
        DatasetConfig,
        TrainingConfig,
        _predict,
        _resolve_path,
        assert_disjoint_splits,
        binary_metrics,
        class_weights,
        load_split,
        write_predictions,
    )
except ImportError:  # pragma: no cover - supports package-style imports
    from research.scripts.detector_interface import DetectorContract, KerasDetectorAdapter
    from research.scripts.event_vocabulary import load_vocabulary
    from research.scripts.seed_utils import set_global_seed
    from research.scripts.train_loginsight_lstm import (
        DatasetConfig,
        TrainingConfig,
        _predict,
        _resolve_path,
        assert_disjoint_splits,
        binary_metrics,
        class_weights,
        load_split,
        write_predictions,
    )


SCRIPT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LogRobustTrainingConfig:
    """All choices that affect a LogRobust-style detector run."""

    max_sequence_length: int = 50
    padding: str = "pre"
    truncating: str = "pre"
    embedding_dim: int = 128
    bilstm_units: int = 128
    attention_units: int = 64
    dense_units: int = 32
    bilstm_dropout: float = 0.3
    dense_dropout: float = 0.2
    mask_zero: bool = True
    learning_rate: float = 0.001
    batch_size: int = 128
    prediction_batch_size: int = 256
    epochs: int = 20
    early_stopping_patience: int = 3
    reduce_lr_patience: int = 2
    threshold: float = 0.5
    deterministic: bool = True
    use_class_weights: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LogRobustTrainingConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown LogRobust training fields: {', '.join(unknown)}")
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
            "bilstm_units",
            "attention_units",
            "dense_units",
            "batch_size",
            "prediction_batch_size",
            "epochs",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("bilstm_dropout", "dense_dropout"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.early_stopping_patience < 0 or self.reduce_lr_patience < 0:
            raise ValueError("callback patience values must be non-negative")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_run_config(path: Path) -> tuple[DatasetConfig, LogRobustTrainingConfig]:
    payload = _load_json(path)
    try:
        data = payload["data"]
    except KeyError as exc:
        raise ValueError("Training configuration is missing 'data'") from exc
    training = payload.get("training", {})
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
        architecture=str(data.get("architecture", "logrobust_attention_bilstm")),
    )
    return dataset, LogRobustTrainingConfig.from_dict(training)


def build_model(vocabulary_size: int, config: LogRobustTrainingConfig):
    """Build the attention Bi-LSTM and expose its attention layer by name."""

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("TensorFlow is required for model training/evaluation") from exc
    config.validate()

    inputs = layers.Input(
        shape=(config.max_sequence_length,),
        dtype="int32",
        name="event_ids",
    )
    embeddings = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=config.embedding_dim,
        mask_zero=config.mask_zero,
        name="event_embedding",
    )(inputs)
    encoded = layers.Bidirectional(
        layers.LSTM(
            config.bilstm_units,
            return_sequences=True,
            dropout=config.bilstm_dropout,
            name="bilstm_cell",
        ),
        name="bilstm",
    )(embeddings)
    pooled = TemporalAttention(
        attention_units=config.attention_units,
        name="temporal_attention",
    )(encoded)
    hidden = layers.Dense(config.dense_units, activation="relu", name="dense_relu")(pooled)
    hidden = layers.Dropout(config.dense_dropout, name="dense_dropout")(hidden)
    outputs = layers.Dense(1, activation="sigmoid", name="anomaly_probability")(hidden)
    model = models.Model(inputs=inputs, outputs=outputs, name="logrobust_attention_bilstm")
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


def _prepare_loaded_splits(dataset: DatasetConfig, config: LogRobustTrainingConfig, repo_root: Path, show_progress: bool):
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
    # The shared loader only needs these four fields from the LSTM config.
    loader_config = TrainingConfig(
        max_sequence_length=config.max_sequence_length,
        padding=config.padding,
        truncating=config.truncating,
    )
    for split in ("train", "validation", "test"):
        loaded[split] = load_split(paths[split], vocabulary, loader_config, show_progress=show_progress)
    assert_disjoint_splits({split: value[2] for split, value in loaded.items()})
    return vocabulary, paths, loaded


def _callbacks(tf: Any, config: LogRobustTrainingConfig, checkpoint_path: Path):
    return [
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


def run_training(
    dataset: DatasetConfig,
    config: LogRobustTrainingConfig,
    repo_root: Path,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train once, save artifacts, and return the run manifest."""

    config.validate()
    seed_report = set_global_seed(dataset.seed, deterministic=config.deterministic)
    vocabulary, paths, loaded = _prepare_loaded_splits(dataset, config, repo_root, show_progress)
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("TensorFlow is required for model training/evaluation") from exc
    adapter = KerasDetectorAdapter(DetectorContract(dataset.architecture), build_model)
    model = adapter.build_model(vocabulary.size, config)
    checkpoint_path = paths["output_dir"] / f"best_seed_{dataset.seed}.keras"
    final_model_path = paths["output_dir"] / f"final_seed_{dataset.seed}.keras"
    fit_kwargs: dict[str, Any] = {
        "x": loaded["train"][0],
        "y": loaded["train"][1],
        "validation_data": (loaded["validation"][0], loaded["validation"][1]),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "callbacks": _callbacks(tf, config, checkpoint_path),
        "verbose": 1,
    }
    if config.use_class_weights:
        fit_kwargs["class_weight"] = class_weights(loaded["train"][1])
    history = model.fit(**fit_kwargs)
    model.save(final_model_path)

    metrics: dict[str, Any] = {}
    metadata = {split: value[2] for split, value in loaded.items()}
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
        "architecture_source": "Zhang et al., FSE 2019 LogRobust-style attention Bi-LSTM",
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
    config: LogRobustTrainingConfig,
    checkpoint: Path,
    repo_root: Path,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Evaluate an existing attention Bi-LSTM checkpoint."""

    config.validate()
    seed_report = set_global_seed(dataset.seed, deterministic=config.deterministic)
    vocabulary, paths, loaded = _prepare_loaded_splits(dataset, config, repo_root, show_progress)
    checkpoint_path = _resolve_path(checkpoint, repo_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    adapter = KerasDetectorAdapter(
        DetectorContract(dataset.architecture),
        build_model,
        custom_objects={
            "TemporalAttention": TemporalAttention,
            "_TemporalAttention": TemporalAttention,
        },
    )
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
        "architecture_source": "Zhang et al., FSE 2019 LogRobust-style attention Bi-LSTM",
        "seed": dataset.seed,
        "seed_report": seed_report,
        "vocabulary": vocabulary.to_dict(),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(paths["output_dir"]),
        "training_config": asdict(config),
        "split_counts": {split: len(value[2]) for split, value in loaded.items()},
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
        manifest = run_evaluation(dataset, config, args.checkpoint, repo_root, show_progress=not args.no_progress)
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


class TemporalAttention:  # replaced with a TensorFlow layer on demand below
    """Placeholder overwritten by :func:`_make_temporal_attention` imports."""


def _make_temporal_attention_base():
    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover - import-only environments
        return None

    class _TemporalAttention(tf.keras.layers.Layer):
        def __init__(self, attention_units: int = 64, **kwargs: Any):
            super().__init__(**kwargs)
            self.attention_units = int(attention_units)
            self.supports_masking = True

        def build(self, input_shape):
            feature_dim = int(input_shape[-1])
            self.projection = self.add_weight(
                name="projection",
                shape=(feature_dim, self.attention_units),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.bias = self.add_weight(
                name="bias",
                shape=(self.attention_units,),
                initializer="zeros",
                trainable=True,
            )
            self.score = self.add_weight(
                name="score",
                shape=(self.attention_units, 1),
                initializer="glorot_uniform",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs, mask=None):
            hidden = tf.tanh(tf.tensordot(inputs, self.projection, axes=1) + self.bias)
            scores = tf.squeeze(tf.tensordot(hidden, self.score, axes=1), axis=-1)
            if mask is not None:
                mask = tf.cast(mask, scores.dtype)
                scores = scores + (1.0 - mask) * tf.cast(-1e9, scores.dtype)
            weights = tf.nn.softmax(scores, axis=1)
            return tf.reduce_sum(inputs * tf.expand_dims(weights, axis=-1), axis=1)

        def get_config(self):
            return {**super().get_config(), "attention_units": self.attention_units}

        def compute_mask(self, inputs, mask=None):
            # Attention consumes the timestep mask and returns one vector.
            return None

    return _TemporalAttention


_attention_class = _make_temporal_attention_base()
if _attention_class is not None:
    TemporalAttention = _attention_class


if __name__ == "__main__":
    main()
