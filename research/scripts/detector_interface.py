"""Shared detector contract for the LogInsight-FV model implementations.

Both detector trainers use this module for the detector-facing batch contract,
probability prediction, binary evaluation, and prediction-file serialization.
Model construction remains architecture-specific; everything downstream of a
model's probability output is intentionally shared.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np


INTERFACE_SCHEMA_VERSION = "1.0"
EPSILON = 1e-7


@dataclass(frozen=True)
class DetectorContract:
    """Stable contract shared by all binary sequence detectors."""

    detector_id: str
    prediction_unit_field: str = "unit_id"
    probability_field: str = "probability"
    label_field: str = "label_id"
    output_schema_version: str = INTERFACE_SCHEMA_VERSION

    def validate(self) -> None:
        if not self.detector_id or any(char.isspace() for char in self.detector_id):
            raise ValueError("detector_id must be a non-empty whitespace-free string")
        for name in ("prediction_unit_field", "probability_field", "label_field"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True)
class DetectorBatch:
    """The architecture-neutral input passed to a detector."""

    split: str
    inputs: np.ndarray
    labels: np.ndarray
    metadata: tuple[Mapping[str, Any], ...]

    def validate(self) -> None:
        if not self.split:
            raise ValueError("split must be non-empty")
        if self.inputs.ndim != 2:
            raise ValueError("inputs must be a two-dimensional integer array")
        if not np.issubdtype(self.inputs.dtype, np.integer):
            raise ValueError("inputs must have an integer dtype")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.inputs.shape[0]:
            raise ValueError("labels must be one-dimensional and match inputs")
        if np.any((self.labels < 0) | (self.labels > 1)):
            raise ValueError("labels must be binary")
        if len(self.metadata) != self.inputs.shape[0]:
            raise ValueError("metadata must have one record per input")
        unit_ids = [record.get("unit_id") for record in self.metadata]
        if any(value in (None, "") for value in unit_ids):
            raise ValueError("every metadata record must contain unit_id")
        if len(set(map(str, unit_ids))) != len(unit_ids):
            raise ValueError("metadata contains duplicate unit_id values")


class DetectorAdapter(Protocol):
    """Runtime interface for architecture-specific Keras adapters."""

    contract: DetectorContract

    def build_model(self, vocabulary_size: int, config: Any) -> Any:
        ...

    def load_model(self, checkpoint: Path) -> Any:
        ...

    def predict(self, model: Any, inputs: np.ndarray, batch_size: int) -> np.ndarray:
        ...


def _validate_inference_inputs(inputs: np.ndarray, batch_size: int) -> np.ndarray:
    inputs = np.asarray(inputs)
    if inputs.ndim != 2:
        raise ValueError("detector inputs must be a two-dimensional array")
    if not np.issubdtype(inputs.dtype, np.integer):
        raise ValueError("detector inputs must have an integer dtype")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return inputs


def predict_probabilities(
    model: Any,
    inputs: np.ndarray,
    batch_size: int,
    *,
    verbose: int = 1,
) -> np.ndarray:
    """Run the common probability-output contract for a Keras-like model."""

    inputs = _validate_inference_inputs(inputs, batch_size)
    probabilities = np.asarray(model.predict(inputs, batch_size=batch_size, verbose=verbose)).reshape(-1)
    if probabilities.shape[0] != inputs.shape[0]:
        raise ValueError("model returned a different number of predictions than inputs")
    probabilities = probabilities.astype(np.float64)
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("model probabilities must be finite values in [0, 1]")
    return probabilities


def predict_probabilities_batched(
    model: Any,
    inputs: np.ndarray,
    batch_size: int,
    *,
    verbose: int = 0,
) -> np.ndarray:
    """Predict in bounded chunks while preserving input order."""

    inputs = _validate_inference_inputs(inputs, batch_size)
    if inputs.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    chunks = [
        predict_probabilities(
            model,
            inputs[start : start + batch_size],
            min(batch_size, inputs.shape[0] - start),
            verbose=verbose,
        )
        for start in range(0, inputs.shape[0], batch_size)
    ]
    return np.concatenate(chunks)


def _inference_row_key(detector_id: str, model_key: str, row: np.ndarray) -> str:
    """Create a stable cache key for one detector input row."""

    row = np.ascontiguousarray(row)
    digest = hashlib.sha256()
    digest.update(detector_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(model_key.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(row.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(tuple(row.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(row.tobytes(order="C"))
    return digest.hexdigest()


@dataclass
class InferenceCacheStats:
    """Counters describing cache reuse for a detector inference session."""

    requested_rows: int = 0
    cache_hit_rows: int = 0
    deduplicated_rows: int = 0
    model_rows: int = 0


class DetectorInferenceCache:
    """Cache and batch detector probabilities for repeated perturbation inputs.

    The cache is namespaced by detector ID and model key. Callers should use a
    checkpoint hash, run-manifest path, or another stable model identity as
    ``model_key`` so predictions from different seeds or checkpoints cannot
    collide. The cache stores probabilities only; labels, thresholds, and
    explanation metadata remain outside this inference layer.
    """

    def __init__(self, detector_id: str, model_key: str) -> None:
        if not detector_id or not model_key:
            raise ValueError("detector_id and model_key must be non-empty")
        self.detector_id = detector_id
        self.model_key = model_key
        self._probabilities: dict[str, float] = {}
        self.stats = InferenceCacheStats()

    def __len__(self) -> int:
        return len(self._probabilities)

    def clear(self) -> None:
        self._probabilities.clear()
        self.stats = InferenceCacheStats()

    def predict(self, model: Any, inputs: np.ndarray, batch_size: int) -> np.ndarray:
        """Return probabilities, evaluating only unique cache misses."""

        inputs = _validate_inference_inputs(inputs, batch_size)
        self.stats.requested_rows += int(inputs.shape[0])
        if inputs.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        keys = [_inference_row_key(self.detector_id, self.model_key, row) for row in inputs]
        output = np.full(inputs.shape[0], np.nan, dtype=np.float64)
        pending_indices: dict[str, int] = {}
        for index, key in enumerate(keys):
            if key in self._probabilities:
                output[index] = self._probabilities[key]
                self.stats.cache_hit_rows += 1
            elif key in pending_indices:
                self.stats.deduplicated_rows += 1
            else:
                pending_indices[key] = index

        if pending_indices:
            unique_indices = list(pending_indices.values())
            unique_inputs = inputs[unique_indices]
            probabilities = predict_probabilities_batched(
                model,
                unique_inputs,
                batch_size,
                verbose=0,
            )
            self.stats.model_rows += int(probabilities.shape[0])
            for key, probability in zip(pending_indices, probabilities):
                self._probabilities[key] = float(probability)

        for index, key in enumerate(keys):
            if np.isnan(output[index]):
                output[index] = self._probabilities[key]
        return output


class KerasDetectorAdapter:
    """Adapter that standardizes Keras build/load/predict operations."""

    def __init__(
        self,
        contract: DetectorContract,
        builder: Callable[[int, Any], Any],
        custom_objects: Mapping[str, Any] | None = None,
    ) -> None:
        contract.validate()
        self.contract = contract
        self._builder = builder
        self._custom_objects = dict(custom_objects or {})

    def build_model(self, vocabulary_size: int, config: Any) -> Any:
        return self._builder(vocabulary_size, config)

    def load_model(self, checkpoint: Path) -> Any:
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("TensorFlow is required for checkpoint evaluation") from exc
        return tf.keras.models.load_model(checkpoint, custom_objects=self._custom_objects)

    def predict(self, model: Any, inputs: np.ndarray, batch_size: int) -> np.ndarray:
        return predict_probabilities(model, inputs, batch_size)


def binary_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute the shared fixed-threshold binary metrics."""

    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be one-dimensional arrays with equal length")
    if labels.size == 0 or np.any((labels < 0) | (labels > 1)):
        raise ValueError("labels must be a non-empty binary array")
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
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
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    loss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))
    return {
        "count": int(labels.size),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "threshold": threshold,
        "loss": loss,
        "accuracy": float(np.mean(predictions == labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def make_detector_batch(
    split: str,
    inputs: np.ndarray,
    labels: np.ndarray,
    metadata: Iterable[Mapping[str, Any]],
) -> DetectorBatch:
    batch = DetectorBatch(split, np.asarray(inputs), np.asarray(labels), tuple(metadata))
    batch.validate()
    return batch


def prediction_records(
    contract: DetectorContract,
    batch: DetectorBatch,
    probabilities: np.ndarray,
    threshold: float,
    seed: int,
    dataset: str = "HDFS",
) -> list[dict[str, Any]]:
    """Create the shared prediction schema used by calibration/thresholding."""

    contract.validate()
    batch.validate()
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.shape[0] != batch.inputs.shape[0]:
        raise ValueError("probabilities must be one-dimensional and match the detector batch")
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    output: list[dict[str, Any]] = []
    for record, probability in zip(batch.metadata, probabilities):
        label_id = int(record.get("label_id", batch.labels[len(output)]))
        output.append({
            "schema_version": INTERFACE_SCHEMA_VERSION,
            "dataset": dataset,
            "architecture": contract.detector_id,
            "seed": int(seed),
            "split": batch.split,
            "unit_id": str(record[contract.prediction_unit_field]),
            "label": "Anomaly" if label_id else "Normal",
            "label_id": label_id,
            "sequence_length": int(record.get("sequence_length", 0)),
            "probability": float(probability),
            "confidence": float(max(probability, 1.0 - probability)),
            "prediction": int(probability >= threshold),
            "threshold": float(threshold),
        })
    return output


def write_prediction_records(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
            count += 1
    return count
