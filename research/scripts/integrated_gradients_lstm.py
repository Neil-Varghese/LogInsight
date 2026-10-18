"""Compute Integrated Gradients for the research LogInsight LSTM.

The detector consumes integer event IDs, so gradients are computed in the
model's embedding space rather than with respect to the integer IDs. The
default baseline is an all-padding sequence. Attributions are then summed
over embedding dimensions to produce one signed score per retained event
position. Padding positions are retained in the output for alignment but are
excluded from event ranking.

This script records completeness diagnostics for every prediction unit. It is
intended to produce the prompt-visible attribution artifact for Step 31;
occlusion and leave-one-event-out diagnostics are separate verification
artifacts produced by later research scripts.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from event_vocabulary import EventVocabulary, load_vocabulary
    from seed_utils import set_global_seed
    from train_loginsight_lstm import (
        DatasetConfig,
        TrainingConfig,
        _resolve_path,
        load_run_config,
        load_split,
    )
    from detector_interface import DetectorContract, KerasDetectorAdapter
    from train_loginsight_lstm import build_model
except ImportError:  # pragma: no cover - supports package-style imports
    from research.scripts.event_vocabulary import EventVocabulary, load_vocabulary
    from research.scripts.seed_utils import set_global_seed
    from research.scripts.train_loginsight_lstm import (
        DatasetConfig,
        TrainingConfig,
        _resolve_path,
        load_run_config,
        load_split,
        build_model,
    )
    from research.scripts.detector_interface import DetectorContract, KerasDetectorAdapter


SCRIPT_SCHEMA_VERSION = "1.0"
METHOD = "integrated_gradients"
DEFAULT_INTEGRATION_STEPS = 64
DEFAULT_BATCH_SIZE = 32


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("TensorFlow is required for Integrated Gradients") from exc
    return tf


def _validate_inputs(inputs: np.ndarray, integration_steps: int, baseline_token_id: int) -> np.ndarray:
    inputs = np.asarray(inputs)
    if inputs.ndim != 2:
        raise ValueError("inputs must be a two-dimensional array")
    if not np.issubdtype(inputs.dtype, np.integer):
        raise ValueError("inputs must have an integer dtype")
    if inputs.shape[0] == 0 or inputs.shape[1] == 0:
        raise ValueError("inputs must contain at least one sequence and one position")
    if np.any(inputs < 0):
        raise ValueError("inputs must contain non-negative token IDs")
    if integration_steps < 1:
        raise ValueError("integration_steps must be positive")
    if baseline_token_id < 0:
        raise ValueError("baseline_token_id must be non-negative")
    return inputs.astype(np.int32, copy=False)


def _find_embedding_layer(model: Any) -> tuple[int, Any]:
    """Return the first Keras Embedding layer and its model-layer index."""

    for index, layer in enumerate(model.layers):
        if layer.__class__.__name__ == "Embedding":
            return index, layer
    raise ValueError("model does not contain a Keras Embedding layer")


def _call_layer(layer: Any, tensor: Any, tf: Any) -> Any:
    """Call a Keras layer in inference mode when it accepts ``training``."""

    try:
        return layer(tensor, training=False)
    except TypeError:
        return layer(tensor)


def _forward_from_embeddings(model: Any, embedding_index: int, embeddings: Any, tf: Any) -> Any:
    """Run the model suffix after the embedding layer."""

    tensor = embeddings
    for layer in model.layers[embedding_index + 1 :]:
        tensor = _call_layer(layer, tensor, tf)
    return tf.reshape(tensor, [-1])


def _model_outputs(model: Any, inputs: np.ndarray, tf: Any) -> Any:
    values = model(tf.convert_to_tensor(inputs, dtype=tf.int32), training=False)
    return tf.reshape(values, [-1])


def randomize_trainable_weights(model: Any, seed: int) -> None:
    """Replace trainable weights deterministically for the sanity check."""

    rng = np.random.default_rng(seed)
    for variable in model.trainable_variables:
        shape = tuple(int(size) for size in variable.shape)
        scale = 0.05 / max(math.sqrt(shape[0]) if shape else 1.0, 1.0)
        values = rng.normal(0.0, scale, size=shape).astype(variable.numpy().dtype)
        variable.assign(values)


def _to_output_space(values: Any, output_space: str, tf: Any) -> Any:
    if output_space == "probability":
        return values
    if output_space == "logit":
        epsilon = tf.cast(tf.keras.backend.epsilon(), values.dtype)
        clipped = tf.clip_by_value(values, epsilon, 1.0 - epsilon)
        return tf.math.log(clipped) - tf.math.log1p(-clipped)
    raise ValueError("output_space must be 'probability' or 'logit'")


def integrated_gradients_batch(
    model: Any,
    inputs: np.ndarray,
    integration_steps: int = DEFAULT_INTEGRATION_STEPS,
    baseline_token_id: int = 0,
    output_space: str = "probability",
) -> dict[str, np.ndarray]:
    """Compute embedding-space Integrated Gradients for a batch.

    Returns actual and baseline anomaly probabilities, one attribution per
    padded input position, and the completeness error. The approximation uses
    the trapezoidal rule over ``integration_steps`` intervals.
    """

    inputs = _validate_inputs(inputs, integration_steps, baseline_token_id)
    if output_space not in {"probability", "logit"}:
        raise ValueError("output_space must be 'probability' or 'logit'")
    tf = _require_tensorflow()
    embedding_index, embedding_layer = _find_embedding_layer(model)
    input_dim = int(getattr(embedding_layer, "input_dim"))
    if baseline_token_id >= input_dim:
        raise ValueError("baseline_token_id is outside the embedding vocabulary")
    if np.any(inputs >= input_dim):
        raise ValueError("inputs contain a token ID outside the embedding vocabulary")

    actual_ids = tf.convert_to_tensor(inputs, dtype=tf.int32)
    baseline_inputs = np.full_like(inputs, baseline_token_id, dtype=np.int32)
    baseline_ids = tf.convert_to_tensor(baseline_inputs, dtype=tf.int32)
    actual_embeddings = embedding_layer(actual_ids)
    baseline_embeddings = embedding_layer(baseline_ids)
    difference = actual_embeddings - baseline_embeddings

    with tf.GradientTape() as tape:
        tape.watch(baseline_embeddings)
        baseline_suffix_output = _to_output_space(_forward_from_embeddings(
            model, embedding_index, baseline_embeddings, tf
        ), output_space, tf)
    baseline_gradients = tape.gradient(baseline_suffix_output, baseline_embeddings)
    if baseline_gradients is None:
        raise RuntimeError("Could not compute gradients at the Integrated Gradients baseline")

    gradient_sum = tf.zeros_like(actual_embeddings)
    for step in range(integration_steps + 1):
        alpha = tf.cast(step / integration_steps, actual_embeddings.dtype)
        interpolated = baseline_embeddings + alpha * difference
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            output = _to_output_space(
                _forward_from_embeddings(model, embedding_index, interpolated, tf),
                output_space,
                tf,
            )
        gradients = tape.gradient(output, interpolated)
        if gradients is None:
            raise RuntimeError("Could not compute Integrated Gradients gradients")
        weight = 0.5 if step in (0, integration_steps) else 1.0
        gradient_sum = gradient_sum + tf.cast(weight, gradients.dtype) * gradients

    average_gradients = gradient_sum / float(integration_steps)
    attributions = tf.reduce_sum(difference * average_gradients, axis=-1)
    actual_probability = _model_outputs(model, inputs, tf)
    baseline_probability = _model_outputs(model, baseline_inputs, tf)
    actual_output = _to_output_space(actual_probability, output_space, tf)
    baseline_output = _to_output_space(baseline_probability, output_space, tf)
    attribution_sum = tf.reduce_sum(attributions, axis=-1)
    output_delta = actual_output - baseline_output
    completeness_error = tf.abs(attribution_sum - output_delta)

    return {
        "probability": np.asarray(actual_probability.numpy(), dtype=np.float64),
        "baseline_probability": np.asarray(baseline_probability.numpy(), dtype=np.float64),
        "attribution": np.asarray(attributions.numpy(), dtype=np.float64),
        "attribution_sum": np.asarray(attribution_sum.numpy(), dtype=np.float64),
        "output_delta": np.asarray(output_delta.numpy(), dtype=np.float64),
        "completeness_error": np.asarray(completeness_error.numpy(), dtype=np.float64),
        "output_space": output_space,
    }


def ranked_event_positions(
    input_row: np.ndarray,
    attribution_row: np.ndarray,
    vocabulary: EventVocabulary,
    sequence_length: int,
    padding: str = "pre",
) -> list[dict[str, Any]]:
    """Serialize aligned event positions and rank non-padding events."""

    if input_row.ndim != 1 or attribution_row.ndim != 1 or input_row.shape != attribution_row.shape:
        raise ValueError("input_row and attribution_row must be equal-length vectors")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if padding not in {"pre", "post"}:
        raise ValueError("padding must be 'pre' or 'post'")
    max_length = input_row.shape[0]
    retained_length = min(sequence_length, max_length)
    padding_count = max_length - retained_length if padding == "pre" else 0
    retained_start = max(sequence_length - max_length, 0)
    active_positions = [
        index for index, token_id in enumerate(input_row) if int(token_id) != vocabulary.pad_id
    ]
    ranked = sorted(active_positions, key=lambda index: (-abs(float(attribution_row[index])), index))
    rank_by_position = {position: rank for rank, position in enumerate(ranked, start=1)}
    events: list[dict[str, Any]] = []
    for position, token_id in enumerate(input_row):
        token_id = int(token_id)
        is_padding = token_id == vocabulary.pad_id
        event = {
            "input_position": position,
            "original_sequence_index": (
                None
                if is_padding
                else retained_start + position - padding_count
            ),
            "event_id": token_id,
            "event_token": vocabulary.decode([token_id])[0],
            "is_padding": is_padding,
            "attribution": float(attribution_row[position]),
            "absolute_attribution": abs(float(attribution_row[position])),
            "rank": None if is_padding else rank_by_position[position],
        }
        events.append(event)
    return events


def _manifest_path(output_path: Path) -> Path:
    name = output_path.name
    if name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")] + ".manifest.json"
    else:
        name = output_path.stem + ".manifest.json"
    return output_path.with_name(name)


def _open_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("wt", encoding="utf-8")


def _default_output(repo_root: Path, dataset: DatasetConfig, split: str) -> Path:
    return _resolve_path(
        Path("research/models/attributions/loginsight_lstm")
        / f"seed_{dataset.seed}_{split}_integrated_gradients.jsonl.gz",
        repo_root,
    )


def run_integrated_gradients(
    dataset: DatasetConfig,
    config: TrainingConfig,
    checkpoint: Path,
    repo_root: Path,
    split: str = "test",
    output: Path | None = None,
    integration_steps: int = DEFAULT_INTEGRATION_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_space: str = "probability",
    max_records: int | None = None,
    randomize_model_seed: int | None = None,
    show_progress: bool = True,
    unit_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Compute and write one attribution record per split prediction unit."""

    config.validate()
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if integration_steps < 1:
        raise ValueError("integration_steps must be positive")
    if output_space not in {"probability", "logit"}:
        raise ValueError("output_space must be 'probability' or 'logit'")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive when supplied")
    seed_report = set_global_seed(dataset.seed, deterministic=config.deterministic)
    vocabulary = load_vocabulary(_resolve_path(dataset.vocabulary, repo_root))
    paths = {
        name: _resolve_path(getattr(dataset, name), repo_root)
        for name in ("train", "validation", "test")
    }
    split_inputs, split_labels, metadata = load_split(
        paths[split], vocabulary, config, show_progress=show_progress
    )
    if unit_ids is not None:
        selected_indices = [index for index, item in enumerate(metadata) if str(item["unit_id"]) in unit_ids]
        found_ids = {str(metadata[index]["unit_id"]) for index in selected_indices}
        missing_ids = sorted(unit_ids - found_ids)
        if missing_ids:
            raise ValueError(f"requested unit IDs are missing from {split}: {missing_ids[:10]}")
        split_inputs = split_inputs[selected_indices]
        split_labels = split_labels[selected_indices]
        metadata = [metadata[index] for index in selected_indices]
    if max_records is not None:
        split_inputs = split_inputs[:max_records]
        split_labels = split_labels[:max_records]
        metadata = metadata[:max_records]
    checkpoint_path = _resolve_path(checkpoint, repo_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    output_path = output or _default_output(repo_root, dataset, split)
    output_path = _resolve_path(output_path, repo_root)

    adapter = KerasDetectorAdapter(DetectorContract(dataset.architecture), build_model)
    model = adapter.load_model(checkpoint_path)
    if randomize_model_seed is not None:
        randomize_trainable_weights(model, randomize_model_seed)
    count = 0
    max_completeness_error = 0.0
    sum_completeness_error = 0.0
    with _open_output(output_path) as handle:
        for start in range(0, len(split_inputs), batch_size):
            end = min(start + batch_size, len(split_inputs))
            result = integrated_gradients_batch(
                model,
                split_inputs[start:end],
                integration_steps=integration_steps,
                baseline_token_id=vocabulary.pad_id,
                output_space=output_space,
            )
            for offset, record_metadata in enumerate(metadata[start:end]):
                error = float(result["completeness_error"][offset])
                events = ranked_event_positions(
                    split_inputs[start + offset],
                    result["attribution"][offset],
                    vocabulary,
                    int(record_metadata["sequence_length"]),
                    padding=config.padding,
                )
                record = {
                    "schema_version": SCRIPT_SCHEMA_VERSION,
                    "method": METHOD,
                    "target": "anomaly_probability" if output_space == "probability" else "anomaly_logit",
                    "output_space": output_space,
                    "architecture": dataset.architecture,
                    "seed": dataset.seed,
                    "split": split,
                    "unit_id": record_metadata["unit_id"],
                    "label_id": int(record_metadata["label_id"]),
                    "sequence_length": int(record_metadata["sequence_length"]),
                    "padded_length": int(split_inputs.shape[1]),
                    "probability": float(result["probability"][offset]),
                    "baseline_probability": float(result["baseline_probability"][offset]),
                    "output_delta": float(result["output_delta"][offset]),
                    "attribution_sum": float(result["attribution_sum"][offset]),
                    "completeness_error": error,
                    "integration_steps": integration_steps,
                    "baseline": {
                        "type": "all_padding",
                        "event_id": vocabulary.pad_id,
                        "event_token": vocabulary.pad_token,
                    },
                    "events": events,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
                sum_completeness_error += error
                max_completeness_error = max(max_completeness_error, error)
            if show_progress:
                print(
                    f"\rIntegrated Gradients: {end:,}/{len(split_inputs):,}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
        if show_progress:
            print(file=sys.stderr)

    manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "method": METHOD,
        "target": "anomaly_probability" if output_space == "probability" else "anomaly_logit",
        "output_space": output_space,
        "max_records": max_records,
        "randomize_model_seed": randomize_model_seed,
        "architecture": dataset.architecture,
        "seed": dataset.seed,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "vocabulary": vocabulary.to_dict(),
        "vocabulary_fit_scope": "training_only",
        "input_path": str(paths[split]),
        "output_path": str(output_path),
        "integration_steps": integration_steps,
        "batch_size": batch_size,
        "baseline": {
            "type": "all_padding",
            "event_id": vocabulary.pad_id,
            "event_token": vocabulary.pad_token,
        },
        "count": count,
        "mean_completeness_error": sum_completeness_error / max(count, 1),
        "max_completeness_error": max_completeness_error,
        "seed_report": seed_report,
    }
    manifest_path = _manifest_path(output_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="LSTM training configuration")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained LSTM checkpoint")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path, help="Output JSONL(.gz) attribution file")
    parser.add_argument("--integration-steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--output-space", choices=("probability", "logit"), default="probability")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--unit-list", type=Path, help="Optional JSONL(.gz) artifact containing unit_id values to process")
    parser.add_argument("--randomize-model-seed", type=int)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, help="Override the seed from the configuration")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    dataset, config = load_run_config(_resolve_path(args.config, repo_root))
    if args.seed is not None:
        dataset = DatasetConfig(
            train=dataset.train,
            validation=dataset.validation,
            test=dataset.test,
            vocabulary=dataset.vocabulary,
            output_dir=dataset.output_dir,
            seed=args.seed,
            architecture=dataset.architecture,
        )
    unit_ids = None
    if args.unit_list:
        opener = gzip.open if args.unit_list.suffix == ".gz" else open
        unit_ids = set()
        with opener(args.unit_list, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not record.get("unit_id"):
                    raise ValueError(f"{args.unit_list}:{line_number} must contain a non-empty unit_id")
                if record.get("architecture") not in (None, dataset.architecture):
                    continue
                unit_ids.add(str(record["unit_id"]))
    manifest = run_integrated_gradients(
        dataset,
        config,
        args.checkpoint,
        repo_root,
        split=args.split,
        output=args.output,
        integration_steps=args.integration_steps,
        batch_size=args.batch_size,
        output_space=args.output_space,
        max_records=args.max_records,
        randomize_model_seed=args.randomize_model_seed,
        show_progress=not args.no_progress,
        unit_ids=unit_ids,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
