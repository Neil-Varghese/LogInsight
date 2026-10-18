"""Compute Integrated Gradients for the LogRobust-style attention Bi-LSTM.

The LogRobust research detector consumes integer event IDs through a trainable
embedding, followed by a masked bidirectional LSTM and temporal attention.
Gradients are therefore integrated in embedding space. The baseline uses the
PAD embedding while preserving each input's valid-event timestep mask along
the integration path; this avoids introducing a discrete all-false mask jump
that gradients cannot explain when ``mask_zero=True``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from detector_interface import DetectorContract, KerasDetectorAdapter
    from event_vocabulary import EventVocabulary, load_vocabulary
    from integrated_gradients_lstm import (
        DEFAULT_INTEGRATION_STEPS,
        _manifest_path,
        _open_output,
        _resolve_path,
        randomize_trainable_weights,
        ranked_event_positions,
    )
    from seed_utils import set_global_seed
    from train_logrobust_attention_bilstm import (
        DatasetConfig,
        LogRobustTrainingConfig,
        TemporalAttention,
        _prepare_loaded_splits,
        build_model,
        load_run_config,
    )
except ImportError:  # pragma: no cover - supports package-style imports
    from research.scripts.detector_interface import DetectorContract, KerasDetectorAdapter
    from research.scripts.event_vocabulary import EventVocabulary, load_vocabulary
    from research.scripts.integrated_gradients_lstm import (
        DEFAULT_INTEGRATION_STEPS,
        _manifest_path,
        _open_output,
        _resolve_path,
        randomize_trainable_weights,
        ranked_event_positions,
    )
    from research.scripts.seed_utils import set_global_seed
    from research.scripts.train_logrobust_attention_bilstm import (
        DatasetConfig,
        LogRobustTrainingConfig,
        TemporalAttention,
        _prepare_loaded_splits,
        build_model,
        load_run_config,
    )


SCRIPT_SCHEMA_VERSION = "1.0"
METHOD = "integrated_gradients"
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
    for index, layer in enumerate(model.layers):
        if layer.__class__.__name__ == "Embedding" or layer.name == "event_embedding":
            return index, layer
    raise ValueError("model does not contain the LogRobust event embedding layer")


def _call_layer(layer: Any, tensor: Any, mask: Any, tf: Any) -> Any:
    """Call the attention path with its timestep mask when appropriate."""

    layer_name = getattr(layer, "name", "")
    class_name = layer.__class__.__name__
    if layer_name in {"bilstm", "temporal_attention"} or class_name in {
        "Bidirectional",
        "_TemporalAttention",
        "TemporalAttention",
    }:
        return layer(tensor, mask=mask, training=False) if layer_name == "bilstm" else layer(tensor, mask=mask)
    try:
        return layer(tensor, training=False)
    except TypeError:
        return layer(tensor)


def _forward_from_embeddings(
    model: Any,
    embedding_index: int,
    embeddings: Any,
    mask: Any,
    tf: Any,
) -> Any:
    tensor = embeddings
    for layer in model.layers[embedding_index + 1 :]:
        tensor = _call_layer(layer, tensor, mask, tf)
    return tf.reshape(tensor, [-1])


def _model_outputs(model: Any, inputs: np.ndarray, tf: Any) -> Any:
    values = model(tf.convert_to_tensor(inputs, dtype=tf.int32), training=False)
    return tf.reshape(values, [-1])


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
    """Compute masked embedding-space Integrated Gradients for one batch."""

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

    baseline_inputs = np.full_like(inputs, baseline_token_id, dtype=np.int32)
    actual_ids = tf.convert_to_tensor(inputs, dtype=tf.int32)
    mask = tf.not_equal(actual_ids, baseline_token_id)
    baseline_ids = tf.convert_to_tensor(baseline_inputs, dtype=tf.int32)
    actual_embeddings = embedding_layer(actual_ids)
    baseline_embeddings = embedding_layer(baseline_ids)
    difference = actual_embeddings - baseline_embeddings

    with tf.GradientTape() as tape:
        tape.watch(baseline_embeddings)
        baseline_suffix_output = _to_output_space(_forward_from_embeddings(
            model, embedding_index, baseline_embeddings, mask, tf
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
                _forward_from_embeddings(model, embedding_index, interpolated, mask, tf),
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
    # The input mask is held fixed along the path. Calling the full model on
    # an all-padding integer sequence would switch every timestep off and
    # introduce a discrete mask jump that gradients cannot explain.
    baseline_probability = _forward_from_embeddings(
        model, embedding_index, baseline_embeddings, mask, tf
    )
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


def _default_output(repo_root: Path, dataset: DatasetConfig, split: str) -> Path:
    return repo_root / "research/models/attributions/logrobust_attention_bilstm" / (
        f"seed_{dataset.seed}_{split}_integrated_gradients.jsonl.gz"
    )


def run_integrated_gradients(
    dataset: DatasetConfig,
    config: LogRobustTrainingConfig,
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
) -> dict[str, Any]:
    """Compute and write one attribution record per prediction unit."""

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
    vocabulary, paths, loaded = _prepare_loaded_splits(dataset, config, repo_root, show_progress)
    checkpoint_path = _resolve_path(checkpoint, repo_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    output_path = _resolve_path(output, repo_root) if output is not None else _default_output(repo_root, dataset, split)
    inputs, labels, metadata = loaded[split]
    if max_records is not None:
        inputs = inputs[:max_records]
        labels = labels[:max_records]
        metadata = metadata[:max_records]

    adapter = KerasDetectorAdapter(
        DetectorContract(dataset.architecture),
        build_model,
        custom_objects={
            "TemporalAttention": TemporalAttention,
            "_TemporalAttention": TemporalAttention,
        },
    )
    try:
        model = adapter.load_model(checkpoint_path)
        checkpoint_load_mode = "full_model"
    except (TypeError, ValueError) as error:
        # Colab may serialize a newer Keras config than the local runtime can
        # deserialize. The architecture/configuration is frozen locally, so
        # loading the weight store into a freshly built model is equivalent for
        # inference and keeps the transfer auditable.
        model = build_model(vocabulary.size, config)
        model.load_weights(checkpoint_path)
        checkpoint_load_mode = "architecture_rebuilt_weights_only"
    if randomize_model_seed is not None:
        randomize_trainable_weights(model, randomize_model_seed)
    count = 0
    max_completeness_error = 0.0
    sum_completeness_error = 0.0
    with _open_output(output_path) as handle:
        for start in range(0, len(inputs), batch_size):
            end = min(start + batch_size, len(inputs))
            result = integrated_gradients_batch(
                model,
                inputs[start:end],
                integration_steps=integration_steps,
                baseline_token_id=vocabulary.pad_id,
                output_space=output_space,
            )
            for offset, record_metadata in enumerate(metadata[start:end]):
                error = float(result["completeness_error"][offset])
                events = ranked_event_positions(
                    inputs[start + offset],
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
                    "padded_length": int(inputs.shape[1]),
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
                        "mask_policy": "preserve_input_event_mask_during_integration",
                    },
                    "events": events,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
                sum_completeness_error += error
                max_completeness_error = max(max_completeness_error, error)
            if show_progress:
                print(
                    f"\rIntegrated Gradients: {end:,}/{len(inputs):,}",
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
        "checkpoint_load_mode": checkpoint_load_mode,
        "vocabulary": vocabulary.to_dict(),
        "vocabulary_fit_scope": "training_only",
        "input_path": str(paths[split]),
        "output_path": str(output_path),
        "integration_steps": integration_steps,
        "batch_size": batch_size,
        "padding": config.padding,
        "mask_zero": config.mask_zero,
        "baseline": {
            "type": "all_padding",
            "event_id": vocabulary.pad_id,
            "event_token": vocabulary.pad_token,
            "mask_policy": "preserve_input_event_mask_during_integration",
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
    parser.add_argument("--config", type=Path, required=True, help="LogRobust training configuration")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained LogRobust checkpoint")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path, help="Output JSONL(.gz) attribution file")
    parser.add_argument("--integration-steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--output-space", choices=("probability", "logit"), default="probability")
    parser.add_argument("--max-records", type=int)
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
        dataset = replace(dataset, seed=args.seed)
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
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
