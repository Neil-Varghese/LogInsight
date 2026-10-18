"""Compute verification-only single-event occlusion scores.

For each retained event position, this script removes that event, re-pads the
sequence using the detector's configured padding policy, and recomputes the
anomaly probability. The occlusion score is ``p(original) - p(occluded)``.
Occlusion values are deliberately written to a separate verification artifact
and are never part of the Integrated Gradients prompt-visible schema.

Both research detectors are supported. Repeated and duplicate perturbation
inputs use the Step 30 inference cache so later verification workloads do not
re-evaluate identical sequences unnecessarily.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from detector_interface import DetectorContract, DetectorInferenceCache, KerasDetectorAdapter
    from event_vocabulary import load_vocabulary
    from perturbation_interface import apply_perturbation
    from train_loginsight_lstm import _resolve_path
except ImportError:  # pragma: no cover - supports package-style imports
    from research.scripts.detector_interface import DetectorContract, DetectorInferenceCache, KerasDetectorAdapter
    from research.scripts.event_vocabulary import load_vocabulary
    from research.scripts.perturbation_interface import apply_perturbation
    from research.scripts.train_loginsight_lstm import _resolve_path


SCRIPT_SCHEMA_VERSION = "1.0"
METHOD = "event_occlusion"
STRATEGY = "single_event_deletion"


def _open_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wt", encoding="utf-8") if path.suffix == ".gz" else path.open("wt", encoding="utf-8")


def _manifest_path(output_path: Path) -> Path:
    name = output_path.name
    if name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")] + ".manifest.json"
    else:
        name = output_path.stem + ".manifest.json"
    return output_path.with_name(name)


def build_occluded_inputs(
    inputs: np.ndarray,
    pad_id: int,
    padding: str,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Build one deleted-event input and reference record per active position."""

    result = apply_perturbation(
        "single_event_deletion",
        inputs,
        pad_id=pad_id,
        padding=padding,
    )
    return result.inputs, [dict(reference) for reference in result.references]


def _original_sequence_index(
    input_position: int,
    sequence_length: int,
    padded_length: int,
    padding: str,
) -> int:
    retained_length = min(sequence_length, padded_length)
    retained_start = max(sequence_length - padded_length, 0)
    pad_count = padded_length - retained_length if padding == "pre" else 0
    return retained_start + input_position - pad_count if padding == "pre" else retained_start + input_position


def compute_occlusion_batch(
    model: Any,
    inputs: np.ndarray,
    *,
    pad_id: int,
    padding: str,
    cache: DetectorInferenceCache,
    inference_batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compute original and single-event-deletion probabilities for a batch."""

    original_probabilities = cache.predict(model, inputs, inference_batch_size)
    occluded_inputs, references = build_occluded_inputs(inputs, pad_id, padding)
    if occluded_inputs.shape[0]:
        occluded_probabilities = cache.predict(model, occluded_inputs, inference_batch_size)
    else:
        occluded_probabilities = np.empty(0, dtype=np.float64)
    for reference, probability in zip(references, occluded_probabilities):
        original_probability = float(original_probabilities[reference["row_index"]])
        reference["original_probability"] = original_probability
        reference["occluded_probability"] = float(probability)
        reference["occlusion_score"] = original_probability - float(probability)
    return original_probabilities, references


def _load_runtime(config_path: Path, checkpoint: Path, repo_root: Path, split: str, seed_override: int | None):
    """Load one detector runtime without importing TensorFlow until checkpoint load."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    architecture = str(payload.get("data", {}).get("architecture", ""))
    if architecture == "loginsight_lstm":
        try:
            from train_loginsight_lstm import build_model, load_run_config, load_split
        except ImportError:  # pragma: no cover - package-style execution
            from research.scripts.train_loginsight_lstm import build_model, load_run_config, load_split

        dataset, config = load_run_config(config_path)
        if seed_override is not None:
            dataset = replace(dataset, seed=seed_override)
        vocabulary = load_vocabulary(_resolve_path(dataset.vocabulary, repo_root))
        split_path = _resolve_path(getattr(dataset, split), repo_root)
        inputs, labels, metadata = load_split(split_path, vocabulary, config, show_progress=False)
        custom_objects = {}
    elif architecture == "logrobust_attention_bilstm":
        try:
            from train_logrobust_attention_bilstm import (
                TemporalAttention,
                _prepare_loaded_splits,
                build_model,
                load_run_config,
            )
        except ImportError:  # pragma: no cover - package-style execution
            from research.scripts.train_logrobust_attention_bilstm import (
                TemporalAttention,
                _prepare_loaded_splits,
                build_model,
                load_run_config,
            )

        dataset, config = load_run_config(config_path)
        if seed_override is not None:
            dataset = replace(dataset, seed=seed_override)
        vocabulary, paths, loaded = _prepare_loaded_splits(dataset, config, repo_root, show_progress=False)
        inputs, labels, metadata = loaded[split]
        custom_objects = {"TemporalAttention": TemporalAttention, "_TemporalAttention": TemporalAttention}
    else:
        raise ValueError(f"Unsupported detector architecture: {architecture!r}")

    checkpoint_path = _resolve_path(checkpoint, repo_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    adapter = KerasDetectorAdapter(
        DetectorContract(dataset.architecture),
        build_model,
        custom_objects=custom_objects,
    )
    model = adapter.load_model(checkpoint_path)
    return dataset, config, vocabulary, inputs, labels, metadata, model, checkpoint_path


def _default_output(repo_root: Path, architecture: str, seed: int, split: str) -> Path:
    return repo_root / "research/models/attributions" / architecture / (
        f"seed_{seed}_{split}_occlusion.jsonl.gz"
    )


def run_occlusion(
    config_path: Path,
    checkpoint: Path,
    repo_root: Path,
    *,
    split: str = "test",
    output: Path | None = None,
    unit_batch_size: int = 32,
    inference_batch_size: int = 256,
    seed_override: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run single-event deletion occlusion and write verification artifacts."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if unit_batch_size < 1 or inference_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    dataset, config, vocabulary, inputs, labels, metadata, model, checkpoint_path = _load_runtime(
        config_path, checkpoint, repo_root, split, seed_override
    )
    output_path = _resolve_path(output, repo_root) if output is not None else _default_output(
        repo_root, dataset.architecture, dataset.seed, split
    )
    model_stat = checkpoint_path.stat()
    model_key = f"{checkpoint_path.resolve()}:{model_stat.st_size}:{model_stat.st_mtime_ns}"
    cache = DetectorInferenceCache(dataset.architecture, model_key)
    count = 0
    event_count = 0
    with _open_output(output_path) as handle:
        for start in range(0, len(inputs), unit_batch_size):
            end = min(start + unit_batch_size, len(inputs))
            original_probabilities, references = compute_occlusion_batch(
                model,
                inputs[start:end],
                pad_id=vocabulary.pad_id,
                padding=config.padding,
                cache=cache,
                inference_batch_size=inference_batch_size,
            )
            grouped: list[list[dict[str, Any]]] = [[] for _ in range(end - start)]
            for reference in references:
                row_index = reference["row_index"]
                record_metadata = metadata[start + row_index]
                sequence_length = int(record_metadata["sequence_length"])
                original_probability = float(original_probabilities[row_index])
                occluded_probability = float(reference["occluded_probability"])
                position = int(reference["input_position"])
                event_id = int(reference["event_id"])
                grouped[row_index].append({
                    "input_position": position,
                    "original_sequence_index": _original_sequence_index(
                        position, sequence_length, inputs.shape[1], config.padding
                    ),
                    "event_id": event_id,
                    "event_token": vocabulary.decode([event_id])[0],
                    "occluded_probability": occluded_probability,
                    "occlusion_score": original_probability - occluded_probability,
                    "strategy": STRATEGY,
                    "verification_only": True,
                    "prompt_visible": False,
                })
                event_count += 1
            for row_index in range(end - start):
                record_metadata = metadata[start + row_index]
                events = sorted(grouped[row_index], key=lambda item: item["input_position"])
                original_probability = float(original_probabilities[row_index])
                record = {
                    "schema_version": SCRIPT_SCHEMA_VERSION,
                    "method": METHOD,
                    "strategy": STRATEGY,
                    "verification_only": True,
                    "prompt_visible": False,
                    "architecture": dataset.architecture,
                    "seed": dataset.seed,
                    "split": split,
                    "unit_id": record_metadata["unit_id"],
                    "label_id": int(record_metadata["label_id"]),
                    "sequence_length": int(record_metadata["sequence_length"]),
                    "padded_length": int(inputs.shape[1]),
                    "padding": config.padding,
                    "pad_id": vocabulary.pad_id,
                    "probability": original_probability,
                    "event_count": len(events),
                    "events": events,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
            if show_progress:
                print(
                    f"\rOcclusion: {end:,}/{len(inputs):,}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
        if show_progress:
            print(file=sys.stderr)

    stats = asdict(cache.stats)
    manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "method": METHOD,
        "strategy": STRATEGY,
        "verification_only": True,
        "prompt_visible": False,
        "architecture": dataset.architecture,
        "seed": dataset.seed,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "input_path": str(_resolve_path(getattr(dataset, split), repo_root)),
        "output_path": str(output_path),
        "vocabulary": vocabulary.to_dict(),
        "vocabulary_fit_scope": "training_only",
        "padding": config.padding,
        "pad_id": vocabulary.pad_id,
        "unit_batch_size": unit_batch_size,
        "inference_batch_size": inference_batch_size,
        "count": count,
        "event_count": event_count,
        "cache_statistics": stats,
    }
    manifest_path = _manifest_path(output_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Detector training configuration")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained detector checkpoint")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path, help="Output JSONL(.gz) verification file")
    parser.add_argument("--unit-batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, help="Override the seed from the configuration")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = run_occlusion(
        args.config.resolve(),
        args.checkpoint,
        args.repo_root.resolve(),
        split=args.split,
        output=args.output,
        unit_batch_size=args.unit_batch_size,
        inference_batch_size=args.inference_batch_size,
        seed_override=args.seed,
        show_progress=not args.no_progress,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
