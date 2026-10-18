"""Compute verification-only leave-one-event-out detector scores.

The registered leave-one-event-out check holds out each retained event in
turn, re-pads the remaining sequence with the detector's configured policy,
and reruns the detector. Under the current protocol this uses the same
single-event deletion operation as Step 34, but writes a separately named
artifact so the two verification diagnostics cannot be confused or silently
fed into an LLM prompt.

The score is ``p(original) - p(sequence with event held out)``. Repeated and
duplicate perturbations use the Step 30 inference cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from detector_interface import DetectorInferenceCache
    from occlusion_scores import (
        _load_runtime,
        _manifest_path,
        _open_output,
        _original_sequence_index,
        build_occluded_inputs,
    )
    from train_loginsight_lstm import _resolve_path
except ImportError:  # pragma: no cover - supports package-style imports
    from research.scripts.detector_interface import DetectorInferenceCache
    from research.scripts.occlusion_scores import (
        _load_runtime,
        _manifest_path,
        _open_output,
        _original_sequence_index,
        build_occluded_inputs,
    )
    from research.scripts.train_loginsight_lstm import _resolve_path


SCRIPT_SCHEMA_VERSION = "1.0"
METHOD = "leave_one_event_out"
STRATEGY = "single_event_deletion"


def build_leave_one_event_out_inputs(
    inputs: np.ndarray,
    pad_id: int,
    padding: str,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Build one held-out-event input and reference per active position."""

    return build_occluded_inputs(inputs, pad_id=pad_id, padding=padding)


def compute_leave_one_event_out_batch(
    model: Any,
    inputs: np.ndarray,
    *,
    pad_id: int,
    padding: str,
    cache: DetectorInferenceCache,
    inference_batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compute original and held-out-event probabilities for a batch."""

    original_probabilities = cache.predict(model, inputs, inference_batch_size)
    held_out_inputs, references = build_leave_one_event_out_inputs(inputs, pad_id, padding)
    if held_out_inputs.shape[0]:
        held_out_probabilities = cache.predict(model, held_out_inputs, inference_batch_size)
    else:
        held_out_probabilities = np.empty(0, dtype=np.float64)
    for reference, probability in zip(references, held_out_probabilities):
        original_probability = float(original_probabilities[reference["row_index"]])
        held_out_probability = float(probability)
        reference["original_probability"] = original_probability
        reference["leave_one_event_out_probability"] = held_out_probability
        reference["leave_one_event_out_score"] = original_probability - held_out_probability
    return original_probabilities, references


def _default_output(repo_root: Path, architecture: str, seed: int, split: str) -> Path:
    return repo_root / "research/models/attributions" / architecture / (
        f"seed_{seed}_{split}_leave_one_event_out.jsonl.gz"
    )


def run_leave_one_event_out(
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
    """Run leave-one-event-out verification and write its separate artifact."""

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
            original_probabilities, references = compute_leave_one_event_out_batch(
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
                position = int(reference["input_position"])
                event_id = int(reference["event_id"])
                grouped[row_index].append({
                    "input_position": position,
                    "original_sequence_index": _original_sequence_index(
                        position, sequence_length, inputs.shape[1], config.padding
                    ),
                    "event_id": event_id,
                    "event_token": vocabulary.decode([event_id])[0],
                    "held_out_probability": float(reference["leave_one_event_out_probability"]),
                    "leave_one_event_out_score": float(reference["leave_one_event_out_score"]),
                    "strategy": STRATEGY,
                    "verification_only": True,
                    "prompt_visible": False,
                })
                event_count += 1
            for row_index in range(end - start):
                record_metadata = metadata[start + row_index]
                events = sorted(grouped[row_index], key=lambda item: item["input_position"])
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
                    "probability": float(original_probabilities[row_index]),
                    "event_count": len(events),
                    "events": events,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
            if show_progress:
                print(
                    f"\rLeave-one-event-out: {end:,}/{len(inputs):,}",
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
    manifest = run_leave_one_event_out(
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
