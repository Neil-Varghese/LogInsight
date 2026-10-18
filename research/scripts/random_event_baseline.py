"""Create the matched random-event baseline.

For each serialized Integrated Gradients record, this script samples the same
number of non-padding events as the attribution-only baseline, without
replacement. The registered perturbation seed is deterministically derived
per prediction unit, so results do not depend on file iteration order. The
baseline does not use attribution scores, render a prompt, or invoke an LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

try:
    from attribution_only_baseline import _manifest_path, _open_input, _open_output, iter_attribution_records
    from seed_utils import derive_seed, make_numpy_rng
except ImportError:  # pragma: no cover - supports package-style imports
    from research.scripts.attribution_only_baseline import (
        _manifest_path,
        _open_input,
        _open_output,
        iter_attribution_records,
    )
    from research.scripts.seed_utils import derive_seed, make_numpy_rng


SCRIPT_SCHEMA_VERSION = "1.0"
METHOD = "random_event"
SOURCE_METHOD = "integrated_gradients"
SELECTION_SCOPE = "random_event_baseline"
DEFAULT_PERTURBATION_SEED = 2026


def _active_events(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if record.get("method") != SOURCE_METHOD:
        raise ValueError("random-event baseline requires an Integrated Gradients source artifact")
    events = record.get("events")
    if not isinstance(events, list):
        raise ValueError("Integrated Gradients record must contain an events list")
    active: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("every attribution event must be an object")
        if event.get("is_padding", False):
            continue
        if "event_id" not in event or "input_position" not in event:
            raise ValueError("every random-event candidate must contain event_id and input_position")
        active.append(event)
    return active


def select_random_events(
    record: Mapping[str, Any],
    *,
    top_k: int,
    perturbation_seed: int = DEFAULT_PERTURBATION_SEED,
) -> tuple[list[dict[str, Any]], int]:
    """Select a deterministic random event set and return it with its seed."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    active = _active_events(record)
    selection_seed = derive_seed(
        perturbation_seed,
        SELECTION_SCOPE,
        record.get("architecture"),
        record.get("seed"),
        record.get("split"),
        record.get("unit_id"),
        top_k,
    )
    count = min(top_k, len(active))
    if count == 0:
        return [], selection_seed
    rng = make_numpy_rng(perturbation_seed, SELECTION_SCOPE, record.get("architecture"), record.get("seed"), record.get("split"), record.get("unit_id"), top_k)
    selected_indices = rng.choice(len(active), size=count, replace=False)
    selected = [active[int(index)] for index in selected_indices]
    selected.sort(key=lambda event: (int(event["input_position"]), int(event["event_id"])))
    return selected, selection_seed


def build_random_event_record(
    record: Mapping[str, Any],
    *,
    source_path: Path,
    top_k: int,
    perturbation_seed: int = DEFAULT_PERTURBATION_SEED,
) -> dict[str, Any]:
    """Build a random baseline record without copying attribution scores."""

    selected, selection_seed = select_random_events(
        record,
        top_k=top_k,
        perturbation_seed=perturbation_seed,
    )
    selected_events = [
        {
            "input_position": int(event["input_position"]),
            "original_sequence_index": event.get("original_sequence_index"),
            "event_id": event["event_id"],
            "event_token": event.get("event_token"),
        }
        for event in selected
    ]
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "method": METHOD,
        "condition": "random_event",
        "llm_used": False,
        "prompt_rendered": False,
        "source_method": SOURCE_METHOD,
        "source_attribution_file": str(source_path),
        "uses_attribution_scores": False,
        "architecture": record.get("architecture"),
        "seed": record.get("seed"),
        "split": record.get("split"),
        "unit_id": record.get("unit_id"),
        "label_id": record.get("label_id"),
        "probability": record.get("probability"),
        "top_k": top_k,
        "selected_event_count": len(selected_events),
        "selected_event_ids": [event["event_id"] for event in selected_events],
        "selected_events": selected_events,
        "sampling": "without_replacement",
        "perturbation_seed": perturbation_seed,
        "selection_seed": selection_seed,
        "selection_scope": SELECTION_SCOPE,
    }


def _default_output(source_path: Path, top_k: int) -> Path:
    name = source_path.name
    if name.endswith("_integrated_gradients.jsonl.gz"):
        name = name[: -len("_integrated_gradients.jsonl.gz")]
    elif name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")]
    else:
        name = source_path.stem
    return source_path.with_name(f"{name}_random_event_k_{top_k}.jsonl.gz")


def run_random_event(
    source_path: Path,
    *,
    top_k: int,
    perturbation_seed: int = DEFAULT_PERTURBATION_SEED,
    output: Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Create one matched random-event record per Integrated Gradients record."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    source_path = source_path.resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    output_path = (output or _default_output(source_path, top_k)).resolve()
    count = 0
    selected_event_count = 0
    architectures: set[Any] = set()
    seeds: set[Any] = set()
    splits: set[Any] = set()
    with _open_output(output_path) as handle:
        for record in iter_attribution_records(source_path):
            baseline_record = build_random_event_record(
                record,
                source_path=source_path,
                top_k=top_k,
                perturbation_seed=perturbation_seed,
            )
            handle.write(json.dumps(baseline_record, sort_keys=True) + "\n")
            count += 1
            selected_event_count += int(baseline_record["selected_event_count"])
            architectures.add(record.get("architecture"))
            seeds.add(record.get("seed"))
            splits.add(record.get("split"))
            if show_progress:
                print(f"\rRandom-event baseline: {count:,}", end="", file=sys.stderr, flush=True)
    if show_progress:
        print(file=sys.stderr)
    manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "method": METHOD,
        "condition": "random_event",
        "source_method": SOURCE_METHOD,
        "source_attribution_file": str(source_path),
        "output_path": str(output_path),
        "top_k": top_k,
        "llm_used": False,
        "prompt_rendered": False,
        "prompt_visible": False,
        "verification_only": False,
        "uses_attribution_scores": False,
        "sampling": "without_replacement",
        "perturbation_seed": perturbation_seed,
        "selection_scope": SELECTION_SCOPE,
        "count": count,
        "selected_event_count": selected_event_count,
        "architectures": sorted(value for value in architectures if value is not None),
        "seeds": sorted(value for value in seeds if value is not None),
        "splits": sorted(value for value in splits if value is not None),
    }
    _manifest_path(output_path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-file", type=Path, required=True, help="Serialized Integrated Gradients JSONL(.gz)")
    parser.add_argument("--top-k", type=int, required=True, help="Matched number of non-padding events to sample")
    parser.add_argument("--seed", type=int, default=DEFAULT_PERTURBATION_SEED, help="Registered perturbation seed")
    parser.add_argument("--output", type=Path, help="Output JSONL(.gz) baseline file")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = run_random_event(
        args.attribution_file,
        top_k=args.top_k,
        perturbation_seed=args.seed,
        output=args.output,
        show_progress=not args.no_progress,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
