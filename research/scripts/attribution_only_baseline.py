"""Create the no-LLM attribution-only baseline from Integrated Gradients.

This baseline selects the top-k non-padding events from a serialized
Integrated Gradients artifact. It does not call an LLM, render a prompt, or
consume occlusion/leave-one-event-out values. The full attribution vector
remains in the source artifact; this output stores only the selected event set
and its provenance for the registered attribution-only comparison.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SCRIPT_SCHEMA_VERSION = "1.0"
METHOD = "attribution_only"
SOURCE_METHOD = "integrated_gradients"


def _open_input(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


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


def iter_attribution_records(path: Path) -> Iterator[dict[str, Any]]:
    """Read one JSON object per line from a compressed or plain artifact."""

    with _open_input(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid attribution JSON at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"attribution record at line {line_number} must be an object")
            yield record


def select_top_k_events(record: Mapping[str, Any], top_k: int) -> list[dict[str, Any]]:
    """Select ranked non-padding Integrated Gradients events from one record."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if record.get("method") != SOURCE_METHOD:
        raise ValueError("attribution-only baseline requires an Integrated Gradients source artifact")
    events = record.get("events")
    if not isinstance(events, list):
        raise ValueError("Integrated Gradients record must contain an events list")
    active_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("every attribution event must be an object")
        if event.get("is_padding", False):
            continue
        if event.get("rank") is None:
            raise ValueError("every non-padding attribution event must have a rank")
        if "event_id" not in event or "attribution" not in event:
            raise ValueError("every selected event must contain event_id and attribution")
        active_events.append(dict(event))
    active_events.sort(key=lambda event: (int(event["rank"]), int(event.get("input_position", 0))))
    return active_events[:top_k]


def build_attribution_only_record(
    record: Mapping[str, Any],
    *,
    source_path: Path,
    top_k: int,
) -> dict[str, Any]:
    """Build one no-LLM baseline record without copying the full vector."""

    selected_events = select_top_k_events(record, top_k)
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "method": METHOD,
        "condition": "attribution_only",
        "llm_used": False,
        "prompt_rendered": False,
        "source_method": SOURCE_METHOD,
        "source_attribution_file": str(source_path),
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
    }


def _default_output(source_path: Path, top_k: int) -> Path:
    name = source_path.name
    if name.endswith("_integrated_gradients.jsonl.gz"):
        name = name[: -len("_integrated_gradients.jsonl.gz")]
    elif name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")]
    else:
        name = source_path.stem
    return source_path.with_name(f"{name}_attribution_only_k_{top_k}.jsonl.gz")


def run_attribution_only(
    source_path: Path,
    *,
    top_k: int,
    output: Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Create one attribution-only record per Integrated Gradients record."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    source_path = source_path.resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    output_path = (output or _default_output(source_path, top_k)).resolve()
    count = 0
    selected_event_count = 0
    with _open_output(output_path) as handle:
        for record in iter_attribution_records(source_path):
            baseline_record = build_attribution_only_record(
                record,
                source_path=source_path,
                top_k=top_k,
            )
            handle.write(json.dumps(baseline_record, sort_keys=True) + "\n")
            count += 1
            selected_event_count += int(baseline_record["selected_event_count"])
            if show_progress:
                print(f"\rAttribution-only baseline: {count:,}", end="", file=sys.stderr, flush=True)
    if show_progress:
        print(file=sys.stderr)

    architectures = set()
    seeds = set()
    splits = set()
    for record in iter_attribution_records(output_path):
        architectures.add(record.get("architecture"))
        seeds.add(record.get("seed"))
        splits.add(record.get("split"))
    manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "method": METHOD,
        "condition": "attribution_only",
        "source_method": SOURCE_METHOD,
        "source_attribution_file": str(source_path),
        "output_path": str(output_path),
        "top_k": top_k,
        "llm_used": False,
        "prompt_rendered": False,
        "prompt_visible": False,
        "verification_only": False,
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
    parser.add_argument("--top-k", type=int, required=True, help="Number of top-ranked non-padding events to select")
    parser.add_argument("--output", type=Path, help="Output JSONL(.gz) baseline file")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = run_attribution_only(
        args.attribution_file,
        top_k=args.top_k,
        output=args.output,
        show_progress=not args.no_progress,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
