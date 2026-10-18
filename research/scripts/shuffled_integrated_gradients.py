"""Create prompt-visible shuffled Integrated Gradients artifacts for Step 43.

Two provisional shuffle types are supported:

* ``within_block_permutation`` permutes attribution values across the active
  events of the same block while preserving event positions and IDs.
* ``cross_block_donor`` transfers values from a donor block with the same
  active-event count onto the target block's event positions.

Both modes preserve provenance and are separate from verification-only
occlusion/leave-one-event-out artifacts. The primary shuffle type remains a
registration choice for Step 62.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:
    from attribution_only_baseline import _manifest_path, _open_input, _open_output, iter_attribution_records
    from seed_utils import make_numpy_rng
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.attribution_only_baseline import (
        _manifest_path,
        _open_input,
        _open_output,
        iter_attribution_records,
    )
    from research.scripts.seed_utils import make_numpy_rng


SCRIPT_SCHEMA_VERSION = "1.0"
SOURCE_METHOD = "integrated_gradients"
METHOD = "shuffled_integrated_gradients"
SHUFFLE_TYPES = ("within_block_permutation", "cross_block_donor")
DEFAULT_SHUFFLE_SEED = 2026


def _active_events(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if record.get("method") != SOURCE_METHOD:
        raise ValueError("shuffling requires an Integrated Gradients source artifact")
    events = record.get("events")
    if not isinstance(events, list):
        raise ValueError("Integrated Gradients record must contain an events list")
    active: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("every attribution event must be an object")
        if event.get("is_padding", False):
            continue
        if "input_position" not in event or "event_id" not in event or "attribution" not in event:
            raise ValueError("active attribution events require input_position, event_id, and attribution")
        active.append(dict(event))
    if not active:
        raise ValueError("Integrated Gradients record contains no active events")
    return active


def _recompute_ranks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(events, key=lambda event: (-abs(float(event["attribution"])), int(event["input_position"])))
    ranks = {int(event["input_position"]): rank for rank, event in enumerate(ranked, start=1)}
    result = []
    for event in events:
        item = dict(event)
        item["attribution"] = float(item["attribution"])
        item["absolute_attribution"] = abs(item["attribution"])
        item["rank"] = ranks[int(item["input_position"])]
        result.append(item)
    return result


def build_shuffled_prompt_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return only shuffled attribution fields safe for a prompt."""

    if record.get("method") != METHOD:
        raise ValueError("prompt payload requires a shuffled Integrated Gradients record")
    events = []
    for event in record.get("events", []):
        if event.get("is_padding", False):
            continue
        events.append(
            {
                "input_position": int(event["input_position"]),
                "original_sequence_index": event.get("original_sequence_index"),
                "event_id": event["event_id"],
                "event_token": event.get("event_token"),
                "attribution": float(event["attribution"]),
                "absolute_attribution": float(event["absolute_attribution"]),
                "rank": int(event["rank"]),
            }
        )
    return {
        "events": events,
        "shuffle_type": record["shuffle_type"],
    }


def build_shuffled_record(
    record: Mapping[str, Any],
    *,
    shuffle_type: str,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    donor_record: Mapping[str, Any] | None = None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Build one shuffled record while preserving event-position alignment."""

    if shuffle_type not in SHUFFLE_TYPES:
        raise ValueError(f"unsupported shuffle_type: {shuffle_type!r}")
    active = _active_events(record)
    source_values = [float(event["attribution"]) for event in active]
    donor_unit_id = None
    if shuffle_type == "within_block_permutation":
        rng = make_numpy_rng(
            shuffle_seed,
            METHOD,
            shuffle_type,
            record.get("architecture"),
            record.get("seed"),
            record.get("split"),
            record.get("unit_id"),
        )
        values = [source_values[int(index)] for index in rng.permutation(len(source_values))]
    else:
        if donor_record is None:
            raise ValueError("cross_block_donor requires donor_record")
        donor_active = _active_events(donor_record)
        if len(donor_active) != len(active):
            raise ValueError("cross_block_donor requires equal active-event counts")
        donor_unit_id = donor_record.get("unit_id")
        if donor_unit_id in (None, record.get("unit_id")):
            raise ValueError("cross_block_donor requires a distinct donor unit")
        values = [float(event["attribution"]) for event in donor_active]

    shuffled_active = []
    for event, value in zip(active, values):
        item = dict(event)
        item["attribution"] = value
        shuffled_active.append(item)
    shuffled_active = _recompute_ranks(shuffled_active)
    active_by_position = {int(event["input_position"]): event for event in shuffled_active}
    output_events = []
    for event in record["events"]:
        if event.get("is_padding", False):
            output_events.append(dict(event))
        else:
            output_events.append(active_by_position[int(event["input_position"])])
    result = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "method": METHOD,
        "source_method": SOURCE_METHOD,
        "shuffle_type": shuffle_type,
        "shuffle_seed": shuffle_seed,
        "prompt_visible": True,
        "verification_only": False,
        "source_attribution_file": str(source_path) if source_path else None,
        "architecture": record.get("architecture"),
        "seed": record.get("seed"),
        "split": record.get("split"),
        "unit_id": record.get("unit_id"),
        "label_id": record.get("label_id"),
        "probability": record.get("probability"),
        "source_unit_id": record.get("unit_id"),
        "donor_unit_id": donor_unit_id,
        "active_event_count": len(active),
        "events": output_events,
    }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_output(source_path: Path, shuffle_type: str) -> Path:
    name = source_path.name
    suffix = "_integrated_gradients.jsonl.gz"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    elif name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")]
    else:
        name = source_path.stem
    return source_path.with_name(f"{name}_shuffled_{shuffle_type}.jsonl.gz")


def run_shuffled_integrated_gradients(
    source_path: Path,
    *,
    shuffle_type: str,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    donor_path: Path | None = None,
    output: Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Create one shuffled prompt-visible record per source record."""

    source_path = Path(source_path).resolve()
    output_path = Path(output or _default_output(source_path, shuffle_type)).resolve()
    donor_records: list[dict[str, Any]] | None = None
    if shuffle_type == "cross_block_donor":
        if donor_path is None:
            raise ValueError("cross_block_donor requires --donor-file")
        donor_records = list(iter_attribution_records(Path(donor_path).resolve()))
    count = 0
    with _open_output(output_path) as handle:
        for index, record in enumerate(iter_attribution_records(source_path)):
            donor = None
            if donor_records is not None:
                if index >= len(donor_records):
                    raise ValueError("donor file has fewer records than source file")
                donor = donor_records[index]
            shuffled = build_shuffled_record(
                record,
                shuffle_type=shuffle_type,
                shuffle_seed=shuffle_seed,
                donor_record=donor,
                source_path=source_path,
            )
            handle.write(json.dumps(shuffled, sort_keys=True) + "\n")
            count += 1
            if show_progress:
                print(f"\rShuffled Integrated Gradients: {count:,}", end="", file=sys.stderr, flush=True)
    if show_progress:
        print(file=sys.stderr)
    manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "method": METHOD,
        "source_method": SOURCE_METHOD,
        "shuffle_type": shuffle_type,
        "shuffle_seed": shuffle_seed,
        "source_attribution_file": str(source_path),
        "source_attribution_sha256": _sha256(source_path),
        "donor_attribution_file": str(Path(donor_path).resolve()) if donor_path else None,
        "output_path": str(output_path),
        "count": count,
        "prompt_visible": True,
        "verification_only": False,
        "cross_block_pairing": "input_order_one_to_one" if shuffle_type == "cross_block_donor" else None,
        "registration_status": "provisional_before_registration",
    }
    _manifest_path(output_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-file", type=Path, required=True)
    parser.add_argument("--shuffle-type", choices=SHUFFLE_TYPES, default="within_block_permutation")
    parser.add_argument("--donor-file", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = run_shuffled_integrated_gradients(
        args.attribution_file,
        shuffle_type=args.shuffle_type,
        shuffle_seed=args.seed,
        donor_path=args.donor_file,
        output=args.output,
        show_progress=not args.no_progress,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
