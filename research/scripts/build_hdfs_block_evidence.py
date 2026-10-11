"""Build HDFS block records with both raw and parsed event evidence.

The compact block file produced by ``build_hdfs_blocks.py`` is sufficient for
an event-ID sequence model, but explanations need the evidence behind each
event.  This script keeps the same deterministic block grouping and adds, for
every event, its exact raw line, parsed template, parsed parameters, source
line ID, event ID, and position within the block.

The output is compressed JSONL.  It is intentionally a separate artifact so
existing detector inputs remain backward compatible.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

from build_hdfs_blocks import ProgressBar, extract_block_id, load_labels


REQUIRED_COLUMNS = {
    "LineId",
    "Date",
    "Time",
    "Pid",
    "Level",
    "Component",
    "Content",
    "EventId",
    "EventTemplate",
    "ParameterList",
}
def parse_parameter_list(value: str) -> list[str]:
    """Safely convert Drain's serialized ParameterList into a JSON list."""

    if value in ("", "[]", None):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid ParameterList {value!r}") from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"ParameterList must be a list or tuple, got {type(parsed).__name__}")
    return [str(parameter) for parameter in parsed]


def reconstruct_hdfs_raw_line(row: dict[str, str]) -> str:
    """Reconstruct the canonical HDFS raw line from structured fields."""

    return (
        f"{row['Date']} {row['Time']} {row['Pid']} {row['Level']} "
        f"{row['Component']}: {row['Content']}"
    )


def _iter_structured_rows(
    structured_path: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Iterable[dict[str, str]]:
    total_bytes = structured_path.stat().st_size
    with structured_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Structured CSV is missing columns: {sorted(missing)}")
        for row in reader:
            yield row
            if progress_callback and reader.line_num % 10_000 == 0:
                progress_callback(handle.buffer.tell(), reader.line_num - 1)
        if progress_callback:
            progress_callback(total_bytes, max(reader.line_num - 1, 0))


def _read_raw_line(raw_handle) -> str:
    line = raw_handle.readline()
    if line == "":
        raise ValueError("Raw log ended before the structured CSV")
    return line.rstrip("\r\n")


def build_hdfs_block_evidence(
    structured_path: Path,
    labels_path: Path,
    output_path: Path,
    raw_log_path: Path | None = None,
    manifest_path: Path | None = None,
    max_rows: int | None = None,
    show_progress: bool = True,
) -> dict[str, int | str]:
    """Build deterministic, evidence-preserving HDFS block records."""

    labels = load_labels(labels_path)
    # Each event tuple is (LineId, evidence dictionary).  The tuple is sorted
    # at write time, preserving the original chronological order.
    sequences: OrderedDict[str, list[tuple[int, dict[str, object]]]] = OrderedDict()
    first_line: dict[str, int] = {}
    total_rows = 0
    rows_with_block_id = 0
    rows_without_block_id = 0
    raw_rows = 0
    raw_source = "reconstructed_from_structured_fields"

    read_progress = ProgressBar(
        "Reading structured CSV with evidence",
        structured_path.stat().st_size,
        enabled=show_progress,
    )

    def report_read_progress(bytes_read: int, rows_read: int) -> None:
        read_progress.update(bytes_read, suffix=f"rows={rows_read:,}")

    raw_handle = None
    if raw_log_path is not None:
        raw_handle = raw_log_path.open("r", encoding="utf-8", newline="")
        raw_source = str(raw_log_path)

    try:
        for row in _iter_structured_rows(structured_path, report_read_progress):
            total_rows += 1
            if max_rows is not None and total_rows > max_rows:
                break

            raw_line = _read_raw_line(raw_handle) if raw_handle is not None else reconstruct_hdfs_raw_line(row)
            if raw_handle is not None:
                expected_raw_line = reconstruct_hdfs_raw_line(row)
                if raw_line != expected_raw_line:
                    raise ValueError(
                        "Raw/structured alignment mismatch at "
                        f"LineId={row['LineId']}: raw log and structured fields disagree"
                    )
            raw_rows += int(raw_handle is not None)
            block_id = extract_block_id(row["Content"])
            if block_id is None:
                rows_without_block_id += 1
                continue

            rows_with_block_id += 1
            line_id = int(row["LineId"])
            evidence = {
                "LineId": line_id,
                "EventId": row["EventId"],
                "EventTemplate": row["EventTemplate"],
                "ParameterList": parse_parameter_list(row["ParameterList"]),
                "ParameterListRaw": row["ParameterList"],
                "RawLine": raw_line,
            }
            if block_id not in sequences:
                sequences[block_id] = []
                first_line[block_id] = line_id
            else:
                first_line[block_id] = min(first_line[block_id], line_id)
            sequences[block_id].append((line_id, evidence))
    finally:
        if raw_handle is not None:
            raw_handle.close()

    read_progress.finish(suffix=f"rows={min(total_rows, max_rows) if max_rows else total_rows:,}")

    partial_input = max_rows is not None and total_rows > max_rows
    if not partial_input:
        missing_labels = set(sequences) - set(labels)
        labels_without_events = set(labels) - set(sequences)
        if missing_labels or labels_without_events:
            raise ValueError(
                "Block/label mismatch: "
                f"missing_labels={len(missing_labels)}, "
                f"labels_without_events={len(labels_without_events)}"
            )

    if raw_handle is None and not partial_input:
        # No external raw log was supplied, so every raw line was reconstructed
        # from the structured fields.  This is still deterministic, but the
        # manifest makes the provenance explicit.
        raw_rows = total_rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_progress = ProgressBar(
        "Writing evidence block JSONL",
        len(sequences),
        enabled=show_progress,
    )
    with gzip.open(output_path, "wt", encoding="utf-8", newline="\n") as handle:
        ordered_block_ids = sorted(sequences, key=lambda value: first_line[value])
        for index, block_id in enumerate(ordered_block_ids, start=1):
            label_name, label_id = labels.get(block_id, (None, None))
            ordered_events = [
                evidence for _, evidence in sorted(sequences[block_id], key=lambda item: item[0])
            ]
            for position, evidence in enumerate(ordered_events):
                evidence["Position"] = position
            record = {
                "BlockId": block_id,
                "Sequence": [event["EventId"] for event in ordered_events],
                "Evidence": ordered_events,
                "Label": label_name,
                "LabelId": label_id,
                "FirstLineId": first_line[block_id],
                "EventCount": len(ordered_events),
                "EvidenceSchemaVersion": "1.0",
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            if index % 1_000 == 0 or index == len(ordered_block_ids):
                write_progress.update(index, suffix=f"blocks={index:,}")
    write_progress.finish(suffix=f"blocks={len(sequences):,}")

    summary: dict[str, int | str] = {
        "input_rows": total_rows if not partial_input else max_rows or 0,
        "rows_with_block_id": rows_with_block_id,
        "rows_without_block_id": rows_without_block_id,
        "block_count": len(sequences),
        "labelled_block_count": sum(block_id in labels for block_id in sequences),
        "evidence_event_count": sum(len(events) for events in sequences.values()),
        "raw_rows": raw_rows,
        "raw_source": raw_source,
        "partial_input": int(partial_input),
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "artifact": "hdfs_block_evidence",
            "schema_version": "1.0",
            "structured_input": str(structured_path),
            "labels_input": str(labels_path),
            "raw_input": raw_source,
            "output": str(output_path),
            "record_fields": [
                "BlockId",
                "Sequence",
                "Evidence",
                "Label",
                "LabelId",
                "FirstLineId",
                "EventCount",
                "EvidenceSchemaVersion",
            ],
            "evidence_fields": [
                "Position",
                "LineId",
                "EventId",
                "EventTemplate",
                "ParameterList",
                "ParameterListRaw",
                "RawLine",
            ],
            "summary": summary,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structured",
        type=Path,
        default=Path("research/output/HDFS.log_structured.csv"),
        help="Drain structured HDFS CSV",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("research/anomaly_label.csv"),
        help="HDFS BlockId label CSV",
    )
    parser.add_argument(
        "--raw-log",
        type=Path,
        default=Path("test sets/HDFS.log"),
        help="Original raw HDFS log, aligned one-to-one with structured rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/data/processed/hdfs_block_evidence.jsonl.gz"),
        help="Compressed JSONL output path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/data/manifests/hdfs_block_evidence_manifest.json"),
        help="Evidence schema and run summary manifest",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional bounded read for smoke tests; do not use for final data",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars",
    )
    args = parser.parse_args()
    if not args.raw_log.exists():
        raise FileNotFoundError(
            f"Raw log not found: {args.raw_log}. Supply the original HDFS log with --raw-log."
        )
    summary = build_hdfs_block_evidence(
        args.structured,
        args.labels,
        args.output,
        raw_log_path=args.raw_log,
        manifest_path=args.manifest,
        max_rows=args.max_rows,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
