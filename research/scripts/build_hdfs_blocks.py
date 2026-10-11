"""Build chronological event sequences grouped by HDFS BlockId.

The input is Drain's structured HDFS CSV. The output is one compressed JSONL
record per complete BlockId sequence. This is a research preprocessing step;
it does not train a detector or choose an evaluation split.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable


BLOCK_ID_PATTERN = re.compile(r"\b(blk_-?\d+)\b")
REQUIRED_COLUMNS = {"LineId", "Content", "EventId"}
LABEL_TO_ID = {"Normal": 0, "Anomaly": 1}


def extract_block_id(content: str) -> str | None:
    """Extract the HDFS BlockId from a parsed Content field."""

    match = BLOCK_ID_PATTERN.search(str(content))
    return match.group(1) if match else None


def load_labels(labels_path: Path) -> dict[str, tuple[str, int]]:
    """Load and validate the official HDFS block labels."""

    labels: dict[str, tuple[str, int]] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["BlockId", "Label"]:
            raise ValueError(f"Expected BlockId,Label columns; found {reader.fieldnames}")
        for row in reader:
            block_id = row["BlockId"]
            label = row["Label"]
            if label not in LABEL_TO_ID:
                raise ValueError(f"Unknown HDFS label {label!r} for {block_id}")
            if block_id in labels:
                raise ValueError(f"Duplicate HDFS label for {block_id}")
            labels[block_id] = (label, LABEL_TO_ID[label])
    return labels


class ProgressBar:
    """Small dependency-free terminal progress bar for long preprocessing runs."""

    def __init__(self, label: str, total: int, enabled: bool = True) -> None:
        self.label = label
        self.total = max(total, 1)
        self.enabled = enabled
        self.last_update = 0.0
        self.current = 0

    def update(self, current: int, suffix: str = "") -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if current < self.total and now - self.last_update < 0.15:
            return
        self.last_update = now
        current = min(max(current, 0), self.total)
        self.current = current
        fraction = current / self.total
        width = 30
        filled = int(width * fraction)
        bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
        message = f"\r{self.label} [{bar}] {fraction * 100:6.2f}% {suffix}"
        print(message, end="", file=sys.stderr, flush=True)
        if current >= self.total:
            print(file=sys.stderr)

    def finish(self, suffix: str = "") -> None:
        if not self.enabled or self.current >= self.total:
            return
        self.last_update = 0.0
        self.update(self.total, suffix)


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
            if progress_callback and reader.line_num % 10000 == 0:
                progress_callback(handle.buffer.tell(), reader.line_num - 1)
        if progress_callback:
            progress_callback(total_bytes, max(reader.line_num - 1, 0))


def build_hdfs_blocks(
    structured_path: Path,
    labels_path: Path,
    output_path: Path,
    max_rows: int | None = None,
    show_progress: bool = True,
) -> dict[str, int]:
    """Group structured events by BlockId and write deterministic JSONL output."""

    labels = load_labels(labels_path)
    sequences: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
    first_line: dict[str, int] = {}
    total_rows = 0
    rows_with_block_id = 0
    rows_without_block_id = 0
    read_progress = ProgressBar(
        "Reading structured CSV",
        structured_path.stat().st_size,
        enabled=show_progress,
    )

    def report_read_progress(bytes_read: int, rows_read: int) -> None:
        read_progress.update(bytes_read, suffix=f"rows={rows_read:,}")

    for row in _iter_structured_rows(structured_path, report_read_progress):
        total_rows += 1
        if max_rows is not None and total_rows > max_rows:
            break
        block_id = extract_block_id(row["Content"])
        if block_id is None:
            rows_without_block_id += 1
            continue
        rows_with_block_id += 1
        line_id = int(row["LineId"])
        if block_id not in sequences:
            sequences[block_id] = []
            first_line[block_id] = line_id
        else:
            first_line[block_id] = min(first_line[block_id], line_id)
        sequences[block_id].append((line_id, row["EventId"]))

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_progress = ProgressBar(
        "Writing block JSONL",
        len(sequences),
        enabled=show_progress,
    )
    with gzip.open(output_path, "wt", encoding="utf-8", newline="\n") as handle:
        ordered_block_ids = sorted(sequences, key=lambda value: first_line[value])
        for index, block_id in enumerate(ordered_block_ids, start=1):
            label_name, label_id = labels.get(block_id, (None, None))
            ordered_events = [
                event_id for _, event_id in sorted(sequences[block_id], key=lambda item: item[0])
            ]
            record = {
                "BlockId": block_id,
                "Sequence": ordered_events,
                "Label": label_name,
                "LabelId": label_id,
                "FirstLineId": first_line[block_id],
                "EventCount": len(ordered_events),
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            if index % 1000 == 0 or index == len(ordered_block_ids):
                write_progress.update(index, suffix=f"blocks={index:,}")
    write_progress.finish(suffix=f"blocks={len(sequences):,}")

    return {
        "input_rows": total_rows if not partial_input else max_rows or 0,
        "rows_with_block_id": rows_with_block_id,
        "rows_without_block_id": rows_without_block_id,
        "block_count": len(sequences),
        "labelled_block_count": sum(block_id in labels for block_id in sequences),
        "partial_input": int(partial_input),
    }


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
        "--output",
        type=Path,
        default=Path("research/data/processed/hdfs_block_sequences.jsonl.gz"),
        help="Compressed JSONL output path",
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
    summary = build_hdfs_blocks(
        args.structured,
        args.labels,
        args.output,
        args.max_rows,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
