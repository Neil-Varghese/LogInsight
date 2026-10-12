"""Shared validation rules for the HDFS and BGL preprocessing artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


VALID_LABELS = {("Normal", 0), ("Anomaly", 1)}


def _require(record: dict[str, Any], fields: set[str], context: str) -> None:
    missing = fields - set(record)
    if missing:
        raise ValueError(f"{context}: missing fields {sorted(missing)}")


def validate_hdfs_records(
    records: Iterable[dict[str, Any]],
    expected_summary: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Validate HDFS block evidence records and return aggregate counts."""

    required = {
        "BlockId", "Sequence", "Evidence", "Label", "LabelId",
        "FirstLineId", "EventCount",
    }
    evidence_required = {
        "Position", "LineId", "EventId", "EventTemplate",
        "ParameterList", "RawLine",
    }
    block_count = 0
    event_count = 0
    labelled_block_count = 0
    previous_first_line: int | None = None
    seen_blocks: set[str] = set()

    for block_count, record in enumerate(records, start=1):
        context = f"HDFS record {block_count}"
        if not isinstance(record, dict):
            raise ValueError(f"{context}: record is not an object")
        _require(record, required, context)
        block_id = record["BlockId"]
        if not isinstance(block_id, str) or not block_id:
            raise ValueError(f"{context}: invalid BlockId")
        if block_id in seen_blocks:
            raise ValueError(f"{context}: duplicate BlockId {block_id}")
        seen_blocks.add(block_id)
        sequence = record["Sequence"]
        evidence = record["Evidence"]
        if not isinstance(sequence, list) or not isinstance(evidence, list):
            raise ValueError(f"{context}: Sequence and Evidence must be lists")
        if len(sequence) != len(evidence) or len(sequence) != record["EventCount"]:
            raise ValueError(f"{context}: sequence/evidence/EventCount length mismatch")
        if not sequence:
            raise ValueError(f"{context}: empty event sequence")
        label_pair = (record["Label"], record["LabelId"])
        if label_pair not in VALID_LABELS:
            raise ValueError(f"{context}: invalid label pair {label_pair}")
        # The manifest's labelled_block_count means that an official label is
        # present, not that the block is anomalous.  Both Normal and Anomaly
        # therefore count as labelled blocks.
        labelled_block_count += 1
        line_ids: list[int] = []
        for position, (event_id, event) in enumerate(zip(sequence, evidence)):
            event_context = f"{context} event {position}"
            if not isinstance(event, dict):
                raise ValueError(f"{event_context}: evidence is not an object")
            _require(event, evidence_required, event_context)
            if event["Position"] != position:
                raise ValueError(f"{event_context}: position is not contiguous")
            if event["EventId"] != event_id:
                raise ValueError(f"{event_context}: Sequence/EventId mismatch")
            if not isinstance(event["ParameterList"], list):
                raise ValueError(f"{event_context}: ParameterList must be a list")
            if not isinstance(event["RawLine"], str) or not event["RawLine"]:
                raise ValueError(f"{event_context}: RawLine is empty")
            if not isinstance(event["LineId"], int):
                raise ValueError(f"{event_context}: LineId must be an integer")
            line_ids.append(event["LineId"])
        if line_ids != sorted(line_ids) or len(set(line_ids)) != len(line_ids):
            raise ValueError(f"{context}: evidence LineIds are not strictly chronological")
        if record["FirstLineId"] != line_ids[0]:
            raise ValueError(f"{context}: FirstLineId does not match evidence")
        if previous_first_line is not None and record["FirstLineId"] <= previous_first_line:
            raise ValueError(f"{context}: blocks are not ordered by FirstLineId")
        previous_first_line = record["FirstLineId"]
        event_count += len(evidence)

    summary = {
        "block_count": block_count,
        "event_count": event_count,
        "labelled_block_count": labelled_block_count,
    }
    if expected_summary is not None:
        expected = {
            "block_count": expected_summary.get("block_count"),
            "event_count": expected_summary.get("evidence_event_count"),
            "labelled_block_count": expected_summary.get("labelled_block_count"),
        }
        for key, expected_value in expected.items():
            if expected_value is not None and summary[key] != expected_value:
                raise ValueError(
                    f"HDFS summary mismatch for {key}: expected {expected_value}, got {summary[key]}"
                )
        if expected_summary.get("partial_input") != 0:
            raise ValueError("HDFS artifact is marked as partial input")
    return summary


def validate_bgl_windows(
    records: Iterable[dict[str, Any]],
    expected_summary: dict[str, Any] | None = None,
    window_seconds: int = 60,
) -> dict[str, int]:
    """Validate BGL window ordering, disjointness, labels, and line ranges."""

    required = {
        "WindowId", "WindowStartEpoch", "WindowEndEpochExclusive",
        "LineStart", "LineEnd", "EventCount", "Label", "LabelId",
    }
    window_count = 0
    alert_window_count = 0
    normal_window_count = 0
    total_lines = 0
    previous_start: int | None = None
    previous_line_end: int | None = None
    seen_ids: set[str] = set()

    for window_count, record in enumerate(records, start=1):
        context = f"BGL window {window_count}"
        if not isinstance(record, dict):
            raise ValueError(f"{context}: record is not an object")
        _require(record, required, context)
        window_id = record["WindowId"]
        if window_id in seen_ids:
            raise ValueError(f"{context}: duplicate WindowId {window_id}")
        seen_ids.add(window_id)
        start = record["WindowStartEpoch"]
        end = record["WindowEndEpochExclusive"]
        line_start = record["LineStart"]
        line_end = record["LineEnd"]
        event_count = record["EventCount"]
        if not all(isinstance(value, int) for value in (start, end, line_start, line_end, event_count)):
            raise ValueError(f"{context}: numeric fields must be integers")
        if end - start != window_seconds:
            raise ValueError(f"{context}: incorrect window duration")
        if line_start < 1 or line_end < line_start:
            raise ValueError(f"{context}: invalid line range")
        if event_count != line_end - line_start + 1:
            raise ValueError(f"{context}: EventCount does not match line range")
        if (record["Label"], record["LabelId"]) not in VALID_LABELS:
            raise ValueError(f"{context}: invalid label pair")
        if previous_start is not None and start <= previous_start:
            raise ValueError(f"{context}: windows are not chronologically ordered")
        if previous_line_end is not None and line_start != previous_line_end + 1:
            raise ValueError(f"{context}: windows overlap or leave a line gap")
        previous_start = start
        previous_line_end = line_end
        total_lines += event_count
        if record["LabelId"] == 1:
            alert_window_count += 1
        else:
            normal_window_count += 1

    summary = {
        "window_count": window_count,
        "alert_window_count": alert_window_count,
        "normal_window_count": normal_window_count,
        "line_count": total_lines,
    }
    if expected_summary is not None:
        expected = {
            "window_count": expected_summary.get("window_count"),
            "alert_window_count": expected_summary.get("alert_window_count"),
            "normal_window_count": expected_summary.get("normal_window_count"),
        }
        for key, expected_value in expected.items():
            if expected_value is not None and summary[key] != expected_value:
                raise ValueError(
                    f"BGL summary mismatch for {key}: expected {expected_value}, got {summary[key]}"
                )
        if expected_summary.get("partial_input") != 0:
            raise ValueError("BGL artifact is marked as partial input")
    return summary
