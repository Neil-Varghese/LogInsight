"""Construct the registered BGL evaluation units.

Frozen rule for this study:

* 60-second, non-overlapping windows anchored to Unix epoch time;
* every line belongs to exactly one window using ``timestamp // 60``;
* a window is anomalous when at least one source line has an alert prefix;
* model input is referenced by clean-log line range only; label audit fields
  never enter the detector or LLM input.

The BGL raw log is chronological, so each window is represented compactly by
its inclusive clean-log line range rather than duplicating 700+ MiB of text.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

from strip_bgl_label_prefix import split_label_prefix


WINDOW_SECONDS = 60


class ProgressBar:
    """Dependency-free byte-based progress bar."""

    def __init__(self, total: int, enabled: bool = True) -> None:
        self.total = max(total, 1)
        self.enabled = enabled
        self.last_update = 0.0

    def update(self, current: int, suffix: str = "") -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if current < self.total and now - self.last_update < 0.15:
            return
        self.last_update = now
        fraction = min(max(current, 0) / self.total, 1.0)
        width = 30
        filled = int(width * fraction)
        bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
        print(
            f"\rBuilding BGL windows [{bar}] {fraction * 100:6.2f}% {suffix}",
            end="",
            file=sys.stderr,
            flush=True,
        )
        if current >= self.total:
            print(file=sys.stderr)


def window_start(timestamp: int, window_seconds: int = WINDOW_SECONDS) -> int:
    """Return the Unix-epoch-aligned start of a fixed window."""

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    return timestamp - (timestamp % window_seconds)


def _write_window(handle, state: dict[str, int | bool] | None) -> None:
    if state is None:
        return
    is_alert = int(bool(state["has_alert"]))
    record = {
        "WindowId": f"bgl_{state['window_start']}",
        "WindowStartEpoch": state["window_start"],
        "WindowEndEpochExclusive": state["window_start"] + state["window_seconds"],
        "LineStart": state["line_start"],
        "LineEnd": state["line_end"],
        "EventCount": state["line_end"] - state["line_start"] + 1,
        "Label": "Anomaly" if is_alert else "Normal",
        "LabelId": is_alert,
        "LabelRule": "any_alert_line_in_window",
        "WindowRule": "non_overlapping_unix_epoch_60_seconds",
    }
    handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def build_bgl_windows(
    clean_log_path: Path,
    audit_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
    window_seconds: int = WINDOW_SECONDS,
    max_lines: int | None = None,
    show_progress: bool = True,
    force: bool = False,
) -> dict[str, object]:
    """Build compact, disjoint BGL window records."""

    if window_seconds != WINDOW_SECONDS:
        raise ValueError(
            f"The registered BGL rule is fixed at {WINDOW_SECONDS} seconds; refusing {window_seconds}"
        )
    if not clean_log_path.exists():
        raise FileNotFoundError(clean_log_path)
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}; use --force to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")

    progress = ProgressBar(clean_log_path.stat().st_size, enabled=show_progress)
    window_count = 0
    alert_window_count = 0
    normal_window_count = 0
    input_lines = 0
    alert_lines = 0
    previous_timestamp: int | None = None
    current_window: dict[str, int | bool] | None = None
    label_counts: Counter[str] = Counter()

    try:
        with (
            clean_log_path.open("r", encoding="utf-8", newline="") as clean_handle,
            gzip.open(audit_path, "rt", encoding="utf-8", newline="") as audit_handle,
            gzip.open(part_path, "wt", encoding="utf-8", newline="\n") as output_handle,
        ):
            clean_iterator = iter(clean_handle)
            audit_iterator = iter(audit_handle)
            while True:
                clean_with_ending = next(clean_iterator, None)
                audit_line = next(audit_iterator, None)
                if clean_with_ending is None and audit_line is None:
                    break
                if clean_with_ending is None or audit_line is None:
                    raise ValueError("Clean log and audit stream have different line counts")
                input_lines += 1
                if max_lines is not None and input_lines > max_lines:
                    break
                clean_line = clean_with_ending.rstrip("\r\n")
                try:
                    audit = json.loads(audit_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid audit JSON at line {input_lines}") from exc
                if audit.get("LineId") != input_lines:
                    raise ValueError(f"Audit line mismatch at input line {input_lines}")
                audit_clean_line = split_label_prefix(audit["OriginalLine"])[1]
                if audit_clean_line != clean_line:
                    raise ValueError(f"Clean/audit content mismatch at input line {input_lines}")
                fields = clean_line.split(maxsplit=1)
                if not fields or not fields[0].isdigit():
                    raise ValueError(f"Missing BGL epoch timestamp at clean line {input_lines}")
                timestamp = int(fields[0])
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise ValueError(
                        f"BGL timestamps are not chronological at line {input_lines}: "
                        f"{timestamp} < {previous_timestamp}"
                    )
                previous_timestamp = timestamp
                prefix = audit["RemovedLabelPrefix"]
                is_alert = int(audit["IsAlert"])
                label_counts[prefix] += 1
                alert_lines += is_alert
                start = window_start(timestamp, window_seconds)
                if current_window is None or start != current_window["window_start"]:
                    if current_window is not None:
                        _write_window(output_handle, current_window)
                        window_count += 1
                        if current_window["has_alert"]:
                            alert_window_count += 1
                        else:
                            normal_window_count += 1
                    current_window = {
                        "window_start": start,
                        "window_seconds": window_seconds,
                        "line_start": input_lines,
                        "line_end": input_lines,
                        "has_alert": bool(is_alert),
                    }
                else:
                    current_window["line_end"] = input_lines
                    current_window["has_alert"] = bool(current_window["has_alert"] or is_alert)
                if input_lines % 10_000 == 0:
                    progress.update(clean_handle.buffer.tell(), suffix=f"lines={input_lines:,}")

            if current_window is not None:
                _write_window(output_handle, current_window)
                window_count += 1
                if current_window["has_alert"]:
                    alert_window_count += 1
                else:
                    normal_window_count += 1
            progress.update(clean_handle.buffer.tell(), suffix=f"lines={min(input_lines, max_lines) if max_lines else input_lines:,}")
        part_path.replace(output_path)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    partial_input = max_lines is not None and input_lines > max_lines
    summary: dict[str, object] = {
        "clean_input": str(clean_log_path),
        "audit_input": str(audit_path),
        "output": str(output_path),
        "window_seconds": window_seconds,
        "overlap_seconds": 0,
        "anchor": "unix_epoch",
        "label_rule": "any_alert_line_in_window",
        "input_line_count": max_lines if partial_input else input_lines,
        "alert_line_count": alert_lines,
        "window_count": window_count,
        "alert_window_count": alert_window_count,
        "normal_window_count": normal_window_count,
        "distinct_label_prefixes": len(label_counts),
        "partial_input": int(partial_input),
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "manifest_version": "1.0.0",
            "artifact": "BGL_fixed_windows",
            "prediction_unit": "BGL_window",
            "window_rule": {
                "duration_seconds": window_seconds,
                "overlap_seconds": 0,
                "anchor": "Unix epoch",
                "assignment": "floor(epoch_timestamp / duration) * duration",
                "label": "Anomaly if any source line in the window has a non '-' BGL alert prefix; otherwise Normal",
                "input": "clean BGL lines only; label audit is used to derive targets and is excluded from model/LLM evidence",
            },
            "line_range_semantics": "inclusive LineStart and LineEnd refer to BGL_clean.log order",
            "clean_input": str(clean_log_path),
            "audit_input": str(audit_path),
            "output": str(output_path),
            "record_fields": [
                "WindowId",
                "WindowStartEpoch",
                "WindowEndEpochExclusive",
                "LineStart",
                "LineEnd",
                "EventCount",
                "Label",
                "LabelId",
                "LabelRule",
                "WindowRule",
            ],
            "summary": summary,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-log", type=Path, default=Path("research/data/processed/BGL_clean.log"))
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("research/data/processed/BGL_label_audit.jsonl.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/data/processed/BGL_windows.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/data/manifests/bgl_window_manifest.json"),
    )
    parser.add_argument("--max-lines", type=int, default=None, help="Bounded smoke test; not for final data")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace existing output")
    args = parser.parse_args()
    summary = build_bgl_windows(
        args.clean_log,
        args.audit,
        args.output,
        args.manifest,
        WINDOW_SECONDS,
        args.max_lines,
        not args.no_progress,
        args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
