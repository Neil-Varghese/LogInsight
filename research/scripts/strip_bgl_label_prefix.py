"""Remove BGL alert-label prefixes without losing auditability.

BGL stores its line label as the first whitespace-separated token.  This
script removes that token from the parser/detector input and writes the
original line plus the removed prefix to a separate compressed audit file.
The audit file must never be passed to the detector or LLM.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path


EXPECTED_TIMESTAMP = re.compile(r"^\d{10}$")


class ProgressBar:
    """Dependency-free byte-based progress bar."""

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
        self.current = current
        fraction = min(max(current, 0) / self.total, 1.0)
        width = 30
        filled = int(width * fraction)
        bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
        print(
            f"\rStripping BGL labels [{bar}] {fraction * 100:6.2f}% {suffix}",
            end="",
            file=sys.stderr,
            flush=True,
        )
        if current >= self.total:
            print(file=sys.stderr)


def split_label_prefix(line: str) -> tuple[str, str, int]:
    """Return (removed prefix, clean line, alert indicator)."""

    fields = line.split(maxsplit=1)
    if len(fields) != 2:
        raise ValueError(f"BGL line does not contain a label and payload: {line!r}")
    prefix, clean_line = fields
    # After removing the label, BGL's next field is its epoch timestamp.  This
    # catches accidental removal of a token from a malformed or already-cleaned
    # input instead of silently creating a contaminated dataset.
    if not EXPECTED_TIMESTAMP.match(clean_line.split(maxsplit=1)[0]):
        raise ValueError(
            f"Unexpected BGL field after label prefix {prefix!r}: {clean_line[:80]!r}"
        )
    return prefix, clean_line, int(prefix != "-")


def file_hashes(path: Path) -> dict[str, str | int]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            byte_count += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {"bytes": byte_count, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def strip_bgl_labels(
    input_path: Path,
    clean_output_path: Path,
    audit_output_path: Path,
    manifest_path: Path | None = None,
    max_lines: int | None = None,
    show_progress: bool = True,
    force: bool = False,
) -> dict[str, object]:
    """Write parser-safe BGL lines and a separate label audit stream."""

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    for path in (clean_output_path, audit_output_path):
        if path.exists() and not force:
            raise FileExistsError(f"Output already exists: {path}; use --force to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)

    clean_part = clean_output_path.with_suffix(clean_output_path.suffix + ".part")
    audit_part = audit_output_path.with_suffix(audit_output_path.suffix + ".part")
    progress = ProgressBar("", input_path.stat().st_size, enabled=show_progress)
    label_counts: Counter[str] = Counter()
    alert_count = 0
    line_count = 0
    bytes_read = 0
    try:
        with (
            input_path.open("r", encoding="utf-8", newline="") as source,
            clean_part.open("w", encoding="utf-8", newline="\n") as clean_handle,
            gzip.open(audit_part, "wt", encoding="utf-8", newline="\n") as audit_handle,
        ):
            for raw_with_ending in source:
                line_count += 1
                if max_lines is not None and line_count > max_lines:
                    break
                raw_line = raw_with_ending.rstrip("\r\n")
                prefix, clean_line, is_alert = split_label_prefix(raw_line)
                label_counts[prefix] += 1
                alert_count += is_alert
                clean_handle.write(clean_line + "\n")
                audit_handle.write(
                    json.dumps(
                        {
                            "LineId": line_count,
                            "OriginalLine": raw_line,
                            "RemovedLabelPrefix": prefix,
                            "IsAlert": is_alert,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                bytes_read += len(raw_with_ending.encode("utf-8"))
                if line_count % 10_000 == 0:
                    progress.update(bytes_read, suffix=f"lines={line_count:,}")
            progress.update(bytes_read, suffix=f"lines={min(line_count, max_lines) if max_lines else line_count:,}")
        clean_part.replace(clean_output_path)
        audit_part.replace(audit_output_path)
    except Exception:
        clean_part.unlink(missing_ok=True)
        audit_part.unlink(missing_ok=True)
        raise

    partial_input = max_lines is not None and line_count > max_lines
    summary: dict[str, object] = {
        "input_path": str(input_path),
        "clean_output_path": str(clean_output_path),
        "audit_output_path": str(audit_output_path),
        "input_line_count": max_lines if partial_input else line_count,
        "clean_line_count": max_lines if partial_input else line_count,
        "alert_line_count": alert_count,
        "normal_line_count": (max_lines if partial_input else line_count) - alert_count,
        "distinct_label_prefixes": len(label_counts),
        "label_prefix_counts": dict(sorted(label_counts.items())),
        "partial_input": int(partial_input),
        "clean_output_contains_label_prefix": False,
        "clean_output": file_hashes(clean_output_path),
        "audit_output": file_hashes(audit_output_path),
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "manifest_version": "1.0.0",
            "artifact": "BGL_label_prefix_stripped",
            "input": str(input_path),
            "clean_output": str(clean_output_path),
            "audit_output": str(audit_output_path),
            "label_rule": "first whitespace-separated token; '-' is normal and any other token is an alert category",
            "leakage_control": "Only clean_output may be used for parsing, detector input, retrieval, attribution, or LLM prompts. audit_output is audit-only.",
            "summary": summary,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("research/data/raw/BGL.log"))
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=Path("research/data/processed/BGL_clean.log"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("research/data/processed/BGL_label_audit.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/data/manifests/bgl_label_strip_manifest.json"),
    )
    parser.add_argument("--max-lines", type=int, default=None, help="Bounded smoke test; not for final data")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace existing outputs")
    args = parser.parse_args()
    summary = strip_bgl_labels(
        args.input,
        args.clean_output,
        args.audit_output,
        args.manifest,
        args.max_lines,
        not args.no_progress,
        args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
