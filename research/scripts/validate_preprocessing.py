"""Run the shared full-artifact HDFS and BGL preprocessing validation."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from preprocessing_validation import validate_bgl_windows, validate_hdfs_records


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            yield record


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_with_progress(
    records: Iterable[dict[str, Any]],
    validator: Callable[..., dict[str, int]],
    expected_summary: dict[str, Any],
    label: str,
    total: int,
    **kwargs: Any,
) -> dict[str, int]:
    count = 0

    def counted_records():
        nonlocal count
        for record in records:
            count += 1
            if count % 10_000 == 0 or count == total:
                print(f"\r{label}: records={count:,}/{total:,}", end="", file=sys.stderr, flush=True)
            yield record

    result = validator(counted_records(), expected_summary=expected_summary, **kwargs)
    print(file=sys.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdfs-evidence",
        type=Path,
        default=Path("research/data/processed/hdfs_block_evidence.jsonl.gz"),
    )
    parser.add_argument(
        "--hdfs-manifest",
        type=Path,
        default=Path("research/data/manifests/hdfs_block_evidence_manifest.json"),
    )
    parser.add_argument(
        "--bgl-windows",
        type=Path,
        default=Path("research/data/processed/BGL_windows.jsonl.gz"),
    )
    parser.add_argument(
        "--bgl-manifest",
        type=Path,
        default=Path("research/data/manifests/bgl_window_manifest.json"),
    )
    args = parser.parse_args()
    hdfs_manifest = load_manifest(args.hdfs_manifest)
    bgl_manifest = load_manifest(args.bgl_manifest)
    hdfs_summary = validate_with_progress(
        iter_jsonl(args.hdfs_evidence),
        validate_hdfs_records,
        hdfs_manifest["summary"],
        "Validating HDFS evidence",
        int(hdfs_manifest["summary"]["block_count"]),
    )
    bgl_summary = validate_with_progress(
        iter_jsonl(args.bgl_windows),
        validate_bgl_windows,
        bgl_manifest["summary"],
        "Validating BGL windows",
        int(bgl_manifest["summary"]["window_count"]),
        window_seconds=int(bgl_manifest["window_rule"]["duration_seconds"]),
    )
    print(json.dumps({"hdfs": hdfs_summary, "bgl": bgl_summary, "status": "PASS"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
