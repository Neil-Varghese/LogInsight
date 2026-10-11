"""Download, extract, and provenance-check the official LogHub BGL dataset.

The script verifies the Zenodo archive before extraction, computes MD5 and
SHA-256 hashes for both the archive and extracted raw log, checks the official
line count, and writes a manifest suitable for the research artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


ARCHIVE_URL = "https://zenodo.org/records/8196385/files/BGL.zip?download=1"
EXPECTED_ARCHIVE_MD5 = "4452953c470f2d95fcb32d5f6e733f7a"
EXPECTED_LINE_COUNT = 4_747_963
EXPECTED_ARCHIVE_SIZE = 57_489_019


class ProgressBar:
    """Dependency-free progress bar for download and extraction."""

    def __init__(self, label: str, total: int | None, enabled: bool = True) -> None:
        self.label = label
        self.total = total or 0
        self.enabled = enabled
        self.last_update = 0.0
        self.current = 0

    def update(self, current: int) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if self.total and current < self.total and now - self.last_update < 0.15:
            return
        self.last_update = now
        self.current = current
        if self.total:
            fraction = min(current / self.total, 1.0)
            width = 30
            filled = int(width * fraction)
            bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
            message = f"\r{self.label} [{bar}] {fraction * 100:6.2f}%"
        else:
            message = f"\r{self.label} bytes={current:,}"
        print(message, end="", file=sys.stderr, flush=True)
        if self.total and current >= self.total:
            print(file=sys.stderr)


def file_hashes(path: Path, chunk_size: int = 1024 * 1024) -> dict[str, str | int]:
    """Return byte count, MD5, and SHA-256 for a file."""

    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            byte_count += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {"bytes": byte_count, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def raw_log_statistics(path: Path) -> dict[str, str | int]:
    """Compute raw-log hashes, line count, and a first-line audit sample."""

    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    byte_count = 0
    line_count = 0
    first_line = ""
    last_byte = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            byte_count += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
            line_count += chunk.count(b"\n")
            if not first_line:
                first_line = chunk.splitlines()[0].decode("utf-8", errors="replace")
            last_byte = chunk[-1:]
    if byte_count and last_byte != b"\n":
        line_count += 1
    return {
        "bytes": byte_count,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "line_count": line_count,
        "first_line": first_line,
    }


def download_archive(
    url: str,
    destination: Path,
    expected_md5: str,
    expected_size: int | None = None,
    show_progress: bool = True,
) -> dict[str, str | int]:
    """Download atomically and verify an archive's expected MD5."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "LogInsight-research/1.0"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length else expected_size
        progress = ProgressBar("Downloading BGL.zip", total, enabled=show_progress)
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            downloaded += len(chunk)
            progress.update(downloaded)
        progress.update(downloaded)
    stats = file_hashes(partial)
    if stats["md5"] != expected_md5:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"BGL.zip MD5 mismatch: expected {expected_md5}, got {stats['md5']}"
        )
    if expected_size is not None and stats["bytes"] != expected_size:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"BGL.zip size mismatch: expected {expected_size}, got {stats['bytes']}"
        )
    os.replace(partial, destination)
    return file_hashes(destination)


def extract_bgl_log(
    archive_path: Path,
    output_path: Path,
    show_progress: bool = True,
) -> dict[str, str | int]:
    """Extract the unique BGL.log member and return its statistics."""

    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            info for info in archive.infolist()
            if Path(info.filename).name.lower() == "bgl.log"
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one BGL.log member, found {len(candidates)}")
        member = candidates[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_suffix(output_path.suffix + ".part")
        progress = ProgressBar("Extracting BGL.log", member.file_size, enabled=show_progress)
        with archive.open(member) as source, partial.open("wb") as destination:
            copied = 0
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
                copied += len(chunk)
                progress.update(copied)
            progress.update(copied)
        stats = raw_log_statistics(partial)
        os.replace(partial, output_path)
    return stats


def build_manifest(
    archive_path: Path,
    raw_path: Path,
    archive_stats: dict[str, str | int],
    raw_stats: dict[str, str | int],
    manifest_path: Path,
) -> dict[str, object]:
    """Write the versioned provenance manifest."""

    manifest: dict[str, object] = {
        "manifest_version": "1.0.0",
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": "BGL",
        "dataset_variant": "LogHub_BGL",
        "provenance_status": "downloaded_and_verified",
        "official_source": {
            "loghub_repository": "https://github.com/logpai/loghub",
            "bgl_readme": "https://github.com/logpai/loghub/blob/master/BGL/README.md",
            "zenodo_record": "https://zenodo.org/records/8196385",
            "doi": "https://doi.org/10.5281/zenodo.8196385",
            "record_version": "v8",
            "published_date": "2023-07-31",
            "archive_name": "BGL.zip",
            "archive_url": ARCHIVE_URL,
            "archive_md5": EXPECTED_ARCHIVE_MD5,
            "official_line_count": EXPECTED_LINE_COUNT,
            "official_raw_size": "708.76 MiB",
            "official_time_span": "214.7 days",
            "label_definition": "first-column '-' is non-alert; other alert category tags are alert",
        },
        "local_materialization": {
            "archive_path": str(archive_path),
            "raw_path": str(raw_path),
            "archive": archive_stats,
            "raw_log": raw_stats,
        },
        "verification": {
            "archive_md5_matches_official": archive_stats["md5"] == EXPECTED_ARCHIVE_MD5,
            "archive_size_matches_download_metadata": archive_stats["bytes"] == EXPECTED_ARCHIVE_SIZE,
            "raw_line_count_matches_official": raw_stats["line_count"] == EXPECTED_LINE_COUNT,
        },
        "citation_required": [
            "Adam J. Oliner and Jon Stearley. What Supercomputers Say: A Study of Five System Logs. DSN, 2007.",
            "Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, and Michael R. Lyu. Loghub: A Large Collection of System Logs for AI-driven Log Analytics. ISSRE, 2023.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("research/data/raw/BGL.zip"))
    parser.add_argument("--raw-log", type=Path, default=Path("research/data/raw/BGL.log"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/data/manifests/bgl_manifest.json"),
    )
    parser.add_argument("--url", default=ARCHIVE_URL)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing archive/raw files after verification",
    )
    args = parser.parse_args()
    show_progress = not args.no_progress

    if args.archive.exists() and not args.force:
        archive_stats = file_hashes(args.archive)
        if archive_stats["md5"] != EXPECTED_ARCHIVE_MD5:
            raise ValueError(
                f"Existing archive has wrong MD5 ({archive_stats['md5']}); use --force to redownload"
            )
    else:
        archive_stats = download_archive(
            args.url,
            args.archive,
            EXPECTED_ARCHIVE_MD5,
            EXPECTED_ARCHIVE_SIZE,
            show_progress,
        )

    if args.raw_log.exists() and not args.force:
        raise FileExistsError(
            f"Raw output already exists: {args.raw_log}; use --force to replace it"
        )
    raw_stats = extract_bgl_log(args.archive, args.raw_log, show_progress)
    if raw_stats["line_count"] != EXPECTED_LINE_COUNT:
        raise ValueError(
            f"BGL.log line count mismatch: expected {EXPECTED_LINE_COUNT}, got {raw_stats['line_count']}"
        )
    manifest = build_manifest(
        args.archive,
        args.raw_log,
        archive_stats,
        raw_stats,
        args.manifest,
    )
    print(json.dumps({
        "archive_md5": manifest["local_materialization"]["archive"]["md5"],
        "archive_bytes": manifest["local_materialization"]["archive"]["bytes"],
        "raw_line_count": manifest["local_materialization"]["raw_log"]["line_count"],
        "raw_sha256": manifest["local_materialization"]["raw_log"]["sha256"],
        "manifest": str(args.manifest),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
