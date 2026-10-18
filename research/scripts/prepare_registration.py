#!/usr/bin/env python3
"""Prepare and verify the Step 63 preregistration artifact manifest.

This script does not contact OSF or AsPredicted. It hashes the exact local
protocol/configuration files and records whether an external registration URL
has been supplied. Exploratory mode can explicitly allow pilot generation;
confirmatory generation remains blocked until external registration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FILES = (
    "research/research_proposal.txt",
    "research/experiments/experiment_config.json",
    "research/experiments/protocol_freeze_step62.json",
    "research/experiments/experiment_ledger_schema.json",
    "research/experiments/llm_explanation_schema.json",
    "research/experiments/annotation_candidate_schema.json",
    "research/experiments/lstm_training_config.json",
    "research/experiments/logrobust_attention_bilstm_training_config.json",
    "research/experiments/llm_model_selection.md",
    "research/results/tables/step61_tost_power_analysis.json",
    "research/results/tables/primary_explanation_seed_selection.json",
    "research/scripts/prompt_attribution_boundary.py",
    "research/scripts/blinded_prompt.py",
    "research/scripts/contrastive_prompt.py",
    "research/scripts/shuffled_integrated_gradients.py",
    "research/scripts/no_detector_context_prompt.py",
    "research/scripts/explanation_schema.py",
    "research/scripts/power_analysis_tost.py",
    "research/experiments/registration/step63_preregistration_packet.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_manifest(
    repo_root: Path,
    files: Iterable[str] = DEFAULT_FILES,
    platform: str | None = None,
    registration_url: str | None = None,
    registration_version: str = "step63-v1",
    exploratory: bool = False,
) -> dict[str, Any]:
    if platform not in {None, "OSF", "AsPredicted"}:
        raise ValueError("platform must be OSF, AsPredicted, or omitted")
    if registration_url and not platform:
        raise ValueError("platform is required when registration_url is supplied")
    if exploratory and registration_url:
        raise ValueError("exploratory mode cannot be combined with a registration URL")

    records = []
    missing = []
    for relative in files:
        path = repo_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        records.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    if missing:
        raise FileNotFoundError("missing preregistration files: " + ", ".join(missing))

    status_lines = git_value(repo_root, "status", "--porcelain") or ""
    llm_output_candidates = []
    for relative in ("research/experiments/pilot", "research/experiments/main"):
        for path in (repo_root / relative).rglob("*"):
            if path.is_file() and any(token in path.name.lower() for token in ("llm", "response", "explanation")):
                llm_output_candidates.append(str(path.relative_to(repo_root)))

    return {
        "step": 63,
        "manifest_version": "1.0",
        "status": "registered" if registration_url else ("exploratory_unregistered" if exploratory else "prepared_pending_external_registration"),
        "registration": {
            "platform": platform,
            "url": registration_url,
            "registered_at_utc": datetime.now(timezone.utc).isoformat() if registration_url else None,
            "version": registration_version,
        },
        "code_version": {
            "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
            "working_tree_dirty": bool(status_lines),
            "working_tree_status": status_lines.splitlines(),
        },
        "llm_output_guard": {
            "llm_output_candidates_found": llm_output_candidates,
            "llm_output_generation_allowed": bool(registration_url or exploratory),
            "external_registration_required_for_current_run": not (registration_url or exploratory),
        },
        "frozen_files": records,
        "registration_packet": "research/experiments/registration/step63_preregistration_packet.md",
        "next_action": (
            "Label all generated outputs exploratory and do not use them for "
            "confirmatory claims; obtain external registration before the main experiment."
            if exploratory else
            "Submit the packet and manifest to OSF or AsPredicted, then regenerate "
            "with --platform and --registration-url before changing status to registered."
            if not registration_url else
            "Record the returned registration URL in experiment ledgers and keep the manifest immutable."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("OSF", "AsPredicted"))
    parser.add_argument("--registration-url")
    parser.add_argument("--exploratory", action="store_true", help="record an exploratory unregistered pilot")
    parser.add_argument(
        "--output", type=Path,
        default=Path("research/experiments/registration/step63_preregistration_manifest.json"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest = build_manifest(
        repo_root,
        platform=args.platform,
        registration_url=args.registration_url,
        exploratory=args.exploratory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
