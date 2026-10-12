"""Run and verify the registered multi-seed detector experiment.

Step 29 requires both detector architectures to be trained with the registered
seed set. Seed-specific artifacts are written into each detector's existing
checkpoint directory, so the seed-13 runs completed in Steps 24 and 26 can be
reused. Missing seeds are trained sequentially by invoking the architecture-
specific trainers with the current Python interpreter.

This script is intentionally an explicit long-running command. It does not
silently replace failed runs, and it writes a resumable manifest after every
architecture/seed pair.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REGISTERED_SEEDS = (13, 42, 73, 101, 202)
ARCHITECTURE_CONFIGS = {
    "loginsight_lstm": "research/experiments/lstm_training_config.json",
    "logrobust_attention_bilstm": "research/experiments/logrobust_attention_bilstm_training_config.json",
}
TRAINING_SCRIPTS = {
    "loginsight_lstm": "research/scripts/train_loginsight_lstm.py",
    "logrobust_attention_bilstm": "research/scripts/train_logrobust_attention_bilstm.py",
}
OUTPUT_DIRS = {
    "loginsight_lstm": Path("research/models/checkpoints/loginsight_lstm"),
    "logrobust_attention_bilstm": Path("research/models/checkpoints/logrobust_attention_bilstm"),
}
REQUIRED_ARTIFACT_PATTERNS = (
    "run_manifest_seed_{seed}.json",
    "best_seed_{seed}.keras",
    "final_seed_{seed}.keras",
    "history_seed_{seed}.json",
    "predictions_train.jsonl.gz",
    "predictions_validation.jsonl.gz",
    "predictions_test.jsonl.gz",
)


@dataclass(frozen=True)
class RunTarget:
    architecture: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.architecture}:seed_{self.seed}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def required_artifacts(architecture: str, seed: int, repo_root: Path) -> list[Path]:
    if architecture not in OUTPUT_DIRS:
        raise ValueError(f"Unknown architecture: {architecture}")
    output_dir = repo_root / OUTPUT_DIRS[architecture]
    return [output_dir / pattern.format(seed=seed) for pattern in REQUIRED_ARTIFACT_PATTERNS]


def validate_run_artifacts(target: RunTarget, repo_root: Path) -> dict[str, Any]:
    """Validate one completed trainer run and return its manifest summary."""

    paths = required_artifacts(target.architecture, target.seed, repo_root)
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"{target.key} is missing artifacts: {', '.join(missing)}")
    manifest_path = paths[0]
    manifest = _load_json(manifest_path)
    if manifest.get("architecture") != target.architecture:
        raise ValueError(f"{manifest_path} has the wrong architecture")
    if int(manifest.get("seed", -1)) != target.seed:
        raise ValueError(f"{manifest_path} has the wrong seed")
    counts = manifest.get("split_counts", {})
    for split in ("train", "validation", "test"):
        if int(counts.get(split, 0)) < 1:
            raise ValueError(f"{manifest_path} has no records for {split}")
    return {
        "manifest": str(manifest_path),
        "architecture": target.architecture,
        "seed": target.seed,
        "split_counts": counts,
        "test_metrics": manifest.get("metrics", {}).get("test", {}),
    }


def build_training_command(target: RunTarget, repo_root: Path) -> list[str]:
    if target.architecture not in TRAINING_SCRIPTS:
        raise ValueError(f"Unknown architecture: {target.architecture}")
    return [
        sys.executable,
        str(repo_root / TRAINING_SCRIPTS[target.architecture]),
        "--config",
        str(repo_root / ARCHITECTURE_CONFIGS[target.architecture]),
        "--seed",
        str(target.seed),
    ]


def all_targets(architectures: Iterable[str], seeds: Iterable[int]) -> list[RunTarget]:
    return [RunTarget(architecture, int(seed)) for architecture in architectures for seed in seeds]


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_multiseed(
    repo_root: Path,
    architectures: tuple[str, ...],
    seeds: tuple[int, ...],
    report_path: Path,
    skip_existing: bool = True,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    for architecture in architectures:
        if architecture not in ARCHITECTURE_CONFIGS:
            raise ValueError(f"Unknown architecture: {architecture}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")

    targets = all_targets(architectures, seeds)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "step": 29,
        "status": "running" if not dry_run else "dry_run",
        "registered_seed_set": list(REGISTERED_SEEDS),
        "architectures": list(architectures),
        "seeds": list(seeds),
        "targets": {},
        "started_at": _utc_now(),
    }
    _write_manifest(report_path, payload)

    for target in targets:
        entry: dict[str, Any] = {
            "architecture": target.architecture,
            "seed": target.seed,
            "command": build_training_command(target, repo_root),
        }
        try:
            if skip_existing:
                entry["artifacts_before_run"] = validate_run_artifacts(target, repo_root)
                entry["status"] = "reused_existing"
            elif dry_run:
                entry["status"] = "planned"
            else:
                raise FileNotFoundError("existing-run reuse disabled")
            if dry_run:
                entry["status"] = "planned_existing" if "artifacts_before_run" in entry else "planned"
        except (FileNotFoundError, ValueError):
            if dry_run:
                entry["status"] = "planned_training"
            else:
                entry["started_at"] = _utc_now()
                completed = subprocess.run(entry["command"], cwd=repo_root, check=False)
                entry["return_code"] = int(completed.returncode)
                if completed.returncode != 0:
                    entry["status"] = "failed"
                    payload["targets"][target.key] = entry
                    _write_manifest(report_path, payload)
                    if not continue_on_error:
                        payload["status"] = "failed"
                        payload["finished_at"] = _utc_now()
                        _write_manifest(report_path, payload)
                        raise RuntimeError(f"Training failed for {target.key}")
                else:
                    entry["status"] = "trained"
                entry["finished_at"] = _utc_now()
                entry["artifacts_after_run"] = validate_run_artifacts(target, repo_root)
        payload["targets"][target.key] = entry
        _write_manifest(report_path, payload)

    failed = [key for key, entry in payload["targets"].items() if entry.get("status") == "failed"]
    planned = [
        key for key, entry in payload["targets"].items()
        if entry.get("status") in {"planned", "planned_training"}
    ]
    payload["failed_targets"] = failed
    payload["planned_targets"] = planned
    payload["status"] = "complete" if not failed and not planned and not dry_run else payload["status"]
    payload["finished_at"] = _utc_now()
    _write_manifest(report_path, payload)
    if failed:
        raise RuntimeError(f"Multi-seed training failed for: {', '.join(failed)}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--architecture",
        choices=("loginsight_lstm", "logrobust_attention_bilstm", "both"),
        default="both",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(REGISTERED_SEEDS))
    parser.add_argument("--report", type=Path, default=Path("research/results/tables/multiseed_training_manifest.json"))
    parser.add_argument("--rerun-existing", action="store_true", help="Do not reuse already complete seed artifacts")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without training")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    architectures = (
        ("loginsight_lstm", "logrobust_attention_bilstm")
        if args.architecture == "both"
        else (args.architecture,)
    )
    report = args.report if args.report.is_absolute() else repo_root / args.report
    payload = run_multiseed(
        repo_root,
        architectures,
        tuple(args.seeds),
        report,
        skip_existing=not args.rerun_existing,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps({
        "report": str(report),
        "status": payload["status"],
        "targets": len(payload["targets"]),
        "failed_targets": payload.get("failed_targets", []),
        "planned_targets": payload.get("planned_targets", []),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
