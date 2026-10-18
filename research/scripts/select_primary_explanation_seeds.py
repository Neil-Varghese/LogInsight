"""Register the primary explanation seed for each detector architecture.

Selection is deliberately performed from validation metrics in the saved
multi-seed run manifests. Test metrics and any explanation/attribution output
are not read for ranking. Every non-primary seed remains available for the
secondary attribution-stability analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_SCHEMA_VERSION = "1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read run manifest {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"run manifest {path} must contain an object")
    architecture = manifest.get("architecture")
    seed = manifest.get("seed")
    validation = manifest.get("metrics", {}).get("validation")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError(f"{path} is missing a valid architecture")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{path} is missing an integer seed")
    if not isinstance(validation, dict):
        raise ValueError(f"{path} is missing validation metrics")
    for field in ("f1", "loss"):
        if isinstance(validation.get(field), bool) or not isinstance(validation.get(field), (int, float)):
            raise ValueError(f"{path} is missing numeric validation {field}")
    if not 0.0 <= float(validation["f1"]) <= 1.0:
        raise ValueError(f"{path} has validation F1 outside [0, 1]")
    if float(validation["loss"]) < 0.0:
        raise ValueError(f"{path} has negative validation loss")
    return manifest


def _candidate(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    validation = manifest["metrics"]["validation"]
    checkpoints = manifest.get("checkpoints", {})
    result = {
        "seed": int(manifest["seed"]),
        "validation": {
            "f1": float(validation["f1"]),
            "loss": float(validation["loss"]),
        },
        "run_manifest": str(path.resolve()),
        "run_manifest_sha256": sha256(path),
    }
    for field in ("precision", "recall", "accuracy", "threshold"):
        if field in validation:
            result["validation"][field] = float(validation[field])
    if isinstance(checkpoints, dict) and checkpoints.get("best"):
        result["best_checkpoint"] = str(checkpoints["best"])
    return result


def _validate_registration(
    manifests: list[Path],
    registration_path: Path | None,
) -> tuple[set[int] | None, set[str] | None]:
    if registration_path is None:
        return None, None
    try:
        with registration_path.open("r", encoding="utf-8") as handle:
            registration = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read multi-seed registration {registration_path}") from error
    if registration.get("status") != "complete":
        raise ValueError("multi-seed registration is not complete")
    registered_seeds = registration.get("registered_seed_set")
    architectures = registration.get("architectures")
    if not isinstance(registered_seeds, list) or not all(isinstance(seed, int) for seed in registered_seeds):
        raise ValueError("multi-seed registration has no valid registered_seed_set")
    if not isinstance(architectures, list) or not all(isinstance(name, str) for name in architectures):
        raise ValueError("multi-seed registration has no valid architectures")
    observed_seeds = {int(_load_manifest(path)["seed"]) for path in manifests}
    observed_architectures = {_load_manifest(path)["architecture"] for path in manifests}
    if observed_seeds != set(registered_seeds):
        raise ValueError("run manifests do not cover the registered seed set")
    if observed_architectures != set(architectures):
        raise ValueError("run manifests do not cover the registered architectures")
    return set(registered_seeds), set(architectures)


def select_primary_seeds(
    manifest_paths: Iterable[Path],
    output: Path,
    registration_path: Path | None = None,
) -> dict[str, Any]:
    paths = [Path(path) for path in manifest_paths]
    if not paths:
        raise ValueError("at least one run manifest is required")
    _validate_registration(paths, registration_path)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for path in paths:
        manifest = _load_manifest(path)
        key = (manifest["architecture"], int(manifest["seed"]))
        if key in seen:
            raise ValueError(f"duplicate run manifest for {key[0]} seed {key[1]}")
        seen.add(key)
        grouped[manifest["architecture"]].append(_candidate(path, manifest))

    architectures: dict[str, Any] = {}
    for architecture in sorted(grouped):
        candidates = sorted(
            grouped[architecture],
            key=lambda item: (-item["validation"]["f1"], item["validation"]["loss"], item["seed"]),
        )
        primary = candidates[0]
        architectures[architecture] = {
            "primary_seed": primary["seed"],
            "primary_run_manifest": primary["run_manifest"],
            "primary_checkpoint": primary.get("best_checkpoint"),
            "primary_validation": primary["validation"],
            "secondary_attribution_stability_seeds": [item["seed"] for item in candidates[1:]],
            "candidates": candidates,
        }

    report = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "audit": "primary_explanation_seed_selection",
        "selection_policy": {
            "selection_split": "validation",
            "primary_metric": "f1",
            "maximize_primary_metric": True,
            "tie_breakers": ["lower_validation_loss", "lower_seed"],
            "test_metrics_used_for_selection": False,
            "explanation_or_attribution_outputs_used_for_selection": False,
            "secondary_seed_role": "attribution_stability",
            "selected_before_explanation_generation": True,
        },
        "registration_source": str(registration_path.resolve()) if registration_path else None,
        "architectures": architectures,
        "input_manifests": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in sorted(paths, key=str)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", action="append", type=Path, dest="run_manifests")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("research/models/checkpoints"),
        help="root searched for run_manifest_seed_*.json when --run-manifest is omitted",
    )
    parser.add_argument(
        "--multi-seed-manifest",
        type=Path,
        default=Path("research/results/tables/multiseed_training_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/tables/primary_explanation_seed_selection.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifests = args.run_manifests
    if not manifests:
        manifests = sorted(args.checkpoint_root.glob("*/run_manifest_seed_*.json"))
    report = select_primary_seeds(manifests, args.output, args.multi_seed_manifest)
    for architecture, result in report["architectures"].items():
        print(
            f"{architecture}: primary seed {result['primary_seed']} "
            f"(validation F1={result['primary_validation']['f1']:.6f}); "
            f"secondary seeds={result['secondary_attribution_stability_seeds']}"
        )


if __name__ == "__main__":
    main()
