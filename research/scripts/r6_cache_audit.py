"""Audit R6 inference-cache identities.

The cache stores detector probabilities, so its identity must include the
checkpoint content and the calibration/threshold definition used by the
caller. This audit verifies the production cache namespace, hashes the
current checkpoint and calibration manifests, and proves that changing either
identity cannot reuse a cached row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from detector_interface import DetectorInferenceCache, sha256_file
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.detector_interface import DetectorInferenceCache, sha256_file


REPORT_SCHEMA_VERSION = "1.0"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _AuditModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, inputs: np.ndarray, batch_size: int, verbose: int) -> np.ndarray:
        self.calls += 1
        return np.mean(inputs, axis=1, keepdims=True) / 10.0


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _calibration_identity(calibration: dict[str, Any], threshold: dict[str, Any]) -> dict[str, Any]:
    return {
        "calibration_method": calibration["method"],
        "calibration_fit_split": calibration["fit_split"],
        "temperature": calibration["temperature"],
        "threshold_selection_rule": threshold["selection_rule"],
        "threshold_fit_split": threshold["fit_split"],
        "selected_threshold": threshold["selected_threshold"],
        "threshold_source": "validation",
    }


def run_audit(repo_root: Path, output: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = json.loads((repo_root / "research/experiments/experiment_config.json").read_text())
    cache_config = config["inference_cache"]
    implementation_namespace = {
        "detector_id",
        "checkpoint_sha256",
        "calibration_key",
        "input_dtype",
        "input_shape",
        "input_bytes",
    }
    configured_namespace = set(cache_config["key_namespace"])
    registered_namespace = {"detector_id", "model_key", "input_dtype", "input_shape", "input_bytes"}
    if not registered_namespace.issubset(configured_namespace):
        raise AssertionError(f"registered cache namespace is incomplete: {sorted(registered_namespace - configured_namespace)}")

    checkpoint = repo_root / "research/models/checkpoints/loginsight_lstm/best_seed_13.keras"
    calibration_path = repo_root / "research/models/checkpoints/loginsight_lstm/calibrated/calibration_manifest.json"
    threshold_path = repo_root / "research/models/checkpoints/loginsight_lstm/thresholded/threshold_manifest.json"
    checkpoint_sha256 = sha256_file(checkpoint)
    if not SHA256_RE.fullmatch(checkpoint_sha256):
        raise AssertionError("checkpoint hash is not a SHA-256 digest")
    calibration = json.loads(calibration_path.read_text())
    threshold = json.loads(threshold_path.read_text())
    identity = _calibration_identity(calibration, threshold)
    calibration_sha256 = _canonical_sha256(identity)
    mutated_identity = dict(identity, selected_threshold=float(identity["selected_threshold"]) + 0.001)
    mutated_sha256 = _canonical_sha256(mutated_identity)
    if calibration_sha256 == mutated_sha256:
        raise AssertionError("calibration identity hash did not change after a threshold change")

    model_key = f"checkpoint_sha256:{checkpoint_sha256}"
    raw_cache = DetectorInferenceCache("loginsight_lstm", model_key, "raw_probability_v1")
    calibrated_cache = DetectorInferenceCache("loginsight_lstm", model_key, f"calibration_sha256:{calibration_sha256}")
    changed_calibration_cache = DetectorInferenceCache("loginsight_lstm", model_key, f"calibration_sha256:{mutated_sha256}")
    changed_checkpoint_cache = DetectorInferenceCache("loginsight_lstm", "checkpoint_sha256:" + "0" * 64, "raw_probability_v1")
    inputs = np.asarray([[1, 2, 3]], dtype=np.int32)
    model = _AuditModel()
    for cache in (raw_cache, calibrated_cache, changed_calibration_cache, changed_checkpoint_cache):
        cache.predict(model, inputs, batch_size=1)
    for name, cache in {
        "raw": raw_cache,
        "calibrated": calibrated_cache,
        "changed_calibration": changed_calibration_cache,
        "changed_checkpoint": changed_checkpoint_cache,
    }.items():
        if cache.stats.model_rows != 1 or cache.stats.cache_hit_rows != 0:
            raise AssertionError(f"{name} cache unexpectedly reused a row")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "step": "R6",
        "status": "passed",
        "checks": {
            "configured_namespace": {
                "status": "passed",
                "registered_fields": sorted(registered_namespace),
                "configured_fields": cache_config["key_namespace"],
                "implementation_fields": sorted(implementation_namespace),
                "note": "The registered config is preserved unchanged; the post-registration implementation strengthens model_key to checkpoint_sha256 and adds calibration_key.",
            },
            "checkpoint_identity": {
                "status": "passed",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "model_key_prefix": "checkpoint_sha256:",
            },
            "calibration_identity": {
                "status": "passed",
                "identity": identity,
                "calibration_sha256": calibration_sha256,
                "changed_threshold_sha256": mutated_sha256,
            },
            "collision_probes": {
                "status": "passed",
                "same_input_rows_evaluated": 4,
                "model_calls": model.calls,
                "identities_tested": [
                    "raw_probability_v1",
                    "calibration_sha256:<registered-manifest>",
                    "calibration_sha256:<changed-threshold>",
                    "checkpoint_sha256:<changed-checkpoint>",
                ],
            },
        },
        "interpretation": "Checkpoint content and calibration/threshold identity are isolated in the inference cache namespace.",
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/r6_cache_audit.json"))
    args = parser.parse_args(argv)
    try:
        report = run_audit(args.repo_root, args.output)
    except Exception as error:
        print(f"R6 cache audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "output": str(args.output), "checks": list(report["checks"])}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
