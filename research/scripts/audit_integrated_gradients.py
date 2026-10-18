"""Audit Integrated Gradients stability and sanity conditions for reviewer gate R4.

The audit is independent of TensorFlow and consumes serialized attribution
artifacts. It reports baseline-probability and completeness-error
distributions, compares artifacts generated with different integration-step
counts, accepts logit-space Integrated Gradients artifacts as a separate
target, and compares trained-model attributions with randomized-model
attributions. Verification-only occlusion and leave-one-event-out vectors are
not accepted as inputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
METHOD = "integrated_gradients"


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def _manifest_path(path: Path) -> Path:
    if path.name.endswith(".jsonl.gz"):
        return path.with_name(path.name[: -len(".jsonl.gz")] + ".manifest.json")
    return path.with_name(path.stem + ".manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("distribution requires finite, non-empty values")
    return {
        "count": len(values),
        "min": min(values),
        "q05": _percentile(values, 0.05),
        "q25": _percentile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "q75": _percentile(values, 0.75),
        "q95": _percentile(values, 0.95),
        "max": max(values),
    }


def _load_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest_path = _manifest_path(path)
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{manifest_path} must contain an object")
        manifest = payload
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            if record.get("method") != METHOD:
                raise ValueError(f"{path}:{line_number} is not an Integrated Gradients record")
            unit_id = str(record.get("unit_id", ""))
            if not unit_id:
                raise ValueError(f"{path}:{line_number} has no unit_id")
            if unit_id in seen:
                raise ValueError(f"{path} contains duplicate unit_id {unit_id!r}")
            seen.add(unit_id)
            if record.get("target") not in {"anomaly_probability", "anomaly_logit"}:
                raise ValueError(f"{path}:{line_number} has an unsupported target")
            for field in ("baseline_probability", "completeness_error"):
                if not _finite(record.get(field)):
                    raise ValueError(f"{path}:{line_number} has non-finite {field}")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no records")
    if manifest.get("count") is not None and int(manifest["count"]) != len(records):
        raise ValueError(f"{path} manifest count does not match records")
    return manifest, records


def _artifact_summary(path: Path) -> dict[str, Any]:
    manifest, records = _load_records(path)
    targets = sorted({str(record["target"]) for record in records})
    spaces = sorted({str(record.get("output_space", "probability")) for record in records})
    steps = sorted({int(record["integration_steps"]) for record in records if "integration_steps" in record})
    baseline_values = [float(record["baseline_probability"]) for record in records]
    completeness_values = [float(record["completeness_error"]) for record in records]
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "manifest_path": str(_manifest_path(path).resolve()) if _manifest_path(path).exists() else None,
        "architecture": manifest.get("architecture", records[0].get("architecture")),
        "seed": manifest.get("seed", records[0].get("seed")),
        "split": manifest.get("split", records[0].get("split")),
        "checkpoint": manifest.get("checkpoint"),
        "checkpoint_load_mode": manifest.get("checkpoint_load_mode"),
        "randomize_model_seed": manifest.get("randomize_model_seed"),
        "target": targets,
        "output_space": spaces,
        "integration_steps": steps,
        "records": len(records),
        "baseline_probability_distribution": distribution(baseline_values),
        "completeness_error_distribution": distribution(completeness_values),
    }


def _event_vectors(record: dict[str, Any]) -> tuple[list[int], list[float]]:
    events = record.get("events")
    if not isinstance(events, list):
        raise ValueError(f"record {record.get('unit_id')} has no events list")
    active = [event for event in events if isinstance(event, dict) and event.get("rank") is not None]
    active.sort(key=lambda event: int(event["input_position"]))
    positions = [int(event["input_position"]) for event in active]
    values = [float(event["attribution"]) for event in active]
    if not positions or any(not math.isfinite(value) for value in values):
        raise ValueError(f"record {record.get('unit_id')} has invalid active attributions")
    return positions, values


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left == right else 0.0
    return numerator / (left_norm * right_norm)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    centered_left = [value - left_mean for value in left_ranks]
    centered_right = [value - right_mean for value in right_ranks]
    return _cosine(centered_left, centered_right)


def _top_k_positions(record: dict[str, Any], top_k: int) -> set[int]:
    events = [event for event in record.get("events", []) if isinstance(event, dict) and event.get("rank") is not None]
    events.sort(key=lambda event: int(event["rank"]))
    return {int(event["input_position"]) for event in events[:top_k]}


def _pair_metrics(left: dict[str, Any], right: dict[str, Any], top_k: int) -> dict[str, float]:
    left_positions, left_values = _event_vectors(left)
    right_positions, right_values = _event_vectors(right)
    if left_positions != right_positions:
        raise ValueError(f"unit {left.get('unit_id')} has different active event positions")
    left_top = _top_k_positions(left, top_k)
    right_top = _top_k_positions(right, top_k)
    union = left_top | right_top
    jaccard = len(left_top & right_top) / len(union) if union else 1.0
    return {
        "signed_cosine": _cosine(left_values, right_values),
        "signed_spearman": _spearman(left_values, right_values),
        "top_k_jaccard": jaccard,
    }


def compare_artifacts(left_path: Path, right_path: Path, top_k: int = 5) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    left_manifest, left_records = _load_records(left_path)
    right_manifest, right_records = _load_records(right_path)
    identity = {}
    for field in ("architecture", "seed", "split", "target", "output_space"):
        left_value = left_manifest.get(field, left_records[0].get(field))
        right_value = right_manifest.get(field, right_records[0].get(field))
        if left_value is not None and right_value is not None and left_value != right_value:
            raise ValueError(f"comparison artifacts have different {field}: {left_value!r} != {right_value!r}")
        identity[field] = left_value if left_value is not None else right_value
    left_by_unit = {str(record["unit_id"]): record for record in left_records}
    right_by_unit = {str(record["unit_id"]): record for record in right_records}
    if set(left_by_unit) != set(right_by_unit):
        raise ValueError("comparison artifacts do not contain the same prediction units")
    metrics = [
        _pair_metrics(left_by_unit[unit_id], right_by_unit[unit_id], top_k)
        for unit_id in sorted(left_by_unit)
    ]
    return {
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "left_integration_steps": left_manifest.get("integration_steps"),
        "right_integration_steps": right_manifest.get("integration_steps"),
        "left_randomize_model_seed": left_manifest.get("randomize_model_seed"),
        "right_randomize_model_seed": right_manifest.get("randomize_model_seed"),
        **identity,
        "top_k": top_k,
        "unit_count": len(metrics),
        "metric_distributions": {
            name: distribution([metric[name] for metric in metrics])
            for name in ("signed_cosine", "signed_spearman", "top_k_jaccard")
        },
    }


def build_audit(
    *,
    attribution_files: Iterable[Path],
    logit_attribution_files: Iterable[Path] = (),
    step_stability_pairs: Iterable[tuple[Path, Path]] = (),
    randomized_model_pairs: Iterable[tuple[Path, Path]] = (),
    top_k: int = 5,
    output: Path | None = None,
) -> dict[str, Any]:
    attribution_files = [Path(path) for path in attribution_files]
    logit_attribution_files = [Path(path) for path in logit_attribution_files]
    if not attribution_files and not logit_attribution_files:
        raise ValueError("at least one attribution artifact is required")
    step_pairs = [tuple(Path(path) for path in pair) for pair in step_stability_pairs]
    randomized_pairs = [tuple(Path(path) for path in pair) for pair in randomized_model_pairs]
    probability_summaries = [_artifact_summary(path) for path in attribution_files]
    logit_summaries = [_artifact_summary(path) for path in logit_attribution_files]
    step_reports = [compare_artifacts(left, right, top_k) for left, right in step_pairs]
    randomized_reports = [compare_artifacts(left, right, top_k) for left, right in randomized_pairs]
    for summary in probability_summaries:
        if summary["target"] != ["anomaly_probability"] or summary["output_space"] != ["probability"]:
            raise ValueError("--attribution-file inputs must be probability-space Integrated Gradients")
        if summary["randomize_model_seed"] is not None:
            raise ValueError("randomized-model artifacts cannot be primary probability artifacts")
    for summary in logit_summaries:
        if summary["target"] != ["anomaly_logit"] or summary["output_space"] != ["logit"]:
            raise ValueError("--logit-attribution-file inputs must be logit-space Integrated Gradients")
        if summary["randomize_model_seed"] is not None:
            raise ValueError("randomized-model artifacts cannot be primary logit artifacts")
    for comparison in step_reports:
        if comparison["left_integration_steps"] == comparison["right_integration_steps"]:
            raise ValueError("step-stability pairs must use different integration-step counts")
        if comparison["left_randomize_model_seed"] is not None or comparison["right_randomize_model_seed"] is not None:
            raise ValueError("step-stability pairs must use trained-model artifacts")
    for comparison in randomized_reports:
        if comparison["left_randomize_model_seed"] is not None:
            raise ValueError("the left side of a randomized-model pair must use trained weights")
        if comparison["right_randomize_model_seed"] is None:
            raise ValueError("the right side of a randomized-model pair must record randomize_model_seed")
    stability_by_architecture = {
        report["architecture"]: report["metric_distributions"]["signed_cosine"]["mean"]
        for report in step_reports
        if report.get("architecture") is not None
    }
    randomized_findings = []
    for comparison in randomized_reports:
        architecture = comparison.get("architecture")
        stability_mean = stability_by_architecture.get(architecture)
        randomized_mean = comparison["metric_distributions"]["signed_cosine"]["mean"]
        passed = None if stability_mean is None else randomized_mean < stability_mean
        randomized_findings.append({
            "architecture": architecture,
            "mean_step_stability_signed_cosine": stability_mean,
            "mean_trained_vs_randomized_signed_cosine": randomized_mean,
            "pass": passed,
            "status": "pass" if passed else "inconclusive" if passed is None else "fail",
        })
    sanity_values = [finding["pass"] for finding in randomized_findings]
    sanity_pass = bool(sanity_values) and all(value is True for value in sanity_values)
    probability_architectures = sorted({item["architecture"] for item in probability_summaries if item["architecture"]})
    logit_architectures = sorted({item["architecture"] for item in logit_summaries if item["architecture"]})
    step_architectures = sorted({item["architecture"] for item in step_reports if item.get("architecture")})
    randomized_architectures = sorted({item["architecture"] for item in randomized_reports if item.get("architecture")})
    coverage_complete = bool(probability_architectures) and (
        probability_architectures == logit_architectures == step_architectures == randomized_architectures
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "audit": "integrated_gradients_r4",
        "policy": {
            "baseline_probability_distribution": True,
            "completeness_error_distribution": True,
            "step_count_stability_metrics": ["signed_cosine", "signed_spearman", "top_k_jaccard"],
            "logit_space_integrated_gradients_separate_from_probability_space": True,
            "randomized_model_sanity_rule": "mean trained-vs-randomized signed cosine must be lower than mean step-count stability signed cosine",
            "verification_only_methods_accepted": [],
            "quality_thresholds_pre_registered": False,
        },
        "coverage": {
            "probability_architectures": probability_architectures,
            "logit_architectures": logit_architectures,
            "step_stability_architectures": step_architectures,
            "randomized_model_architectures": randomized_architectures,
            "complete": coverage_complete,
            "status": "complete" if coverage_complete else "incomplete",
        },
        "interpretation_constraints": {
            "coverage_complete_does_not_imply_attribution_quality": True,
            "randomized_model_sanity_does_not_replace_completeness_or_step_stability": True,
            "no_pre_registered_quality_threshold": True,
            "report_distributions_and_architecture_specific_results": True,
        },
        "probability_artifacts": probability_summaries,
        "logit_artifacts": logit_summaries,
        "step_count_stability": step_reports,
        "randomized_model_sanity": {
            "comparisons": randomized_reports,
            "architecture_findings": randomized_findings,
            "pass": sanity_pass,
            "status": "pass" if sanity_pass else "fail",
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-file", action="append", type=Path, default=[])
    parser.add_argument("--logit-attribution-file", action="append", type=Path, default=[])
    parser.add_argument("--step-stability-pair", action="append", nargs=2, type=Path, default=[])
    parser.add_argument("--randomized-model-pair", action="append", nargs=2, type=Path, default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/integrated_gradients_r4_audit.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_audit(
        attribution_files=args.attribution_file,
        logit_attribution_files=args.logit_attribution_file,
        step_stability_pairs=args.step_stability_pair,
        randomized_model_pairs=args.randomized_model_pair,
        top_k=args.top_k,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
