"""Score detector faithfulness for multiple perturbation strategies.

This module consumes detector scores that were already produced for an
original unit and its perturbation.  It deliberately has no dependency on
human evidence labels, explanation text, or evidence-recovery metrics.  A
faithfulness result therefore remains a detector outcome even when the input
records were created alongside an explanation evaluation ledger.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


SCRIPT_SCHEMA_VERSION = "1.0"
DEFAULT_DIRECTION_TOLERANCE = 1e-6
PREDICTED_EFFECTS = {
    "increase_anomaly_score",
    "decrease_anomaly_score",
    "no_change",
    "uncertain",
    None,
}
RANDOM_CONTROL_STRATEGIES = {
    "random_event_deletion",
    "matched_random_deletion",
    "random_deletion",
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _normalise_effect(value: Any) -> str | None:
    aliases = {
        "increase": "increase_anomaly_score",
        "decrease": "decrease_anomaly_score",
        "unchanged": "no_change",
        "none": "no_change",
    }
    if value in aliases:
        value = aliases[value]
    if value not in PREDICTED_EFFECTS:
        allowed = sorted(effect for effect in PREDICTED_EFFECTS if effect is not None)
        raise ValueError(f"predicted_effect must be one of {allowed} or null")
    return value


def _observed_effect(delta: float, tolerance: float) -> str:
    if delta > tolerance:
        return "increase_anomaly_score"
    if delta < -tolerance:
        return "decrease_anomaly_score"
    return "no_change"


def score_perturbation_record(
    record: Mapping[str, Any],
    *,
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE,
) -> dict[str, Any]:
    """Score one perturbation without consulting human-evidence fields.

    ``detector_score_delta`` is always recomputed as ``after - before``.  If a
    ledger supplied that field, it must agree with the recomputed value within
    ``direction_tolerance`` so stale score records cannot silently pass.
    """

    if not isinstance(record, Mapping):
        raise ValueError("each perturbation record must be an object")
    if direction_tolerance < 0 or not math.isfinite(float(direction_tolerance)):
        raise ValueError("direction_tolerance must be a finite non-negative number")
    strategy = record.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must be a non-empty string")

    before = _finite_number(record.get("detector_score_before"), "detector_score_before")
    after = _finite_number(record.get("detector_score_after"), "detector_score_after")
    delta = after - before
    if "detector_score_delta" in record and record["detector_score_delta"] is not None:
        reported_delta = _finite_number(record["detector_score_delta"], "detector_score_delta")
        if not math.isclose(reported_delta, delta, rel_tol=0.0, abs_tol=direction_tolerance):
            raise ValueError("detector_score_delta does not equal after - before")

    effect = _normalise_effect(record.get("predicted_effect", record.get("expected_direction")))
    observed = _observed_effect(delta, float(direction_tolerance))
    parser_valid = record.get("parser_valid", True)
    if not isinstance(parser_valid, bool):
        raise ValueError("parser_valid must be boolean when supplied")
    status = record.get("status", "completed")
    if status not in {"completed", "failed", "skipped"}:
        raise ValueError("status must be completed, failed, or skipped")
    valid = status == "completed" and parser_valid
    direction_evaluable = valid and effect not in {None, "uncertain"}
    direction_correct = (observed == effect) if direction_evaluable else None

    result = {
        "perturbation_id": record.get("perturbation_id"),
        "unit_id": record.get("unit_id", record.get("prediction_unit_id")),
        "strategy": strategy,
        "detector_score_before": before,
        "detector_score_after": after,
        "detector_score_delta": delta,
        "predicted_effect": effect,
        "observed_effect": observed,
        "direction_evaluable": direction_evaluable,
        "effect_agreement": direction_correct,
        "counterfactual_valid": valid,
        "parser_valid": parser_valid,
        "status": status,
        "random_control": bool(record.get("random_control", False)),
    }
    if "seed" in record:
        result["seed"] = record["seed"]
    return result


def _metric(values: list[bool]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _strategy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["counterfactual_valid"]]
    directional = [record for record in valid if record["direction_evaluable"]]
    return {
        "cases": len(records),
        "valid_cases": len(valid),
        "predicted_direction_accuracy": _metric([record["effect_agreement"] for record in directional]),
        "mean_score_delta": mean(record["detector_score_delta"] for record in valid) if valid else None,
        "counterfactual_validity": len(valid) / len(records) if records else None,
    }


def score_detector_faithfulness(
    records: Iterable[Mapping[str, Any]],
    *,
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE,
) -> dict[str, Any]:
    """Compute schema-compatible faithfulness metrics across strategies.

    The returned ``detector_faithfulness`` object can be copied directly into
    the experiment ledger.  ``by_strategy`` and ``records`` retain audit
    detail, while ``human_evidence_metrics_used`` makes the outcome boundary
    explicit for downstream reviewers.
    """

    scored = [score_perturbation_record(record, direction_tolerance=direction_tolerance) for record in records]
    valid = [record for record in scored if record["counterfactual_valid"]]
    directional = [record for record in valid if record["direction_evaluable"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in scored:
        grouped[record["strategy"]].append(record)

    random_records = [
        record for record in valid
        if record["random_control"] or record["strategy"] in RANDOM_CONTROL_STRATEGIES
    ]
    detector_metrics = {
        "evaluated": bool(valid),
        "predicted_direction_accuracy": _metric([record["effect_agreement"] for record in directional]),
        "mean_score_delta": mean(record["detector_score_delta"] for record in valid) if valid else None,
        "counterfactual_validity": len(valid) / len(scored) if scored else None,
        "random_control_delta": mean(record["detector_score_delta"] for record in random_records)
        if random_records else None,
    }
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "metric": "detector_faithfulness",
        "human_evidence_metrics_used": False,
        "detector_faithfulness": detector_metrics,
        "case_count": len(scored),
        "valid_case_count": len(valid),
        "directional_case_count": len(directional),
        "by_strategy": {strategy: _strategy_summary(rows) for strategy, rows in sorted(grouped.items())},
        "records": scored,
    }


def _load_records(path: Path) -> list[Mapping[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, Mapping):
        payload = payload.get("perturbations", payload.get("records"))
    if not isinstance(payload, list):
        raise ValueError("input must be a JSON array or an object containing perturbations")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON array, JSONL stream, or perturbation object")
    parser.add_argument("--output", type=Path, required=True, help="JSON faithfulness report path")
    parser.add_argument("--direction-tolerance", type=float, default=DEFAULT_DIRECTION_TOLERANCE)
    args = parser.parse_args(argv)
    report = score_detector_faithfulness(
        _load_records(args.input), direction_tolerance=args.direction_tolerance
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
