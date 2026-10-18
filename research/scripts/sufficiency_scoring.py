"""Evaluate detector sufficiency across multiple perturbation strategies.

Sufficiency records are produced after retaining the cited events and
neutralizing their complement with a registered strategy.  This module scores
the resulting detector outputs without using human evidence labels.  A claim
is never accepted from deletion alone: it needs valid results from at least
two distinct strategies and at least one non-deletion strategy.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


SCRIPT_SCHEMA_VERSION = "1.0"
DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_SCORE_TOLERANCE = 1e-6
DELETION_STRATEGIES = {"deletion", "single_event_deletion"}
CORE_NON_DELETION_STRATEGIES = {
    "normal_block_event_replacement",
    "unknown_token_replacement",
    "normal_block_event_resampling",
    "parameter_masking",
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _record_score(record: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in record and record[name] is not None:
            return _finite_number(record[name], name)
    raise ValueError(f"record requires one of {', '.join(names)}")


def _decision(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{field} must be boolean or 0/1")


def _claim_key(record: Mapping[str, Any]) -> str:
    for field in ("claim_id", "unit_id", "prediction_unit_id", "perturbation_id"):
        value = record.get(field)
        if value is not None and str(value):
            return str(value)
    raise ValueError("each sufficiency record requires claim_id, unit_id, or perturbation_id")


def score_sufficiency_record(
    record: Mapping[str, Any],
    *,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
) -> dict[str, Any]:
    """Score one retained-citation detector output.

    The gap is ``original - retained``.  The normalized gap is emitted only
    when the original score is positive, matching the registered definition
    ``1 - retained / original``.
    """

    if not isinstance(record, Mapping):
        raise ValueError("each sufficiency record must be an object")
    if not math.isfinite(float(decision_threshold)):
        raise ValueError("decision_threshold must be finite")
    if score_tolerance < 0 or not math.isfinite(float(score_tolerance)):
        raise ValueError("score_tolerance must be finite and non-negative")
    strategy = record.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must be a non-empty string")

    original = _record_score(record, "detector_score_original", "detector_score_before")
    retained = _record_score(record, "detector_score_retained", "detector_score_after")
    gap = original - retained
    normalized_gap = (1.0 - retained / original) if original > 0.0 else None
    parser_valid = record.get("parser_valid", True)
    if not isinstance(parser_valid, bool):
        raise ValueError("parser_valid must be boolean when supplied")
    status = record.get("status", "completed")
    if status not in {"completed", "failed", "skipped"}:
        raise ValueError("status must be completed, failed, or skipped")
    valid = status == "completed" and parser_valid

    original_decision = _decision(
        record["original_decision"], "original_decision"
    ) if "original_decision" in record else original >= decision_threshold
    retained_decision = retained >= decision_threshold
    if valid:
        conclusion = "preserved" if retained_decision == original_decision else "not_preserved"
    else:
        conclusion = "unassessed"

    return {
        "claim_id": _claim_key(record),
        "strategy": strategy,
        "detector_score_original": original,
        "detector_score_retained": retained,
        "sufficiency_gap": gap,
        "normalized_sufficiency_gap": normalized_gap,
        "original_decision": original_decision,
        "retained_decision": retained_decision,
        "sufficiency_conclusion": conclusion,
        "counterfactual_valid": valid,
        "parser_valid": parser_valid,
        "status": status,
    }


def _claim_summary(records: list[dict[str, Any]], min_strategies: int) -> dict[str, Any]:
    valid = [record for record in records if record["counterfactual_valid"]]
    strategy_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        strategy_map[record["strategy"]].append(record)
    strategies = set(strategy_map)
    has_non_deletion = bool(strategies & CORE_NON_DELETION_STRATEGIES)
    eligible = len(strategies) >= min_strategies and has_non_deletion
    conclusions = [
        record["sufficiency_conclusion"]
        for strategy_records in strategy_map.values()
        for record in strategy_records[:1]
    ]
    agreement = False
    if eligible:
        agreement = any(
            first["sufficiency_conclusion"] == second["sufficiency_conclusion"]
            for first, second in itertools.combinations(
                [
                    strategy_records[0]
                    for strategy_records in strategy_map.values()
                ],
                2,
            )
        )
    return {
        "claims": 1,
        "valid_strategies": sorted(strategies),
        "valid_strategy_count": len(strategies),
        "has_non_deletion_strategy": has_non_deletion,
        "eligible": eligible,
        "agreement_across_at_least_two_strategies": agreement if eligible else None,
        "sufficiency_conclusions": conclusions,
        "mean_sufficiency_gap": mean(record["sufficiency_gap"] for record in valid) if valid else None,
        "mean_normalized_sufficiency_gap": mean(
            record["normalized_sufficiency_gap"]
            for record in valid
            if record["normalized_sufficiency_gap"] is not None
        ) if any(record["normalized_sufficiency_gap"] is not None for record in valid) else None,
    }


def score_multi_strategy_sufficiency(
    records: Iterable[Mapping[str, Any]],
    *,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
    min_strategies: int = 2,
) -> dict[str, Any]:
    """Aggregate sufficiency gaps and require multi-strategy evidence."""

    if isinstance(min_strategies, bool) or min_strategies < 2:
        raise ValueError("min_strategies must be at least 2")
    scored = [
        score_sufficiency_record(
            record,
            decision_threshold=decision_threshold,
            score_tolerance=score_tolerance,
        )
        for record in records
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in scored:
        grouped[record["claim_id"]].append(record)
        by_strategy[record["strategy"]].append(record)

    claims = {
        claim_id: _claim_summary(rows, min_strategies)
        for claim_id, rows in sorted(grouped.items())
    }
    eligible_claims = [summary for summary in claims.values() if summary["eligible"]]
    consistent_claims = [
        summary for summary in eligible_claims
        if summary["agreement_across_at_least_two_strategies"]
    ]
    deletion_only_claims = [
        summary for summary in claims.values()
        if summary["valid_strategy_count"] > 0 and not summary["has_non_deletion_strategy"]
    ]
    valid = [record for record in scored if record["counterfactual_valid"]]
    sufficiency = {
        "evaluated": bool(eligible_claims),
        "claims_tested": len(claims),
        "eligible_claims": len(eligible_claims),
        "claims_consistent_across_at_least_two_strategies": len(consistent_claims),
        "consistency_rate": (
            len(consistent_claims) / len(eligible_claims) if eligible_claims else None
        ),
        "deletion_only_claims_excluded": len(deletion_only_claims),
        "mean_sufficiency_gap": mean(record["sufficiency_gap"] for record in valid) if valid else None,
        "mean_normalized_sufficiency_gap": mean(
            record["normalized_sufficiency_gap"]
            for record in valid
            if record["normalized_sufficiency_gap"] is not None
        ) if any(record["normalized_sufficiency_gap"] is not None for record in valid) else None,
    }
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "metric": "sufficiency",
        "human_evidence_metrics_used": False,
        "min_strategies": min_strategies,
        "required_non_deletion_strategy": True,
        "decision_threshold": decision_threshold,
        "sufficiency": sufficiency,
        "by_strategy": {
            strategy: {
                "cases": len(rows),
                "valid_cases": sum(record["counterfactual_valid"] for record in rows),
                "mean_sufficiency_gap": mean(
                    record["sufficiency_gap"]
                    for record in rows
                    if record["counterfactual_valid"]
                ) if any(record["counterfactual_valid"] for record in rows) else None,
            }
            for strategy, rows in sorted(by_strategy.items())
        },
        "claims": claims,
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
        payload = payload.get("sufficiency", payload.get("records"))
    if not isinstance(payload, list):
        raise ValueError("input must be a JSON array or an object containing sufficiency records")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-threshold", type=float, default=DEFAULT_DECISION_THRESHOLD)
    parser.add_argument("--score-tolerance", type=float, default=DEFAULT_SCORE_TOLERANCE)
    parser.add_argument("--min-strategies", type=int, default=2)
    args = parser.parse_args(argv)
    report = score_multi_strategy_sufficiency(
        _load_records(args.input),
        decision_threshold=args.decision_threshold,
        score_tolerance=args.score_tolerance,
        min_strategies=args.min_strategies,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
