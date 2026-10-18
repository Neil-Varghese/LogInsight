#!/usr/bin/env python3
"""Validate the registration-critical definitions frozen in Step 62."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED = {
    "step": 62,
    "status": "frozen_for_registration",
    "evidence_budget.top_k_events": 5,
    "normal_replacement.strategy": "position_aligned_active_event_replacement",
    "normal_replacement.fallback_alignment": "none",
    "human_evidence_f1.empty_prediction_and_empty_reference": "f1_1.0",
    "human_evidence_f1.empty_prediction_nonempty_reference": "f1_0.0",
    "human_evidence_f1.nonempty_prediction_empty_reference": "f1_0.0",
    "primary_shuffle.method": "within_block_permutation",
    "primary_shuffle.seed": 2026,
    "primary_strata.dataset.primary": "HDFS",
    "primary_strata.detector.primary": "loginsight_lstm",
    "primary_strata.llm.primary": "llm_primary",
}


def _get_path(document: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = document
    for component in dotted_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"missing required frozen field: {dotted_path}")
        value = value[component]
    return value


def validate_freeze(document: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    for path, expected in EXPECTED.items():
        try:
            observed = _get_path(document, path)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if observed != expected:
            failures.append(f"{path}: expected {expected!r}, observed {observed!r}")

    no_counterpart = _get_path(document, "normal_replacement.no_counterpart_rule")
    if "unchanged" not in no_counterpart or "unavailable" not in no_counterpart:
        failures.append("normal_replacement.no_counterpart_rule must record unchanged/unavailable handling")

    fixed_effects = _get_path(document, "mixed_effects_analysis.fixed_effects")
    for required in ("condition", "detector", "llm", "dataset"):
        if required not in fixed_effects:
            failures.append(f"mixed_effects_analysis.fixed_effects missing {required!r}")

    if failures:
        raise ValueError("Step 62 protocol freeze validation failed: " + "; ".join(failures))
    return {
        "step": 62,
        "status": "valid",
        "checked_fields": len(EXPECTED),
        "primary_shuffle": document["primary_shuffle"]["method"],
        "top_k_events": document["evidence_budget"]["top_k_events"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("research/experiments/protocol_freeze_step62.json"),
    )
    args = parser.parse_args()
    document = json.loads(args.protocol.read_text(encoding="utf-8"))
    print(json.dumps(validate_freeze(document), indent=2))


if __name__ == "__main__":
    main()
