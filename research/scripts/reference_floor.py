"""Define attribution-only and matched-random reference levels.

The attribution-only condition is an evidence-alignment reference, not an
upper bound.  The matched random-event condition supplies the random floor.
This module summarizes both at the same detector, strategy, and event-budget
stratum so consistency or faithfulness results are not interpreted against an
unmatched baseline.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping


SCRIPT_SCHEMA_VERSION = "1.0"
ATTRIBUTION_ONLY_CONDITION = "attribution_only"
RANDOM_FLOOR_CONDITIONS = {"random_event", "random_floor", "random"}
DEFAULT_METRIC_FIELD = "metric_value"
STRATUM_FIELDS = (
    "dataset",
    "detector_id",
    "architecture",
    "detector_seed",
    "seed",
    "split",
    "strategy",
    "event_budget",
    "top_k",
)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _condition(record: Mapping[str, Any]) -> str:
    condition = record.get("condition", record.get("method"))
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError("each reference record requires condition or method")
    return condition.strip()


def _metric_value(record: Mapping[str, Any], metric_field: str) -> float:
    if metric_field in record:
        return _finite_number(record[metric_field], metric_field)
    for alias in ("metric_value", "score", "consistency_score", "faithfulness_score"):
        if alias in record:
            return _finite_number(record[alias], alias)
    raise ValueError(f"record requires metric field {metric_field!r}")


def _stratum_key(record: Mapping[str, Any], fields: tuple[str, ...]) -> tuple[tuple[str, Any], ...]:
    return tuple((field, record[field]) for field in fields if field in record)


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    average = mean(values)
    if len(values) < 2:
        low = high = average
    else:
        half_width = 1.96 * stdev(values) / math.sqrt(len(values))
        low, high = average - half_width, average + half_width
    return {
        "count": len(values),
        "mean": average,
        "ci95_low": low,
        "ci95_high": high,
    }


def summarize_reference_levels(
    records: Iterable[Mapping[str, Any]],
    *,
    metric_field: str = DEFAULT_METRIC_FIELD,
    stratum_fields: tuple[str, ...] = STRATUM_FIELDS,
) -> dict[str, Any]:
    """Summarize attribution-only and matched-random levels by stratum."""

    grouped: dict[tuple[tuple[str, Any], ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each reference record must be an object")
        condition = _condition(record)
        value = _metric_value(record, metric_field)
        key = _stratum_key(record, stratum_fields)
        grouped[key][condition].append(value)

    strata: list[dict[str, Any]] = []
    for key, conditions in sorted(grouped.items(), key=lambda item: repr(item[0])):
        attribution_values = conditions.get(ATTRIBUTION_ONLY_CONDITION, [])
        random_values = [
            value
            for condition, values in conditions.items()
            if condition in RANDOM_FLOOR_CONDITIONS
            for value in values
        ]
        attribution = _summary(attribution_values)
        random_floor = _summary(random_values)
        reference_mean = attribution["mean"]
        random_mean = random_floor["mean"]
        condition_summaries: dict[str, dict[str, Any]] = {}
        for condition, values in sorted(conditions.items()):
            summary = _summary(values)
            target_mean = summary["mean"]
            denominator = (reference_mean - random_mean) if reference_mean is not None and random_mean is not None else None
            condition_summaries[condition] = {
                **summary,
                "delta_vs_attribution_only": (
                    target_mean - reference_mean
                    if target_mean is not None and reference_mean is not None else None
                ),
                "delta_vs_random_floor": (
                    target_mean - random_mean
                    if target_mean is not None and random_mean is not None else None
                ),
                "relative_to_random_floor": (
                    (target_mean - random_mean) / denominator
                    if target_mean is not None and random_mean is not None and denominator != 0 else None
                ),
            }
        strata.append({
            "stratum": {field: value for field, value in key},
            "attribution_only_reference": attribution,
            "random_floor": random_floor,
            "conditions": condition_summaries,
            "reference_available": attribution["count"] > 0,
            "random_floor_available": random_floor["count"] > 0,
        })

    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "metric": metric_field,
        "human_evidence_metrics_used": False,
        "attribution_only_is_upper_bound": False,
        "stratum_fields": list(stratum_fields),
        "strata_count": len(strata),
        "evaluated": bool(strata),
        "strata": strata,
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
        payload = payload.get("records", payload.get("results"))
    if not isinstance(payload, list):
        raise ValueError("input must be a JSON array or an object containing records")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-field", default=DEFAULT_METRIC_FIELD)
    args = parser.parse_args(argv)
    report = summarize_reference_levels(_load_records(args.input), metric_field=args.metric_field)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
