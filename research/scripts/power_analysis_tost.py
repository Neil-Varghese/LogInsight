#!/usr/bin/env python3
"""Planning sample sizes for the paired TOST primary endpoint.

The registered planning convention uses a two-sided 95% confidence interval
for the paired mean difference.  Under the planning alternative of a true
difference of zero, the required number of independent paired items is:

    n = ceil(((z_(1-alpha/2) + z_power) * paired_sd / margin) ** 2)

This is the conservative convention used in the research proposal, which
gives approximately 71 pairs for SD=0.15, margin=0.05, alpha=0.05, and
power=0.80.  The script is for planning only; it does not analyze outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, List, Dict, Any


DEFAULT_SD_VALUES = (0.15, 0.20, 0.25, 0.30, 0.35)


def required_pairs(
    paired_sd: float,
    margin: float = 0.05,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> int:
    """Return the planned number of independent paired observations."""
    for name, value in (
        ("paired_sd", paired_sd),
        ("margin", margin),
        ("alpha", alpha),
        ("target_power", target_power),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")
    if alpha >= 1:
        raise ValueError("alpha must be less than one")
    if target_power >= 1:
        raise ValueError("target_power must be less than one")

    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(target_power)
    standardized_requirement = (z_alpha + z_power) * paired_sd / margin
    return math.ceil(standardized_requirement**2)


def build_power_table(
    sd_values: Iterable[float],
    margin: float = 0.05,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> List[Dict[str, Any]]:
    """Build a JSON-serializable table for the requested SD values."""
    values = list(sd_values)
    if not values:
        raise ValueError("at least one paired SD value is required")
    return [
        {
            "paired_sd": sd,
            "required_independent_pairs": required_pairs(
                sd,
                margin=margin,
                alpha=alpha,
                target_power=target_power,
            ),
        }
        for sd in values
    ]


def build_report(
    sd_values: Iterable[float] = DEFAULT_SD_VALUES,
    margin: float = 0.05,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> Dict[str, Any]:
    """Build the complete auditable planning report."""
    return {
        "analysis": "paired_tost_sample_size_planning",
        "step": 61,
        "status": "complete",
        "planning_convention": "two_sided_95_percent_ci_under_true_difference_zero",
        "margin_absolute_f1": margin,
        "alpha": alpha,
        "target_power": target_power,
        "sd_values": list(sd_values),
        "results": build_power_table(
            sd_values,
            margin=margin,
            alpha=alpha,
            target_power=target_power,
        ),
        "interpretation": (
            "These are independent paired annotation items required for the "
            "primary comparison. The planned recruitment target should exceed "
            "the selected SD scenario to allow for unusable cases and annotation loss."
        ),
    }


def _parse_sd_values(raw: str) -> List[float]:
    try:
        values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SD values must be comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one SD value is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sd-values",
        type=_parse_sd_values,
        default=list(DEFAULT_SD_VALUES),
        help="comma-separated paired SD values (default: 0.15,0.20,0.25,0.30,0.35)",
    )
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80, dest="target_power")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/tables/step61_tost_power_analysis.json"),
    )
    args = parser.parse_args()

    report = build_report(
        sd_values=args.sd_values,
        margin=args.margin,
        alpha=args.alpha,
        target_power=args.target_power,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
