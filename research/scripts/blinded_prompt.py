"""Build the Step 41 blinded contrastive prompt.

The prompt contains anomalous evidence, a retrieved normal comparison block,
and basic detector output. It deliberately delegates field enforcement to the
registered prompt-attribution boundary and never accepts attribution values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    from prompt_attribution_boundary import build_prompt_payload
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.prompt_attribution_boundary import build_prompt_payload


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiments" / "experiment_config.json"
PROMPT_TEMPLATE_ID = "blinded_contrastive_v1"
PROMPT_INSTRUCTION = (
    "Analyze the anomalous log block using the normal comparison and detector output. "
    "Return a structured explanation with claims supported by exact event IDs and line IDs. "
    "If the evidence is insufficient, abstain explicitly. Do not invent evidence."
)


def evidence_record_to_prompt_block(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an evidence record to prompt-visible evidence without its unit label."""

    evidence = record.get("Evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence record must contain a non-empty Evidence list")
    events = []
    for index, event in enumerate(evidence):
        if not isinstance(event, Mapping):
            raise ValueError(f"Evidence[{index}] must be an object")
        required = ("Position", "EventId", "LineId", "EventTemplate", "ParameterList", "RawLine")
        missing = [field for field in required if field not in event]
        if missing:
            raise ValueError(f"Evidence[{index}] is missing {', '.join(missing)}")
        if not isinstance(event["EventId"], str) or not event["EventId"]:
            raise ValueError(f"Evidence[{index}].EventId must be a non-empty string")
        if not isinstance(event["ParameterList"], list):
            raise ValueError(f"Evidence[{index}].ParameterList must be a list")
        events.append(
            {
                "position": int(event["Position"]),
                "event_id": event["EventId"],
                "line_id": event["LineId"],
                "event_template": event["EventTemplate"],
                "parameter_list": event["ParameterList"],
                "raw_line": event["RawLine"],
            }
        )
    return {"events": events}


def build_blinded_prompt_context(
    anomalous_record: Mapping[str, Any],
    normal_record: Mapping[str, Any],
    detector_output: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    if not isinstance(detector_output, Mapping):
        raise ValueError("detector_output must be a mapping")
    context = {
        "anomalous_block_evidence": evidence_record_to_prompt_block(anomalous_record),
        "normal_comparison_block": evidence_record_to_prompt_block(normal_record),
        "detector_output": dict(detector_output),
    }
    return build_prompt_payload("blinded", context, config_path=config_path)


def render_blinded_prompt(
    anomalous_record: Mapping[str, Any],
    normal_record: Mapping[str, Any],
    detector_output: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> str:
    payload = build_blinded_prompt_context(
        anomalous_record,
        normal_record,
        detector_output,
        config_path=config_path,
    )
    return PROMPT_INSTRUCTION + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
