"""Build the Step 44 no-detector-context prompt.

This condition exposes only the explanation-ready evidence for the anomalous
block. Detector scores, labels, normal comparison records, attribution values,
and internal artifact identifiers are not accepted as prompt fields. The
registered prompt boundary remains the final allowlist check before rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    from blinded_prompt import evidence_record_to_prompt_block
    from prompt_attribution_boundary import build_prompt_payload
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.blinded_prompt import evidence_record_to_prompt_block
    from research.scripts.prompt_attribution_boundary import build_prompt_payload


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiments" / "experiment_config.json"
PROMPT_TEMPLATE_ID = "no_detector_context_v1"
PROMPT_INSTRUCTION = (
    "Analyze the supplied log evidence without relying on detector information. "
    "Return a structured explanation with claims supported by exact event IDs and line IDs. "
    "If the evidence is insufficient, abstain explicitly. Do not invent evidence."
)


def build_no_detector_context(
    anomalous_record: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Build the registered raw-evidence-only prompt payload.

    ``anomalous_record`` may contain storage metadata such as ``BlockId`` or
    ``Label`` because those fields are removed by
    ``evidence_record_to_prompt_block``. Only the normalized event evidence is
    passed to the condition-aware boundary.
    """

    if not isinstance(anomalous_record, Mapping):
        raise ValueError("anomalous_record must be a mapping")
    context = {
        "anomalous_block_evidence": evidence_record_to_prompt_block(anomalous_record),
    }
    return build_prompt_payload("no_detector_context", context, config_path=config_path)


def render_no_detector_context_prompt(
    anomalous_record: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[str, str, str]:
    """Render ``(condition_id, template_id, prompt_text)`` deterministically."""

    payload = build_no_detector_context(anomalous_record, config_path=config_path)
    rendered = PROMPT_INSTRUCTION + "\n\nINPUT_JSON:\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    return "no_detector_context", PROMPT_TEMPLATE_ID, rendered

