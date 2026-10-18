"""Build the Step 42 normal-comparison × attribution prompt factor.

The factor has two normal-comparison levels (present/absent) crossed with
three attribution levels (none/Integrated Gradients/shuffled). The registered
prompt boundary remains the final allowlist: this module selects a condition
and supplies only the fields allowed for that cell.
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
PROMPT_TEMPLATE_ID = "contrastive_factor_v1"
ATTRIBUTION_MODES = ("none", "integrated_gradients", "shuffled_integrated_gradients")
NORMAL_LEVELS = (True, False)
PROMPT_INSTRUCTION = (
    "Analyze the anomalous log block using only the supplied context. "
    "Return a structured explanation with claims supported by exact event IDs and line IDs. "
    "If the evidence is insufficient, abstain explicitly. Do not invent evidence."
)


def _load_factor_config(config_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    factor = payload.get("contrastive_factor")
    if not isinstance(factor, dict):
        raise ValueError("configuration has no contrastive_factor")
    return factor


def registered_factor_cells(*, config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, Any]]:
    """Return the six registered normal-comparison/attribution cells."""

    factor = _load_factor_config(config_path)
    if factor.get("normal_comparison") != [True, False]:
        raise ValueError("contrastive_factor.normal_comparison must be [true, false]")
    if factor.get("attribution") != list(ATTRIBUTION_MODES):
        raise ValueError("contrastive_factor.attribution has an unexpected order or value")
    mapping = factor.get("condition_map")
    if not isinstance(mapping, dict):
        raise ValueError("contrastive_factor.condition_map must be an object")
    expected_keys = {f"{str(normal).lower()}|{mode}" for normal in NORMAL_LEVELS for mode in ATTRIBUTION_MODES}
    if set(mapping) != expected_keys:
        raise ValueError("contrastive_factor.condition_map must contain exactly six cells")
    return [
        {
            "normal_comparison": normal,
            "attribution": mode,
            "condition_id": mapping[f"{str(normal).lower()}|{mode}"],
        }
        for normal in NORMAL_LEVELS
        for mode in ATTRIBUTION_MODES
    ]


def build_contrastive_prompt_context(
    anomalous_record: Mapping[str, Any],
    detector_output: Mapping[str, Any],
    *,
    normal_comparison: bool,
    attribution: str,
    normal_record: Mapping[str, Any] | None = None,
    integrated_gradients: Mapping[str, Any] | None = None,
    shuffled_integrated_gradients: Mapping[str, Any] | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[str, dict[str, Any]]:
    """Build one registered factor cell and return ``(condition_id, payload)``."""

    if attribution not in ATTRIBUTION_MODES:
        raise ValueError(f"unsupported attribution mode: {attribution!r}")
    if not isinstance(detector_output, Mapping):
        raise ValueError("detector_output must be a mapping")
    cells = registered_factor_cells(config_path=config_path)
    cell = next(item for item in cells if item["normal_comparison"] == normal_comparison and item["attribution"] == attribution)
    if normal_comparison and normal_record is None:
        raise ValueError("normal_record is required when normal_comparison is true")
    if not normal_comparison and normal_record is not None:
        raise ValueError("normal_record must be omitted when normal_comparison is false")
    if attribution == "integrated_gradients" and integrated_gradients is None:
        raise ValueError("integrated_gradients is required for the Integrated Gradients cell")
    if attribution == "shuffled_integrated_gradients" and shuffled_integrated_gradients is None:
        raise ValueError("shuffled_integrated_gradients is required for the shuffled cell")
    if attribution == "none" and (integrated_gradients is not None or shuffled_integrated_gradients is not None):
        raise ValueError("attribution payloads must be omitted for the none cell")

    context: dict[str, Any] = {
        "anomalous_block_evidence": evidence_record_to_prompt_block(anomalous_record),
        "detector_output": dict(detector_output),
    }
    if normal_record is not None:
        context["normal_comparison_block"] = evidence_record_to_prompt_block(normal_record)
    if attribution == "integrated_gradients":
        context["integrated_gradients"] = dict(integrated_gradients or {})
    elif attribution == "shuffled_integrated_gradients":
        context["shuffled_integrated_gradients"] = dict(shuffled_integrated_gradients or {})
    return cell["condition_id"], build_prompt_payload(cell["condition_id"], context, config_path=config_path)


def render_contrastive_prompt(
    anomalous_record: Mapping[str, Any],
    detector_output: Mapping[str, Any],
    *,
    normal_comparison: bool,
    attribution: str,
    normal_record: Mapping[str, Any] | None = None,
    integrated_gradients: Mapping[str, Any] | None = None,
    shuffled_integrated_gradients: Mapping[str, Any] | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[str, str, dict[str, Any]]:
    """Render one factor cell and return ``(condition_id, template_id, text)``."""

    condition_id, payload = build_contrastive_prompt_context(
        anomalous_record,
        detector_output,
        normal_comparison=normal_comparison,
        attribution=attribution,
        normal_record=normal_record,
        integrated_gradients=integrated_gradients,
        shuffled_integrated_gradients=shuffled_integrated_gradients,
        config_path=config_path,
    )
    rendered = PROMPT_INSTRUCTION + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return condition_id, PROMPT_TEMPLATE_ID, rendered
