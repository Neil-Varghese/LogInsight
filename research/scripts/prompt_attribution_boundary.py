"""Enforce the registered attribution boundary before prompt rendering.

Integrated Gradients is the only attribution signal allowed in the registered
unblinded and shuffled-attribution conditions. Occlusion and leave-one-event-
out values are verification-only and must never be silently copied into an
LLM prompt. This module is the strict, condition-aware boundary for future
research prompt consumers: it rejects forbidden fields and unsupported fields
before any prompt text is rendered.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiments" / "experiment_config.json"

_FORBIDDEN_KEY_FRAGMENTS = ("occlusion", "leaveoneeventout")
_FORBIDDEN_KEYS = {
    "verificationonly",
    "heldoutprobability",
    "heldoutevent",
    "leaveoneoutprobability",
    "leaveoneouteventscore",
}
_FORBIDDEN_METHODS = {
    "eventocclusion",
    "occlusion",
    "leaveoneeventout",
    "leaveoneout",
}
_CONDITION_FIELD_NAMES = {
    "anomalous_block_evidence",
    "normal_comparison_block",
    "detector_output",
    "integrated_gradients",
    "shuffled_integrated_gradients",
}


class PromptAttributionBoundaryError(ValueError):
    """Raised when a prompt context violates the registered attribution rules."""


def _normalize_identifier(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _path_text(path: tuple[str, ...]) -> str:
    result = "$"
    for component in path:
        result += f"[{component!r}]"
    return result


def _forbidden_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            normalized_key = _normalize_identifier(key_text)
            child_path = path + (key_text,)
            if (
                normalized_key in _FORBIDDEN_KEYS
                or any(fragment in normalized_key for fragment in _FORBIDDEN_KEY_FRAGMENTS)
                or (normalized_key == "promptvisible" and child is False)
            ):
                findings.append(_path_text(child_path))
            if normalized_key == "method" and _normalize_identifier(child) in _FORBIDDEN_METHODS:
                findings.append(_path_text(child_path))
            findings.extend(_forbidden_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(_forbidden_paths(child, path + (str(index),)))
    return findings


def _nested_condition_field_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Find condition payload fields nested under the wrong root field.

    A detector-output mapping is allowed as a root field, but it must not be
    used as a container for attribution or normal-comparison data. Checking
    only top-level keys would allow that accidental leak.
    """

    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            normalized_key = _normalize_identifier(key_text)
            child_path = path + (key_text,)
            if path and normalized_key in {_normalize_identifier(name) for name in _CONDITION_FIELD_NAMES}:
                findings.append(_path_text(child_path))
            findings.extend(_nested_condition_field_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(_nested_condition_field_paths(child, path + (str(index),)))
    return findings


def load_prompt_condition(
    condition_id: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load one registered prompt condition from the central configuration."""

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for condition in payload.get("prompt_conditions", []):
        if condition.get("id") == condition_id:
            return condition
    raise PromptAttributionBoundaryError(f"Unknown registered prompt condition: {condition_id!r}")


def build_prompt_payload(
    condition_id: str,
    context: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Return only registered visible fields after strict boundary checks.

    The input is rejected, rather than filtered, when it contains occlusion,
    leave-one-event-out, verification-only metadata, or any other field not
    registered as visible for the selected condition.
    """

    if not isinstance(context, Mapping):
        raise PromptAttributionBoundaryError("prompt context must be a mapping")
    condition = load_prompt_condition(condition_id, config_path)
    forbidden_paths = _forbidden_paths(context)
    forbidden_paths.extend(_nested_condition_field_paths(context))
    if forbidden_paths:
        joined = ", ".join(sorted(set(forbidden_paths)))
        raise PromptAttributionBoundaryError(
            f"verification attribution fields are forbidden in prompt context: {joined}"
        )
    visible_fields = condition.get("prompt_visible", [])
    forbidden_fields = set(condition.get("prompt_forbidden", []))
    unsupported = sorted(set(context) - set(visible_fields))
    if unsupported:
        forbidden = sorted(set(unsupported) & forbidden_fields)
        if forbidden:
            raise PromptAttributionBoundaryError(
                f"fields forbidden for condition {condition_id!r}: {', '.join(forbidden)}"
            )
        raise PromptAttributionBoundaryError(
            f"fields are not registered for condition {condition_id!r}: {', '.join(unsupported)}"
        )
    result = {field: deepcopy(context[field]) for field in visible_fields if field in context}
    try:
        json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise PromptAttributionBoundaryError("prompt context is not JSON serializable") from error
    return result


def render_prompt_payload(
    condition_id: str,
    context: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> str:
    """Render a boundary-checked context as deterministic JSON prompt text."""

    payload = build_prompt_payload(condition_id, context, config_path=config_path)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
