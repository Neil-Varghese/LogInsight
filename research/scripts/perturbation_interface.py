"""Shared interface for registered detector perturbation strategies.

Every perturbation strategy receives the same immutable input context and
returns a batch of perturbed model inputs plus per-input references.  Later
strategies (replacement, masking, resampling, and matched random controls) can
register with :func:`register_perturbation_strategy` without creating separate
calling conventions or output validation rules.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from seed_utils import make_numpy_rng
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.seed_utils import make_numpy_rng


PerturbationHandler = Callable[..., "PerturbationBatch"]


@dataclass(frozen=True)
class PerturbationContext:
    """Validated inputs and immutable settings shared by every strategy."""

    inputs: np.ndarray
    pad_id: int
    padding: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        inputs = np.asarray(self.inputs)
        _validate_inputs(inputs)
        if not isinstance(self.pad_id, (int, np.integer)):
            raise TypeError("pad_id must be an integer")
        _validate_padding(self.padding)
        copied = np.array(inputs, dtype=np.int32, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "inputs", copied)
        object.__setattr__(self, "pad_id", int(self.pad_id))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PerturbationBatch:
    """Raw strategy output before common validation and metadata decoration."""

    inputs: np.ndarray
    references: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[tuple[Mapping[str, Any], ...], ...] | None = None


@dataclass(frozen=True)
class PerturbationResult:
    """Validated output returned by :func:`apply_perturbation`."""

    strategy: str
    inputs: np.ndarray
    references: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]
    evidence: tuple[tuple[Mapping[str, Any], ...], ...] | None = None


_STRATEGIES: dict[str, PerturbationHandler] = {}


def _validate_inputs(inputs: np.ndarray) -> None:
    if inputs.ndim != 2:
        raise ValueError("perturbation inputs must be a two-dimensional array")
    if not np.issubdtype(inputs.dtype, np.integer):
        raise TypeError("perturbation inputs must contain integer token IDs")
    if inputs.shape[1] < 1:
        raise ValueError("perturbation inputs must have a positive sequence width")


def _validate_padding(padding: str) -> None:
    if padding not in {"pre", "post"}:
        raise ValueError("padding must be 'pre' or 'post'")


def register_perturbation_strategy(
    name: str,
    handler: PerturbationHandler,
    *,
    replace: bool = False,
) -> PerturbationHandler:
    """Register a named strategy handler.

    Registration fails on duplicate names unless ``replace=True`` is supplied,
    preventing an import-order mistake from silently changing an experiment.
    The handler must accept a :class:`PerturbationContext` as its first
    argument and return a :class:`PerturbationBatch`.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("strategy name must be a non-empty string")
    if not callable(handler):
        raise TypeError("strategy handler must be callable")
    normalized = name.strip()
    if normalized in _STRATEGIES and not replace:
        raise ValueError(f"perturbation strategy already registered: {normalized}")
    _STRATEGIES[normalized] = handler
    return handler


def available_perturbation_strategies() -> tuple[str, ...]:
    """Return registered strategy names in deterministic order."""

    return tuple(sorted(_STRATEGIES))


def get_perturbation_strategy(name: str) -> PerturbationHandler:
    """Return a registered strategy or raise a descriptive error."""

    try:
        return _STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(available_perturbation_strategies()) or "<none>"
        raise ValueError(f"unknown perturbation strategy {name!r}; available: {available}") from exc


def _validate_result(
    strategy: str,
    context: PerturbationContext,
    batch: PerturbationBatch,
) -> PerturbationResult:
    if not isinstance(batch, PerturbationBatch):
        raise TypeError(f"strategy {strategy!r} must return PerturbationBatch")
    perturbed = np.asarray(batch.inputs)
    _validate_inputs(perturbed)
    if perturbed.shape[1] != context.inputs.shape[1]:
        raise ValueError(
            f"strategy {strategy!r} changed sequence width from "
            f"{context.inputs.shape[1]} to {perturbed.shape[1]}"
        )
    if not isinstance(batch.references, tuple):
        references = tuple(batch.references)
    else:
        references = batch.references
    if len(references) != perturbed.shape[0]:
        raise ValueError(
            f"strategy {strategy!r} returned {perturbed.shape[0]} inputs but "
            f"{len(references)} references"
        )
    normalized_references: list[Mapping[str, Any]] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise TypeError(f"reference {index} from {strategy!r} is not a mapping")
        if "row_index" not in reference:
            raise ValueError(f"reference {index} from {strategy!r} is missing row_index")
        row_index = reference["row_index"]
        if not isinstance(row_index, (int, np.integer)) or not 0 <= int(row_index) < context.inputs.shape[0]:
            raise ValueError(f"reference {index} from {strategy!r} has an invalid row_index")
        copied = dict(reference)
        copied["row_index"] = int(row_index)
        normalized_references.append(copied)
    normalized_evidence: tuple[tuple[Mapping[str, Any], ...], ...] | None = None
    if batch.evidence is not None:
        if len(batch.evidence) != perturbed.shape[0]:
            raise ValueError(
                f"strategy {strategy!r} returned {perturbed.shape[0]} inputs but "
                f"{len(batch.evidence)} evidence rows"
            )
        normalized_evidence = tuple(
            tuple(copy.deepcopy(event) for event in row)
            for row in batch.evidence
        )
    copied_inputs = np.array(perturbed, dtype=np.int32, copy=True)
    copied_inputs.setflags(write=False)
    metadata = dict(context.metadata)
    metadata.update(dict(batch.metadata))
    sequence_length_changes: list[int] = []
    vocabulary_changes: list[bool] = []
    chronology_changes: list[bool] = []
    parser_statuses: list[str] = []
    parser_validity: list[bool | None] = []
    out_of_distribution: list[bool] = []
    out_of_distribution_reasons: list[list[str]] = []
    chronology_changed = bool(metadata.get("chronology_changed", False))
    for output_row, reference in zip(copied_inputs, normalized_references):
        source_row = context.inputs[int(reference["row_index"])]
        source_active = [int(value) for value in source_row if int(value) != context.pad_id]
        output_active = [int(value) for value in output_row if int(value) != context.pad_id]
        length_change = len(output_active) - len(source_active)
        vocabulary_change = set(output_active) != set(source_active)
        reasons: list[str] = []
        if length_change:
            reasons.append("sequence_length_change")
        if vocabulary_change:
            reasons.append("vocabulary_change")
        if chronology_changed:
            reasons.append("chronology_change")
        sequence_length_changes.append(length_change)
        vocabulary_changes.append(vocabulary_change)
        chronology_changes.append(chronology_changed)
        if normalized_evidence is None:
            parser_statuses.append("not_evaluated")
            parser_validity.append(None)
        else:
            parser_statuses.append("preserved_unreparsed")
            parser_validity.append(None)
        out_of_distribution.append(bool(reasons))
        out_of_distribution_reasons.append(reasons)
    metadata.update(
        {
            "metadata_schema_version": "1.0",
            "strategy": strategy,
            "source_shape": list(context.inputs.shape),
            "perturbed_count": int(copied_inputs.shape[0]),
            "sequence_length_change": sequence_length_changes,
            "vocabulary_change": vocabulary_changes,
            "chronology_change": chronology_changes,
            "parser_status": parser_statuses,
            "parser_valid": parser_validity,
            "out_of_distribution_indicator": out_of_distribution,
            "out_of_distribution_reasons": out_of_distribution_reasons,
        }
    )
    return PerturbationResult(
        strategy=strategy,
        inputs=copied_inputs,
        references=tuple(normalized_references),
        metadata=metadata,
        evidence=normalized_evidence,
    )


def apply_perturbation(
    strategy: str,
    inputs: np.ndarray,
    *,
    pad_id: int,
    padding: str,
    metadata: Mapping[str, Any] | None = None,
    **parameters: Any,
) -> PerturbationResult:
    """Apply any registered strategy through the common contract."""

    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must be a non-empty string")
    context = PerturbationContext(
        inputs=inputs,
        pad_id=pad_id,
        padding=padding,
        metadata={} if metadata is None else metadata,
    )
    handler = get_perturbation_strategy(strategy.strip())
    return _validate_result(strategy.strip(), context, handler(context, **parameters))


def _repad(tokens: Sequence[int], padded_length: int, pad_id: int, padding: str) -> np.ndarray:
    _validate_padding(padding)
    if len(tokens) > padded_length:
        tokens = tokens[-padded_length:] if padding == "pre" else tokens[:padded_length]
    result = np.full(padded_length, pad_id, dtype=np.int32)
    if tokens:
        values = np.asarray(tokens, dtype=np.int32)
        if padding == "pre":
            result[-len(values) :] = values
        else:
            result[: len(values)] = values
    return result


def _normalize_target_positions(
    target_positions: Sequence[Sequence[int]] | None,
    active_positions: Sequence[Sequence[int]],
) -> list[list[int]]:
    if target_positions is None:
        return [list(positions) for positions in active_positions]
    if len(target_positions) != len(active_positions):
        raise ValueError("target_positions must contain one position list per input row")
    normalized: list[list[int]] = []
    for row_index, (requested, active) in enumerate(zip(target_positions, active_positions)):
        values = [int(position) for position in requested]
        if len(values) != len(set(values)):
            raise ValueError(f"target_positions contains duplicates for row {row_index}")
        invalid = [position for position in values if position not in active]
        if invalid:
            raise ValueError(f"target_positions contains padding or out-of-range positions for row {row_index}: {invalid}")
        normalized.append(values)
    return normalized


def _single_event_deletion(
    context: PerturbationContext,
    *,
    target_positions: Sequence[Sequence[int]] | None = None,
) -> PerturbationBatch:
    """Delete selected active events and preserve the configured sequence width."""

    inputs = context.inputs
    active_positions = [
        [index for index, token_id in enumerate(row) if int(token_id) != context.pad_id]
        for row in inputs
    ]
    selected_positions = _normalize_target_positions(target_positions, active_positions)
    perturbed: list[np.ndarray] = []
    references: list[Mapping[str, Any]] = []
    for row_index, (row, positions) in enumerate(zip(inputs, selected_positions)):
        active = active_positions[row_index]
        tokens = [int(row[position]) for position in active]
        for input_position in positions:
            active_offset = active.index(input_position)
            event_id = tokens[active_offset]
            remaining = tokens[:active_offset] + tokens[active_offset + 1 :]
            perturbed.append(_repad(remaining, row.shape[0], context.pad_id, context.padding))
            references.append(
                {
                    "row_index": row_index,
                    "input_position": int(input_position),
                    "event_id": int(event_id),
                }
            )
    if perturbed:
        output = np.stack(perturbed).astype(np.int32, copy=False)
    else:
        output = np.empty((0, inputs.shape[1]), dtype=np.int32)
    return PerturbationBatch(
        inputs=output,
        references=tuple(references),
        metadata={"selection": "all_active" if target_positions is None else "explicit_positions"},
    )


def _delete_cited_events(
    context: PerturbationContext,
    *,
    target_positions: Sequence[Sequence[int]] | None = None,
) -> PerturbationBatch:
    """Delete all cited positions in each unit and return one output per unit."""

    if target_positions is None:
        raise ValueError("the deletion strategy requires target_positions for each input row")
    inputs = context.inputs
    active_positions = [
        [index for index, token_id in enumerate(row) if int(token_id) != context.pad_id]
        for row in inputs
    ]
    selected_positions = _normalize_target_positions(target_positions, active_positions)
    perturbed: list[np.ndarray] = []
    references: list[Mapping[str, Any]] = []
    for row_index, (row, positions) in enumerate(zip(inputs, selected_positions)):
        if positions:
            selected = set(positions)
            remaining = [int(token_id) for index, token_id in enumerate(row) if index not in selected]
            output = _repad(
                [token_id for token_id in remaining if token_id != context.pad_id],
                row.shape[0],
                context.pad_id,
                context.padding,
            )
        else:
            output = np.array(row, dtype=np.int32, copy=True)
        perturbed.append(output)
        references.append(
            {
                "row_index": row_index,
                "deleted_positions": [int(position) for position in positions],
                "deleted_event_ids": [int(row[position]) for position in positions],
            }
        )
    output = np.stack(perturbed).astype(np.int32, copy=False) if perturbed else np.empty((0, inputs.shape[1]), dtype=np.int32)
    return PerturbationBatch(
        inputs=output,
        references=tuple(references),
        metadata={"selection": "cited_positions", "one_output_per_input": True},
    )


def _normal_block_event_replacement(
    context: PerturbationContext,
    *,
    target_positions: Sequence[Sequence[int]] | None = None,
    normal_inputs: np.ndarray | None = None,
) -> PerturbationBatch:
    """Replace cited events with same-active-position events from a normal block."""

    if target_positions is None:
        raise ValueError("normal-block replacement requires target_positions for each input row")
    if normal_inputs is None:
        raise ValueError("normal-block replacement requires normal_inputs")
    normal = np.asarray(normal_inputs)
    _validate_inputs(normal)
    if normal.shape[0] != context.inputs.shape[0]:
        raise ValueError(
            "normal_inputs must contain one comparison row per input row"
        )
    active_positions = [
        [index for index, token_id in enumerate(row) if int(token_id) != context.pad_id]
        for row in context.inputs
    ]
    normal_active_positions = [
        [index for index, token_id in enumerate(row) if int(token_id) != context.pad_id]
        for row in normal
    ]
    selected_positions = _normalize_target_positions(target_positions, active_positions)
    perturbed: list[np.ndarray] = []
    references: list[Mapping[str, Any]] = []
    unavailable_count = 0
    for row_index, (row, normal_row, positions) in enumerate(
        zip(context.inputs, normal, selected_positions)
    ):
        output = np.array(row, dtype=np.int32, copy=True)
        normal_active = normal_active_positions[row_index]
        active = active_positions[row_index]
        row_replacements: list[dict[str, Any]] = []
        for input_position in positions:
            active_offset = active.index(input_position)
            replacement_position = (
                normal_active[active_offset] if active_offset < len(normal_active) else None
            )
            if replacement_position is None:
                unavailable_count += 1
                row_replacements.append(
                    {
                        "input_position": int(input_position),
                        "active_position": int(active_offset),
                        "original_event_id": int(row[input_position]),
                        "replacement_available": False,
                        "normal_input_position": None,
                        "replacement_event_id": None,
                    }
                )
                continue
            replacement_event_id = int(normal_row[replacement_position])
            output[input_position] = replacement_event_id
            row_replacements.append(
                {
                    "input_position": int(input_position),
                    "active_position": int(active_offset),
                    "original_event_id": int(row[input_position]),
                    "replacement_available": True,
                    "normal_input_position": int(replacement_position),
                    "replacement_event_id": replacement_event_id,
                }
            )
        perturbed.append(output)
        references.append(
            {
                "row_index": row_index,
                "replacements": row_replacements,
            }
        )
    output = np.stack(perturbed).astype(np.int32, copy=False) if perturbed else np.empty((0, context.inputs.shape[1]), dtype=np.int32)
    return PerturbationBatch(
        inputs=output,
        references=tuple(references),
        metadata={
            "alignment": "same_active_position",
            "sequence_length_preserved": True,
            "chronology_preserved": True,
            "replacement_unavailable_count": unavailable_count,
            "one_output_per_input": True,
        },
    )


def _unknown_token_replacement(
    context: PerturbationContext,
    *,
    target_positions: Sequence[Sequence[int]] | None = None,
    unknown_id: int | None = None,
) -> PerturbationBatch:
    """Replace cited active events with the detector's learned unknown token."""

    if target_positions is None:
        raise ValueError("unknown-token replacement requires target_positions for each input row")
    if unknown_id is None:
        raise ValueError("unknown-token replacement requires unknown_id")
    if not isinstance(unknown_id, (int, np.integer)):
        raise TypeError("unknown_id must be an integer")
    unknown_id = int(unknown_id)
    if unknown_id == context.pad_id:
        raise ValueError("unknown_id must differ from pad_id")
    active_positions = [
        [index for index, token_id in enumerate(row) if int(token_id) != context.pad_id]
        for row in context.inputs
    ]
    selected_positions = _normalize_target_positions(target_positions, active_positions)
    perturbed: list[np.ndarray] = []
    references: list[Mapping[str, Any]] = []
    for row_index, (row, positions) in enumerate(zip(context.inputs, selected_positions)):
        output = np.array(row, dtype=np.int32, copy=True)
        row_replacements: list[dict[str, Any]] = []
        for input_position in positions:
            original_event_id = int(row[input_position])
            output[input_position] = unknown_id
            row_replacements.append(
                {
                    "input_position": int(input_position),
                    "original_event_id": original_event_id,
                    "replacement_event_id": unknown_id,
                }
            )
        perturbed.append(output)
        references.append({"row_index": row_index, "replacements": row_replacements})
    output = np.stack(perturbed).astype(np.int32, copy=False) if perturbed else np.empty((0, context.inputs.shape[1]), dtype=np.int32)
    return PerturbationBatch(
        inputs=output,
        references=tuple(references),
        metadata={
            "replacement": "unknown_event_token",
            "unknown_id": unknown_id,
            "sequence_length_preserved": True,
            "one_output_per_input": True,
        },
    )


DEFAULT_PARAMETER_MASK_TOKEN = "<MASKED_PARAMETER>"


def _mask_event_record(
    event: Mapping[str, Any],
    *,
    mask_token: str,
) -> tuple[dict[str, Any], int]:
    """Mask parsed parameter values while retaining event-template metadata."""

    output = copy.deepcopy(dict(event))
    parameter_key = "ParameterList" if "ParameterList" in output else "parameter_list"
    raw_parameter_key = "ParameterListRaw" if "ParameterListRaw" in output else "parameter_list_raw"
    raw_line_key = "RawLine" if "RawLine" in output else "raw_line"
    parameters = output.get(parameter_key, [])
    if parameters is None:
        parameters = []
    if not isinstance(parameters, (list, tuple)):
        raise TypeError("event parameter list must be a list or tuple")
    original_parameters = [str(value) for value in parameters]
    masked_parameters = [mask_token for _ in original_parameters]
    if parameter_key in output:
        output[parameter_key] = masked_parameters
    if raw_parameter_key in output:
        output[raw_parameter_key] = repr(masked_parameters)
    if raw_line_key in output and isinstance(output[raw_line_key], str):
        raw_line = output[raw_line_key]
        for original in original_parameters:
            if original:
                raw_line = raw_line.replace(original, mask_token, 1)
        output[raw_line_key] = raw_line
    return output, len(original_parameters)


def _parameter_masking(
    context: PerturbationContext,
    *,
    target_positions: Sequence[Sequence[int]] | None = None,
    event_evidence: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    mask_token: str = DEFAULT_PARAMETER_MASK_TOKEN,
) -> PerturbationBatch:
    """Mask dynamic parameters for cited events without changing event IDs."""

    if target_positions is None:
        raise ValueError("parameter masking requires target_positions for each input row")
    if event_evidence is None:
        raise ValueError("parameter masking requires event_evidence")
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError("mask_token must be a non-empty string")
    if len(event_evidence) != context.inputs.shape[0]:
        raise ValueError("event_evidence must contain one evidence row per input row")
    active_positions = [
        [index for index, token_id in enumerate(row) if int(token_id) != context.pad_id]
        for row in context.inputs
    ]
    selected_positions = _normalize_target_positions(target_positions, active_positions)
    perturbed: list[np.ndarray] = []
    perturbed_evidence: list[tuple[Mapping[str, Any], ...]] = []
    references: list[Mapping[str, Any]] = []
    masked_parameter_count = 0
    for row_index, (row, evidence_row, positions) in enumerate(
        zip(context.inputs, event_evidence, selected_positions)
    ):
        if any(not isinstance(event, Mapping) for event in evidence_row):
            raise TypeError("every event_evidence item must be a mapping")
        evidence_copy = [copy.deepcopy(dict(event)) for event in evidence_row]
        active = active_positions[row_index]
        by_position: dict[int, int] = {}
        for evidence_index, event in enumerate(evidence_copy):
            event_position = event.get("Position", event.get("position", evidence_index))
            if not isinstance(event_position, (int, np.integer)):
                raise TypeError("event evidence positions must be integers")
            by_position[int(event_position)] = evidence_index
        row_masks: list[dict[str, Any]] = []
        for input_position in positions:
            active_position = active.index(input_position)
            evidence_index = by_position.get(active_position)
            if evidence_index is None and active_position < len(evidence_copy):
                evidence_index = active_position
            if evidence_index is None:
                row_masks.append(
                    {
                        "input_position": int(input_position),
                        "active_position": int(active_position),
                        "masked": False,
                        "parameter_count": 0,
                    }
                )
                continue
            masked_event, parameter_count = _mask_event_record(
                evidence_copy[evidence_index],
                mask_token=mask_token,
            )
            evidence_copy[evidence_index] = masked_event
            masked_parameter_count += parameter_count
            row_masks.append(
                {
                    "input_position": int(input_position),
                    "active_position": int(active_position),
                    "masked": True,
                    "parameter_count": parameter_count,
                }
            )
        perturbed.append(np.array(row, dtype=np.int32, copy=True))
        perturbed_evidence.append(tuple(evidence_copy))
        references.append({"row_index": row_index, "masked_events": row_masks})
    output = np.stack(perturbed).astype(np.int32, copy=False) if perturbed else np.empty((0, context.inputs.shape[1]), dtype=np.int32)
    return PerturbationBatch(
        inputs=output,
        references=tuple(references),
        metadata={
            "mask_token": mask_token,
            "masked_parameter_count": masked_parameter_count,
            "event_templates_preserved": True,
            "sequence_length_preserved": True,
            "vocabulary_change": False,
            "one_output_per_input": True,
        },
        evidence=tuple(perturbed_evidence),
    )


def _normal_block_event_resampling(
    context: PerturbationContext,
    *,
    target_positions: Sequence[Sequence[int]] | None = None,
    normal_inputs: np.ndarray | None = None,
    seed: int | None = None,
) -> PerturbationBatch:
    """Sample normal-block events into cited positions with a stable seed."""

    if target_positions is None:
        raise ValueError("normal-block resampling requires target_positions for each input row")
    if normal_inputs is None:
        raise ValueError("normal-block resampling requires normal_inputs")
    if seed is None:
        raise ValueError("normal-block resampling requires seed")
    normal = np.asarray(normal_inputs)
    _validate_inputs(normal)
    if normal.shape[0] != context.inputs.shape[0]:
        raise ValueError("normal_inputs must contain one comparison row per input row")
    active_positions = [
        [index for index, token_id in enumerate(row) if int(token_id) != context.pad_id]
        for row in context.inputs
    ]
    normal_active_positions = [
        [index for index, token_id in enumerate(row) if int(token_id) != context.pad_id]
        for row in normal
    ]
    selected_positions = _normalize_target_positions(target_positions, active_positions)
    perturbed: list[np.ndarray] = []
    references: list[Mapping[str, Any]] = []
    unavailable_count = 0
    with_replacement_count = 0
    for row_index, (row, normal_row, positions) in enumerate(
        zip(context.inputs, normal, selected_positions)
    ):
        output = np.array(row, dtype=np.int32, copy=True)
        normal_active = normal_active_positions[row_index]
        row_replacements: list[dict[str, Any]] = []
        if positions and normal_active:
            rng = make_numpy_rng(
                int(seed),
                "normal_block_event_resampling",
                context.metadata.get("architecture"),
                context.metadata.get("seed"),
                context.metadata.get("split"),
                context.metadata.get("unit_id", row_index),
            )
            sampling_with_replacement = len(positions) > len(normal_active)
            donor_indices = rng.choice(
                len(normal_active),
                size=len(positions),
                replace=sampling_with_replacement,
            )
        else:
            sampling_with_replacement = False
            donor_indices = []
        if sampling_with_replacement:
            with_replacement_count += 1
        for target_index, input_position in enumerate(positions):
            original_event_id = int(row[input_position])
            if not normal_active:
                unavailable_count += 1
                row_replacements.append(
                    {
                        "input_position": int(input_position),
                        "original_event_id": original_event_id,
                        "replacement_available": False,
                        "donor_active_position": None,
                        "replacement_event_id": None,
                        "sampling_with_replacement": False,
                    }
                )
                continue
            donor_active_index = int(donor_indices[target_index])
            donor_position = int(normal_active[donor_active_index])
            replacement_event_id = int(normal_row[donor_position])
            output[input_position] = replacement_event_id
            row_replacements.append(
                {
                    "input_position": int(input_position),
                    "original_event_id": original_event_id,
                    "replacement_available": True,
                    "donor_active_position": donor_position,
                    "replacement_event_id": replacement_event_id,
                    "sampling_with_replacement": sampling_with_replacement,
                }
            )
        perturbed.append(output)
        references.append({"row_index": row_index, "replacements": row_replacements})
    output = np.stack(perturbed).astype(np.int32, copy=False) if perturbed else np.empty((0, context.inputs.shape[1]), dtype=np.int32)
    return PerturbationBatch(
        inputs=output,
        references=tuple(references),
        metadata={
            "sampling": "normal_block_events",
            "seed": int(seed),
            "replacement_unavailable_count": unavailable_count,
            "rows_sampled_with_replacement": with_replacement_count,
            "sequence_length_preserved": True,
            "chronology_preserved": True,
            "one_output_per_input": True,
        },
    )


register_perturbation_strategy("single_event_deletion", _single_event_deletion)
register_perturbation_strategy("deletion", _delete_cited_events)
register_perturbation_strategy(
    "normal_block_event_replacement",
    _normal_block_event_replacement,
)
register_perturbation_strategy(
    "unknown_token_replacement",
    _unknown_token_replacement,
)
register_perturbation_strategy("parameter_masking", _parameter_masking)
register_perturbation_strategy(
    "normal_block_event_resampling",
    _normal_block_event_resampling,
)


def delete_cited_events(
    inputs: np.ndarray,
    *,
    cited_positions: Sequence[Sequence[int]],
    pad_id: int,
    padding: str,
    metadata: Mapping[str, Any] | None = None,
) -> PerturbationResult:
    """Delete the cited event positions through the registered deletion strategy."""

    return apply_perturbation(
        "deletion",
        inputs,
        pad_id=pad_id,
        padding=padding,
        metadata=metadata,
        target_positions=cited_positions,
    )


def replace_with_normal_events(
    inputs: np.ndarray,
    normal_inputs: np.ndarray,
    *,
    cited_positions: Sequence[Sequence[int]],
    pad_id: int,
    padding: str,
    metadata: Mapping[str, Any] | None = None,
) -> PerturbationResult:
    """Replace cited positions using active-position-aligned normal events."""

    return apply_perturbation(
        "normal_block_event_replacement",
        inputs,
        pad_id=pad_id,
        padding=padding,
        metadata=metadata,
        target_positions=cited_positions,
        normal_inputs=normal_inputs,
    )


def replace_with_unknown_token(
    inputs: np.ndarray,
    *,
    cited_positions: Sequence[Sequence[int]],
    unknown_id: int,
    pad_id: int,
    padding: str,
    metadata: Mapping[str, Any] | None = None,
) -> PerturbationResult:
    """Replace cited active positions with the learned unknown-event token."""

    return apply_perturbation(
        "unknown_token_replacement",
        inputs,
        pad_id=pad_id,
        padding=padding,
        metadata=metadata,
        target_positions=cited_positions,
        unknown_id=unknown_id,
    )


def mask_event_parameters(
    inputs: np.ndarray,
    event_evidence: Sequence[Sequence[Mapping[str, Any]]],
    *,
    cited_positions: Sequence[Sequence[int]],
    pad_id: int,
    padding: str,
    mask_token: str = DEFAULT_PARAMETER_MASK_TOKEN,
    metadata: Mapping[str, Any] | None = None,
) -> PerturbationResult:
    """Mask cited event parameters while retaining event-template tokens."""

    return apply_perturbation(
        "parameter_masking",
        inputs,
        pad_id=pad_id,
        padding=padding,
        metadata=metadata,
        target_positions=cited_positions,
        event_evidence=event_evidence,
        mask_token=mask_token,
    )


def sample_matched_random_positions(
    inputs: np.ndarray,
    cited_positions: Sequence[Sequence[int]],
    *,
    pad_id: int,
    seed: int,
    strategy: str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[list[list[int]], int]:
    """Sample random active positions matching each cited-event budget."""

    context = PerturbationContext(
        inputs=inputs,
        pad_id=pad_id,
        padding="post",
        metadata={} if metadata is None else metadata,
    )
    active_positions = [
        [index for index, token_id in enumerate(row) if int(token_id) != context.pad_id]
        for row in context.inputs
    ]
    normalized_citations = _normalize_target_positions(cited_positions, active_positions)
    random_positions: list[list[int]] = []
    overlap_count = 0
    for row_index, (active, cited) in enumerate(zip(active_positions, normalized_citations)):
        if not cited:
            random_positions.append([])
            continue
        rng = make_numpy_rng(
            int(seed),
            "matched_random_perturbation",
            strategy,
            context.metadata.get("architecture"),
            context.metadata.get("seed"),
            context.metadata.get("split"),
            context.metadata.get("unit_id", row_index),
        )
        sampled = sorted(int(position) for position in rng.choice(len(active), size=len(cited), replace=False))
        random_positions.append([active[index] for index in sampled])
        overlap_count += len(set(cited).intersection(random_positions[-1]))
    return random_positions, overlap_count


def apply_matched_random_perturbation(
    strategy: str,
    inputs: np.ndarray,
    *,
    cited_positions: Sequence[Sequence[int]],
    seed: int,
    pad_id: int,
    padding: str,
    metadata: Mapping[str, Any] | None = None,
    **parameters: Any,
) -> PerturbationResult:
    """Apply a strategy to random positions matched to the cited-event budget."""

    if "target_positions" in parameters:
        raise ValueError("matched random perturbation derives target_positions from cited_positions")
    base_metadata = {} if metadata is None else dict(metadata)
    random_positions, overlap_count = sample_matched_random_positions(
        inputs,
        cited_positions,
        pad_id=pad_id,
        seed=seed,
        strategy=strategy,
        metadata=base_metadata,
    )
    base_metadata.update(
        {
            "control": "matched_random",
            "matched_random_seed": int(seed),
            "cited_positions": [list(map(int, row)) for row in cited_positions],
            "random_positions": random_positions,
            "random_citation_overlap_count": overlap_count,
        }
    )
    strategy_parameters = dict(parameters)
    if strategy == "normal_block_event_resampling":
        strategy_parameters.setdefault("seed", int(seed))
    result = apply_perturbation(
        strategy,
        inputs,
        pad_id=pad_id,
        padding=padding,
        metadata=base_metadata,
        target_positions=random_positions,
        **strategy_parameters,
    )
    return result


def resample_from_normal_events(
    inputs: np.ndarray,
    normal_inputs: np.ndarray,
    *,
    cited_positions: Sequence[Sequence[int]],
    seed: int,
    pad_id: int,
    padding: str,
    metadata: Mapping[str, Any] | None = None,
) -> PerturbationResult:
    """Sample normal-block events into cited positions deterministically."""

    return apply_perturbation(
        "normal_block_event_resampling",
        inputs,
        pad_id=pad_id,
        padding=padding,
        metadata=metadata,
        target_positions=cited_positions,
        normal_inputs=normal_inputs,
        seed=seed,
    )


__all__ = [
    "PerturbationBatch",
    "PerturbationContext",
    "PerturbationResult",
    "apply_perturbation",
    "apply_matched_random_perturbation",
    "available_perturbation_strategies",
    "delete_cited_events",
    "get_perturbation_strategy",
    "register_perturbation_strategy",
    "replace_with_normal_events",
    "replace_with_unknown_token",
    "resample_from_normal_events",
    "sample_matched_random_positions",
    "mask_event_parameters",
]
