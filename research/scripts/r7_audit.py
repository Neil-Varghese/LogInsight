"""Run the independent reviewer audit required by R7.

The audit is deliberately narrow. It checks the registered prompt boundary
with golden payloads, validates an attribution-free annotation export, checks
required fields in a representative experiment-ledger record, recomputes a
small detector-metric set from the stored test predictions, and independently
recomputes one serialized Integrated-Gradients record from the LSTM checkpoint.
It does not generate LLM outputs or change any registered protocol file.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from annotation_packet import build_annotation_packet, validate_annotation_export
    from blinded_prompt import build_blinded_prompt_context
    from contrastive_prompt import render_contrastive_prompt
    from detector_interface import DetectorContract, KerasDetectorAdapter
    from explanation_schema import validate_explanation_record
    from integrated_gradients_lstm import integrated_gradients_batch
    from prompt_attribution_boundary import build_prompt_payload, render_prompt_payload
    from train_loginsight_lstm import build_model, load_run_config, load_split, _resolve_path
    from event_vocabulary import load_vocabulary
    from seed_utils import set_global_seed
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.annotation_packet import build_annotation_packet, validate_annotation_export
    from research.scripts.blinded_prompt import build_blinded_prompt_context
    from research.scripts.contrastive_prompt import render_contrastive_prompt
    from research.scripts.detector_interface import DetectorContract, KerasDetectorAdapter
    from research.scripts.explanation_schema import validate_explanation_record
    from research.scripts.integrated_gradients_lstm import integrated_gradients_batch
    from research.scripts.prompt_attribution_boundary import build_prompt_payload, render_prompt_payload
    from research.scripts.train_loginsight_lstm import build_model, load_run_config, load_split, _resolve_path
    from research.scripts.event_vocabulary import load_vocabulary
    from research.scripts.seed_utils import set_global_seed


SCRIPT_SCHEMA_VERSION = "1.0"
PROMPT_CONDITIONS = (
    "blinded",
    "unblinded",
    "shuffled_integrated_gradients",
    "detector_only",
    "unblinded_no_normal",
    "shuffled_integrated_gradients_no_normal",
    "attribution_only",
    "random_event",
    "no_detector_context",
)
PROMPT_FORBIDDEN_TERMS = ("occlusion", "leave_one_event_out", "verification_only")
GOLDEN_PROMPT_SHA256 = {
    "blinded": "21c8d5ec9714f9079b65dafc74fd1dc148fa0e3e333e32445c3bfd079be8ebfa",
    "unblinded": "6bcbf3d3fe57918898fd1360144804af226a15b076bc7bf2ed4c03e99aab3d3c",
    "shuffled_integrated_gradients": "009586a6cbc07d668426c7574d8529eae1b3bfabd484345726a9b991b31604ab",
    "detector_only": "1e56f7a57d8702626d78f673036722e99de3457597ef8756cab1fcae669289b2",
    "unblinded_no_normal": "9f4d53610fc95763186df3f12c80ce8aeaf9da3c36db148aa293a9cc113373bc",
    "shuffled_integrated_gradients_no_normal": "e97300613779a79f389d5ebf127f3a2b43369628a6d00830c139a060bd12c58e",
    "random_event": "21c8d5ec9714f9079b65dafc74fd1dc148fa0e3e333e32445c3bfd079be8ebfa",
    "no_detector_context": "4075f56c5e6f3d4c413bcf240bf22b658a634794df9fc976beb0bea3149054ab",
}


def _canonical_prompt_context() -> dict[str, Any]:
    return {
        "anomalous_block_evidence": {"events": [{"event_id": "E0001", "line_id": "L0001", "position": 0}]},
        "normal_comparison_block": {"events": [{"event_id": "NE0001", "line_id": "NL0001", "position": 0}]},
        "detector_output": {"probability": 0.9, "prediction": 1},
        "integrated_gradients": {"ranked_event_ids": ["E0001"], "scores": [0.8]},
        "shuffled_integrated_gradients": {"ranked_event_ids": ["E0001"], "scores": [0.8]},
    }


def audit_prompt_boundary(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    conditions = {item["id"]: item for item in config["prompt_conditions"]}
    canonical = _canonical_prompt_context()
    records: list[dict[str, Any]] = []
    for condition_id in PROMPT_CONDITIONS:
        condition = conditions[condition_id]
        visible = list(condition.get("prompt_visible", []))
        if condition_id == "attribution_only":
            if visible:
                raise AssertionError("attribution_only must not expose prompt fields")
            records.append({"condition": condition_id, "status": "not_applicable", "visible_fields": []})
            continue
        context = {field: canonical[field] for field in visible}
        payload = build_prompt_payload(condition_id, context, config_path=config_path)
        rendered = render_prompt_payload(condition_id, context, config_path=config_path)
        if set(payload) != set(visible):
            raise AssertionError(f"{condition_id}: payload fields differ from registered allowlist")
        lowered = rendered.lower()
        leaked = [term for term in PROMPT_FORBIDDEN_TERMS if term in lowered]
        if leaked:
            raise AssertionError(f"{condition_id}: forbidden terms leaked into rendered payload: {leaked}")
        rendered_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        expected_sha256 = GOLDEN_PROMPT_SHA256[condition_id]
        if rendered_sha256 != expected_sha256:
            raise AssertionError(
                f"{condition_id}: rendered prompt changed; expected golden hash "
                f"{expected_sha256}, got {rendered_sha256}"
            )
        records.append({
            "condition": condition_id,
            "status": "passed",
            "visible_fields": visible,
            "rendered_sha256": rendered_sha256,
        })
    return {"status": "passed", "condition_count": len(records), "conditions": records}


def _annotation_source_record() -> dict[str, Any]:
    return {
        "dataset": "HDFS",
        "unit_id": "blk_-r7-fixture",
        "detector_id": "loginsight_lstm",
        "condition": "unblinded",
        "events": [
            {
                "EventId": "e3df2680",
                "LineId": 9512780,
                "EventTemplate": "Received block <*> from <*>",
                "ParameterList": ["blk_-r7-fixture", "/10.1.2.3:50010"],
                "RawLine": "Received block blk_-r7-fixture from /10.1.2.3:50010",
            },
            {
                "EventId": "d63ef163",
                "LineId": 10986891,
                "EventTemplate": "BLOCK* NameSystem.delete: <*>",
                "ParameterList": ["blk_-r7-fixture"],
                "RawLine": "BLOCK* NameSystem.delete: blk_-r7-fixture",
            },
        ],
    }


def audit_annotation_export() -> dict[str, Any]:
    packet = build_annotation_packet(_annotation_source_record())
    export = {
        "schema_version": packet["schema_version"],
        "guideline_version": packet["guideline_version"],
        "packet_id": packet["packet_id"],
        "unit_alias": packet["unit_alias"],
        "events": [
            {
                "event_position": event["event_position"],
                "event_id": event["event_id"],
                "line_id": event["line_id"],
                "relevance": "uncertain",
                "role": None,
                "reason": None,
            }
            for event in packet["events"]
        ],
        "contrastive_assessment": {
            "meaningful_difference": "uncertain",
            "difference_positions": [],
            "reason": None,
        },
    }
    validate_annotation_export(export)
    serialized = json.dumps(export, sort_keys=True).lower()
    forbidden = [term for term in ("attribution", "occlusion", "leave_one", "detector_score", "probability", "condition", "model", "seed") if term in serialized]
    if forbidden:
        raise AssertionError(f"annotation export contains forbidden terms: {forbidden}")
    if "blk_-r7-fixture" in serialized or "10.1.2.3" in serialized:
        raise AssertionError("annotation export contains source identifiers")
    return {
        "status": "passed",
        "packet_id": packet["packet_id"],
        "event_count": len(export["events"]),
        "annotation_eligible": packet["context"]["annotation_eligible"],
        "forbidden_fields_checked": 8,
    }


def _resolve_schema_ref(schema: Mapping[str, Any], node: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = node.get("$ref")
    if not reference:
        return node
    if not reference.startswith("#/$defs/"):
        raise ValueError(f"unsupported schema reference: {reference}")
    return schema["$defs"][reference.split("/")[-1]]


def _validate_required_fields(value: Any, node: Mapping[str, Any], schema: Mapping[str, Any], path: str = "$", count: list[int] | None = None) -> None:
    count = count if count is not None else [0]
    node = _resolve_schema_ref(schema, node)
    if node.get("type") == "object":
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be an object")
        required = node.get("required", [])
        count[0] += len(required)
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        if node.get("additionalProperties") is False:
            allowed = set(node.get("properties", {}))
            extra = sorted(set(value) - allowed)
            if extra:
                raise ValueError(f"{path} contains unsupported fields: {extra}")
        for key, child_schema in node.get("properties", {}).items():
            if key in value:
                _validate_required_fields(value[key], child_schema, schema, f"{path}.{key}", count)
    elif node.get("type") == "array" and isinstance(value, list):
        for index, item in enumerate(value):
            _validate_required_fields(item, node.get("items", {}), schema, f"{path}[{index}]", count)


def _ledger_fixture() -> dict[str, Any]:
    event = {"position": 0, "event_id": "E0001", "event_template": "Failure", "raw_line": "failure"}
    return {
        "schema_version": "1.0", "record_id": "R7:fixture", "run_id": "r7", "created_at_utc": "2026-10-26T00:00:00Z",
        "code_version": {"git_commit": "fixture", "config_version": "1.0"}, "environment_id": "r7-fixture",
        "dataset": {"dataset_id": "HDFS", "split_id": "test", "prediction_unit": "BlockId", "unit_id": "blk-r7", "label": 1},
        "parser": {
            "name": "Drain",
            "version": "1.0",
            "configuration": {
                "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
                "depth": 4,
                "similarity_threshold": 0.5,
            },
            "fit_split": "train",
            "unknown_event_policy": "__UNKNOWN_EVENT__",
            "status": "valid",
        },
        "input": {"events": [event], "event_sequence": [1], "sequence_length": 1},
        "detector": {"detector_id": "loginsight_lstm", "model_version": "fixture", "checkpoint": "fixture", "seed": 13, "anomaly_score": 0.9, "predicted_label": 1, "threshold": 0.5, "threshold_source": "validation"},
        "attribution": {
            "integrated_gradients": {"method": "integrated_gradients", "prompt_visible": True, "event_scores": [{"event_id": "E0001", "score": 0.8}], "ranked_event_ids": ["E0001"]},
            "occlusion": {"method": "event_occlusion", "prompt_visible": False, "event_scores": [{"event_id": "E0001", "score": 0.1}], "ranked_event_ids": ["E0001"]},
            "leave_one_event_out": {"method": "leave_one_event_out", "prompt_visible": False, "event_scores": [{"event_id": "E0001", "score": 0.1}], "ranked_event_ids": ["E0001"]},
        },
        "retrieval": {"enabled": True, "source_split": "train", "normal_unit_id": "blk-normal", "similarity_score": 0.9},
        "llm": {"condition_id": "blinded", "provider": "fixture", "model_name": "fixture", "model_version": "1", "prompt_template_id": "fixture", "prompt_template_version": "1", "exact_prompt": "{}", "raw_response": "{}", "status": "complete"},
        "explanation": {"claims": [{"claim_id": "C1", "text": "Failure", "cited_event_ids": ["E0001"], "cited_line_ids": ["L0001"]}], "cited_event_ids": ["E0001"], "cited_line_ids": ["L0001"], "abstention_status": "none", "verification_status": "pending"},
        "perturbations": [],
        "outcomes": {
            "human_evidence_recovery": {"annotated": False, "relevant_event_ids": [], "precision": None, "recall": None, "f1": None},
            "claim_support": {"annotated": False, "total_claims": 0, "supported_claims": 0, "unsupported_claims": 0, "uncertain_claims": 0, "support_rate": None, "hallucination_rate": None},
            "detector_faithfulness": {"evaluated": False, "predicted_direction_accuracy": None, "mean_score_delta": None, "counterfactual_validity": None},
            "perturbation_consistency": {"evaluated": False, "claims_tested": 0, "claims_consistent_across_at_least_two_strategies": 0, "consistency_rate": None},
        },
        "execution": {"detector_latency_ms": 1.0, "explanation_latency_ms": None, "verification_latency_ms": None, "llm_calls": 0, "detector_perturbation_inference_calls": 0},
        "quality": {"record_status": "complete", "verification_status": "pending", "exclusion_status": "included", "validation_errors": []},
    }


def audit_ledger(schema_path: Path) -> dict[str, Any]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    record = _ledger_fixture()
    count = [0]
    _validate_required_fields(record, schema, schema, count=count)
    broken = copy.deepcopy(record)
    del broken["quality"]
    try:
        _validate_required_fields(broken, schema, schema)
    except ValueError:
        rejected_missing_field = True
    else:  # pragma: no cover - audit failure
        rejected_missing_field = False
    if not rejected_missing_field:
        raise AssertionError("ledger validator did not reject a missing required top-level field")
    return {"status": "passed", "top_level_required_fields": len(schema["required"]), "nested_required_fields_checked": count[0]}


def _open_jsonl_gz(path: Path):
    return gzip.open(path, "rt", encoding="utf-8")


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    positive_count = int(np.sum(sorted_labels == 1))
    if positive_count == 0:
        return 0.0
    cumulative = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, labels.size + 1)
    return float(np.sum((cumulative / ranks)[sorted_labels == 1]) / positive_count)


def audit_metrics(predictions_path: Path, baseline_path: Path) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    with _open_jsonl_gz(predictions_path) as handle:
        for line in handle:
            record = json.loads(line)
            labels.append(int(record["label_id"]))
            predictions.append(int(record["prediction"]))
            probabilities.append(float(record["probability"]))
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(predictions, dtype=np.int8)
    prob = np.asarray(probabilities, dtype=np.float64)
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    recomputed = {
        "count": int(y.size), "accuracy": (tp + tn) / y.size, "precision": precision,
        "recall": recall, "f1": f1, "false_positives": fp, "false_negatives": fn,
        "pr_auc": _average_precision(y, prob),
    }
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics_by_split"]["test"]["thresholded"]
    comparisons = {}
    for key, value in recomputed.items():
        expected = float(baseline[key])
        comparisons[key] = {"recomputed": value, "stored": expected, "matches": math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10)}
    if not all(item["matches"] for item in comparisons.values()):
        raise AssertionError(f"independent metric recomputation disagrees: {comparisons}")
    return {"status": "passed", "source_records": int(y.size), "comparisons": comparisons}


def _read_first_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    with _open_jsonl_gz(path) as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= limit:
                    break
    return records


def audit_serialized_attributions(path: Path, limit: int = 8) -> dict[str, Any]:
    records = _read_first_records(path, limit)
    if not records:
        raise ValueError("attribution artifact is empty")
    max_completeness_error = 0.0
    rank_checks = 0
    for record in records:
        events = record["events"]
        active = [event for event in events if not event["is_padding"]]
        ranked = sorted(active, key=lambda event: (-abs(float(event["attribution"])), int(event["input_position"])))
        expected_ranks = {int(event["input_position"]): rank for rank, event in enumerate(ranked, start=1)}
        for event in active:
            if event["rank"] != expected_ranks[int(event["input_position"])] or not math.isclose(float(event["absolute_attribution"]), abs(float(event["attribution"])), abs_tol=1e-12):
                raise AssertionError(f"stored attribution rank/alignment mismatch for {record['unit_id']}")
            rank_checks += 1
        error = abs(sum(float(event["attribution"]) for event in events) - float(record["output_delta"]))
        max_completeness_error = max(max_completeness_error, error)
        # The serialized attribution sum is accumulated in float32 before it is
        # written; summing the JSON float values in Python can differ by a few
        # ulps while still representing the same stored completeness check.
        if not math.isclose(error, float(record["completeness_error"]), rel_tol=1e-5, abs_tol=1e-6):
            raise AssertionError(f"stored completeness error mismatch for {record['unit_id']}")
    return {"status": "passed", "records_checked": len(records), "active_rank_checks": rank_checks, "max_recomputed_completeness_error": max_completeness_error}


def audit_model_attribution(repo_root: Path, artifact_path: Path, config_path: Path) -> dict[str, Any]:
    artifact = _read_first_records(artifact_path, 1)[0]
    dataset, config = load_run_config(config_path)
    set_global_seed(dataset.seed, deterministic=config.deterministic)
    vocabulary = load_vocabulary(_resolve_path(dataset.vocabulary, repo_root))
    split_path = _resolve_path(getattr(dataset, artifact["split"]), repo_root)
    inputs, _, metadata = load_split(split_path, vocabulary, config, show_progress=False)
    matching = [index for index, item in enumerate(metadata) if str(item["unit_id"]) == str(artifact["unit_id"])]
    if len(matching) != 1:
        raise ValueError(f"expected one matching input for {artifact['unit_id']}, found {len(matching)}")
    checkpoint = _resolve_path(Path("research/models/checkpoints/loginsight_lstm/best_seed_13.keras"), repo_root)
    adapter = KerasDetectorAdapter(DetectorContract(dataset.architecture), build_model)
    model = adapter.load_model(checkpoint)
    result = integrated_gradients_batch(model, inputs[matching[0] : matching[0] + 1], integration_steps=int(artifact["integration_steps"]), baseline_token_id=vocabulary.pad_id, output_space=artifact["output_space"])
    stored = np.asarray([float(event["attribution"]) for event in artifact["events"]])
    recomputed = result["attribution"][0]
    max_abs_diff = float(np.max(np.abs(stored - recomputed)))
    probability_diff = abs(float(artifact["probability"]) - float(result["probability"][0]))
    baseline_diff = abs(float(artifact["baseline_probability"]) - float(result["baseline_probability"][0]))
    if max_abs_diff > 1e-3 or probability_diff > 1e-5 or baseline_diff > 1e-5:
        raise AssertionError(f"independent attribution recomputation exceeded tolerance: attribution={max_abs_diff}, probability={probability_diff}, baseline={baseline_diff}")
    return {"status": "passed", "unit_id": artifact["unit_id"], "integration_steps": int(artifact["integration_steps"]), "max_attribution_abs_diff": max_abs_diff, "probability_abs_diff": probability_diff, "baseline_probability_abs_diff": baseline_diff}


def run_audit(repo_root: Path, output: Path, *, model_attribution: bool = True) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    checks: dict[str, Any] = {}
    checks["prompt_boundary"] = audit_prompt_boundary(repo_root / "research/experiments/experiment_config.json")
    checks["annotation_export"] = audit_annotation_export()
    checks["ledger_completeness"] = audit_ledger(repo_root / "research/experiments/experiment_ledger_schema.json")
    checks["metrics"] = audit_metrics(
        repo_root / "research/models/checkpoints/loginsight_lstm/thresholded/thresholded_predictions_test.jsonl.gz",
        repo_root / "research/results/tables/lstm_baseline_seed_13.json",
    )
    attribution_path = repo_root / "research/models/attributions/loginsight_lstm/seed_13_roar_test_integrated_gradients.jsonl.gz"
    checks["serialized_attribution"] = audit_serialized_attributions(attribution_path)
    if model_attribution:
        checks["model_attribution"] = audit_model_attribution(repo_root, attribution_path, repo_root / "research/experiments/lstm_training_config.json")
    report = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "step": "R7",
        "status": "passed" if all(item.get("status") == "passed" for item in checks.values()) else "failed",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "interpretation": "Independent boundary, annotation, ledger, metric, and attribution checks passed." if all(item.get("status") == "passed" for item in checks.values()) else "One or more R7 checks failed; inspect the per-check errors.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/r7_independent_audit.json"))
    parser.add_argument("--no-model-attribution", action="store_true", help="skip TensorFlow checkpoint recomputation")
    args = parser.parse_args(argv)
    try:
        report = run_audit(args.repo_root, args.output, model_attribution=not args.no_model_attribution)
    except Exception as error:
        print(f"R7 audit failed before report completion: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "output": str(args.output), "checks": list(report["checks"])}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
