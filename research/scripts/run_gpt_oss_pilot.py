#!/usr/bin/env python3
"""Run the Step 47A exploratory GPT-OSS explanation pilot.

The runner is intentionally sequential and resumable. It writes one complete
raw prompt/response record per model-condition-repetition attempt. Structured
response parsing is deliberately deferred to Step 48.

The default run processes only two annotation items. Use ``--max-items``
explicitly for a larger exploratory pilot. No API call is made until the input
artifacts and API key have been validated.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:
    from contrastive_prompt import render_contrastive_prompt
    from no_detector_context_prompt import render_no_detector_context_prompt
    from shuffled_integrated_gradients import build_shuffled_record, build_shuffled_prompt_payload
    from training_only_retrieval import TrainingNormalRetriever, iter_records
    from pilot_record import RECORD_SCHEMA_VERSION, SYSTEM_PROMPT, prompt_messages, reproducibility_fields
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.contrastive_prompt import render_contrastive_prompt
    from research.scripts.no_detector_context_prompt import render_no_detector_context_prompt
    from research.scripts.shuffled_integrated_gradients import build_shuffled_record, build_shuffled_prompt_payload
    from research.scripts.training_only_retrieval import TrainingNormalRetriever, iter_records
    from research.scripts.pilot_record import RECORD_SCHEMA_VERSION, SYSTEM_PROMPT, prompt_messages, reproducibility_fields


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "research/experiments/experiment_config.json"
DEFAULT_POOL = REPO_ROOT / "research/experiments/pilot/annotation_pool.jsonl.gz"
DEFAULT_TEST_EVIDENCE = REPO_ROOT / "research/data/processed/hdfs_block_evidence_test.jsonl.gz"
DEFAULT_TRAIN_EVIDENCE = REPO_ROOT / "research/data/processed/hdfs_block_evidence_train.jsonl.gz"
DEFAULT_PREDICTION = REPO_ROOT / "research/models/checkpoints/loginsight_lstm/predictions_test.jsonl.gz"
DEFAULT_OUTPUT = REPO_ROOT / "research/experiments/pilot/step47a_gpt_oss_generation.jsonl"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_CONDITIONS = (
    "blinded",
    "unblinded",
    "shuffled_integrated_gradients",
    "detector_only",
    "unblinded_no_normal",
    "shuffled_integrated_gradients_no_normal",
    "no_detector_context",
)
ATTRIBUTION_CONDITIONS = {
    "unblinded",
    "shuffled_integrated_gradients",
    "unblinded_no_normal",
    "shuffled_integrated_gradients_no_normal",
}
NORMAL_CONDITIONS = {"blinded", "unblinded", "shuffled_integrated_gradients"}
RETRY_AFTER_SECONDS_RE = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


def load_repo_env(path: Path | None = None) -> None:
    """Load simple KEY=value entries from the repository .env if present.

    Existing process environment variables always win. This keeps API keys out
    of command history while allowing the documented runner command to work
    without a separate shell ``source`` step.
    """

    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            records.append(record)
    return records


def load_by_unit(path: Path, unit_key: str = "unit_id") -> dict[str, dict[str, Any]]:
    result = {}
    for record in load_jsonl(path):
        unit_id = record.get(unit_key, record.get("BlockId"))
        if unit_id in (None, ""):
            raise ValueError(f"{path} contains a record without {unit_key}/BlockId")
        unit_id = str(unit_id)
        if unit_id in result:
            raise ValueError(f"{path} contains duplicate unit {unit_id!r}")
        result[unit_id] = record
    return result


def load_selected_by_unit(path: Path, unit_ids: set[str], unit_key: str = "unit_id") -> dict[str, dict[str, Any]]:
    """Stream only the requested records from a large JSONL artifact.

    The pilot needs one retrieved normal block per query, not the complete
    training split.  Loading the whole compressed training file into a dict
    duplicated the expensive scan performed while fitting the retriever and
    could hold hundreds of megabytes of unnecessary records in memory.
    """

    requested = {str(unit_id) for unit_id in unit_ids}
    if not requested:
        return {}
    result: dict[str, dict[str, Any]] = {}
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            unit_id = record.get(unit_key, record.get("BlockId"))
            if unit_id is None:
                continue
            unit_id = str(unit_id)
            if unit_id not in requested:
                continue
            if unit_id in result:
                raise ValueError(f"{path} contains duplicate requested unit {unit_id!r}")
            result[unit_id] = record
            if len(result) == len(requested):
                break
    missing = sorted(requested - set(result))
    if missing:
        raise ValueError(f"{path} is missing requested units: {missing[:5]}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ARCHITECTURE=PATH")
    architecture, raw_path = value.split("=", 1)
    if not architecture or not raw_path:
        raise argparse.ArgumentTypeError("expected ARCHITECTURE=PATH")
    return architecture, Path(raw_path).expanduser().resolve()


def top_k_attribution_payload(record: Mapping[str, Any], top_k: int) -> dict[str, Any]:
    events = [event for event in record.get("events", []) if not event.get("is_padding", False)]
    events.sort(key=lambda event: (int(event.get("rank", 10**9)), int(event.get("input_position", 10**9))))
    selected = events[:top_k]
    return {
        "top_k": len(selected),
        "events": [
            {
                "input_position": int(event["input_position"]),
                "original_sequence_index": event.get("original_sequence_index"),
                "event_id": str(event["event_id"]),
                "attribution": float(event["attribution"]),
                "absolute_attribution": float(event["absolute_attribution"]),
                "rank": int(event["rank"]),
            }
            for event in selected
        ],
    }


def build_prompt(
    condition: str,
    anomalous_record: Mapping[str, Any],
    detector_output: Mapping[str, Any],
    normal_record: Mapping[str, Any] | None,
    attribution_record: Mapping[str, Any] | None,
    *,
    top_k: int,
    shuffle_seed: int,
    config_path: Path,
) -> tuple[str, str, str]:
    if condition == "no_detector_context":
        return render_no_detector_context_prompt(anomalous_record, config_path=config_path)

    normal = normal_record if condition in NORMAL_CONDITIONS else None
    if condition in ATTRIBUTION_CONDITIONS and attribution_record is None:
        raise ValueError(f"condition {condition!r} requires an Integrated Gradients record")

    attribution = None
    shuffled = None
    if condition in {"unblinded", "unblinded_no_normal"}:
        attribution = top_k_attribution_payload(attribution_record or {}, top_k)
    elif condition in {"shuffled_integrated_gradients", "shuffled_integrated_gradients_no_normal"}:
        shuffled_record = build_shuffled_record(
            attribution_record or {},
            shuffle_type="within_block_permutation",
            shuffle_seed=shuffle_seed,
        )
        shuffled = build_shuffled_prompt_payload(shuffled_record)
        shuffled["events"] = shuffled["events"][:top_k]
        shuffled["top_k"] = len(shuffled["events"])

    rendered_condition, template_id, prompt = render_contrastive_prompt(
        anomalous_record,
        detector_output,
        normal_comparison=condition in NORMAL_CONDITIONS,
        attribution=(
            "integrated_gradients" if condition in {"unblinded", "unblinded_no_normal"}
            else "shuffled_integrated_gradients" if condition in {"shuffled_integrated_gradients", "shuffled_integrated_gradients_no_normal"}
            else "none"
        ),
        normal_record=normal,
        integrated_gradients=attribution,
        shuffled_integrated_gradients=shuffled,
        config_path=config_path,
    )
    return rendered_condition, template_id, prompt


class ProgressBar:
    def __init__(self, total: int, *, enabled: bool = True) -> None:
        self.total = max(total, 1)
        self.enabled = enabled
        self.completed = 0
        self.started = time.monotonic()

    def update(self, label: str) -> None:
        self.completed += 1
        if not self.enabled:
            return
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.completed / elapsed
        remaining = max(self.total - self.completed, 0)
        eta = remaining / rate if rate else 0
        width = 28
        filled = min(width, int(width * self.completed / self.total))
        bar = "#" * filled + "." * (width - filled)
        text = f"Step 47A [{bar}] {self.completed}/{self.total} | {label[:42]:42} | {rate:.2f}/s | ETA {eta:6.1f}s"
        print("\r" + text, end="", file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.enabled:
            print(file=sys.stderr)


class GroqChatClient:
    def __init__(self, *, api_key: str, endpoint: str, model: str, temperature: float, top_p: float, max_output_tokens: int, timeout: float, retries: int, extra_body: Mapping[str, Any] | None = None) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.retries = retries
        self.extra_body = dict(extra_body or {})

    def generate(self, prompt: str, *, request_id: str) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
        }
        body.update(self.extra_body)
        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Groq's edge layer rejects Python urllib's default
                # ``Python-urllib/...`` user-agent with HTTP 403/1010.
                "User-Agent": "LogInsight-Step47A/1.0",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                choices = payload.get("choices") or []
                if not choices:
                    raise ValueError("provider response contains no choices")
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                if not isinstance(content, str):
                    raise ValueError("provider response choice has no text content")
                return {"provider_payload": payload, "raw_text": content, "attempt": attempt}
            except urllib.error.HTTPError as error:
                # Preserve the provider's diagnostic body. A status code alone
                # cannot distinguish an invalid request, model access problem,
                # rate limit, or oversized prompt.
                try:
                    detail = error.read().decode("utf-8", errors="replace")[:2000]
                except Exception:  # pragma: no cover - defensive fallback
                    detail = "<unable to read provider error body>"
                last_error = RuntimeError(
                    f"HTTP {error.code} from provider: {detail or '<empty body>'}"
                )
                if attempt >= self.retries or error.code in {400, 401, 403, 404}:
                    break
                delay = min(30.0, 2.0 ** (attempt - 1) + random.random())
                if error.code == 429:
                    retry_after = None
                    header_value = error.headers.get("Retry-After") if error.headers else None
                    if header_value:
                        try:
                            retry_after = float(header_value)
                        except ValueError:
                            retry_after = None
                    if retry_after is None:
                        match = RETRY_AFTER_SECONDS_RE.search(detail)
                        if match:
                            retry_after = float(match.group(1))
                    if retry_after is not None:
                        delay = max(delay, retry_after + random.uniform(0.25, 1.0))
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                if attempt >= self.retries:
                    break
                delay = min(30.0, 2.0 ** (attempt - 1) + random.random())
                time.sleep(delay)
        raise RuntimeError(f"GPT-OSS request failed for {request_id} after {self.retries} attempts: {last_error}") from last_error


def _read_completed(output: Path) -> set[str]:
    if not output.exists():
        return set()
    completed = set()
    with output.open("rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("status") == "complete":
                    completed.add(str(record["record_id"]))
    return completed


def run_pilot(
    *,
    pool_path: Path,
    evidence_path: Path,
    training_evidence_path: Path,
    prediction_path: Path,
    attribution_path: Path | None,
    output_path: Path,
    config_path: Path,
    architecture: str,
    max_items: int | None,
    repetitions: int,
    conditions: tuple[str, ...],
    top_k: int,
    shuffle_seed: int,
    client: Any,
    show_progress: bool,
    provider_name: str = "groq",
    run_label: str = "gpt_oss",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if repetitions < 1 or top_k < 1:
        raise ValueError("repetitions and top_k must be positive")
    pool = [record for record in load_jsonl(pool_path) if record.get("architecture") == architecture]
    if max_items is not None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        pool = pool[:max_items]
    if not pool:
        raise ValueError(f"no annotation-pool items found for architecture {architecture!r}")
    evidence = load_by_unit(evidence_path)
    predictions = load_by_unit(prediction_path)
    attribution = load_by_unit(attribution_path) if attribution_path else {}
    if any(condition in ATTRIBUTION_CONDITIONS for condition in conditions) and not attribution_path:
        raise ValueError("attribution conditions require --attribution-file ARCH=PATH")

    retriever = TrainingNormalRetriever(training_evidence_path).fit()
    # Retrieve candidate IDs first, then stream only those normal records. This
    # avoids materializing the entire 252 MB compressed training artifact a
    # second time before the first API request.
    matches_by_unit: dict[str, list[dict[str, Any]]] = {}
    requested_normal_units: set[str] = set()
    for item in pool:
        unit_id = str(item["unit_id"])
        matches = retriever.retrieve(evidence[unit_id], retrieval_k=1)
        matches_by_unit[unit_id] = matches
        requested_normal_units.add(str(matches[0]["candidate_unit_id"]))
    training_records = load_selected_by_unit(
        training_evidence_path,
        requested_normal_units,
        unit_key="BlockId",
    )
    completed = _read_completed(output_path)
    run_id = f"{run_label}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(pool) * len(conditions) * repetitions
    progress = ProgressBar(total, enabled=show_progress)
    status_counts = {"complete": 0, "skipped": 0, "failed": 0}
    def emit(event: str, **details: Any) -> None:
        if progress_callback is not None:
            progress_callback({
                "event": event,
                "run_id": run_id,
                "total_requested": total,
                "status_counts": dict(status_counts),
                **details,
            })
    emit("run_started", items=len(pool), conditions=list(conditions), repetitions=repetitions)
    try:
        with output_path.open("a", encoding="utf-8") as handle:
            for item in pool:
                unit_id = str(item["unit_id"])
                if unit_id not in evidence or unit_id not in predictions:
                    raise ValueError(f"unit {unit_id!r} is missing from evidence or predictions")
                anomalous = evidence[unit_id]
                detector_output = {
                    "detector_id": architecture,
                    "probability": predictions[unit_id]["probability"],
                    "prediction": predictions[unit_id].get("prediction"),
                    "threshold": predictions[unit_id].get("threshold"),
                }
                matches = matches_by_unit[unit_id]
                normal_record = training_records[str(matches[0]["candidate_unit_id"])]
                for condition in conditions:
                    for repetition in range(1, repetitions + 1):
                        record_id = f"47A:{run_label}:{architecture}:{unit_id}:{condition}:rep_{repetition}"
                        if record_id in completed:
                            status_counts["skipped"] += 1
                            emit("skipped", record_id=record_id, unit_id=unit_id, condition=condition, repetition=repetition)
                            progress.update(f"skip {unit_id} {condition} r{repetition}")
                            continue
                        started = time.time()
                        prompt = None
                        prompt_metadata: dict[str, Any] = {}
                        try:
                            actual_condition, template_id, prompt = build_prompt(
                                condition,
                                anomalous,
                                detector_output,
                                normal_record,
                                attribution.get(unit_id),
                                top_k=top_k,
                                shuffle_seed=shuffle_seed,
                                config_path=config_path,
                            )
                            emit(
                                "generating",
                                record_id=record_id,
                                unit_id=unit_id,
                                condition=actual_condition,
                                repetition=repetition,
                            )
                            prompt_metadata = reproducibility_fields(
                                user_prompt=prompt,
                                raw_response=None,
                                decoding={
                                    "temperature": getattr(client, "temperature", None),
                                    "top_p": getattr(client, "top_p", None),
                                    "max_output_tokens": getattr(client, "max_output_tokens", None),
                                    "do_sample": False,
                                },
                            )
                            response = client.generate(prompt, request_id=record_id)
                            prompt_metadata.update(
                                reproducibility_fields(
                                    user_prompt=prompt,
                                    raw_response=response["raw_text"],
                                    decoding=prompt_metadata["decoding"],
                                )
                            )
                            record = {
                                "schema_version": RECORD_SCHEMA_VERSION,
                                "record_id": record_id,
                                "run_id": run_id,
                                "status": "complete",
                                "exploratory": True,
                                "step": "47A",
                                "model": {
                                    "provider": provider_name,
                                    "model_name": client.model,
                                    "endpoint": client.endpoint,
                                },
                                "dataset": item.get("dataset", "HDFS"),
                                "unit_id": unit_id,
                                "architecture": architecture,
                                "detector_seed": predictions[unit_id].get("seed"),
                                "condition": actual_condition,
                                "template_id": template_id,
                                "repetition": repetition,
                                "evidence_budget_top_k": top_k,
                                "normal_comparison_unit_id": normal_record.get("BlockId"),
                                "normal_similarity": matches[0]["similarity"],
                                "prompt": prompt,
                                "provider_payload": response["provider_payload"],
                                "response_attempt": response["attempt"],
                                "parsed_explanation": None,
                                "parsing_status": "deferred_to_step48",
                                "latency_seconds": time.time() - started,
                                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            }
                            record.update(prompt_metadata)
                            record["request_id"] = record_id
                            if getattr(client, "extra_body", None):
                                record["model"]["extra_request_parameters"] = dict(client.extra_body)
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            handle.flush()
                            status_counts["complete"] += 1
                            completed.add(record_id)
                            emit("complete", record_id=record_id, unit_id=unit_id, condition=actual_condition, repetition=repetition)
                        except Exception as error:
                            status_counts["failed"] += 1
                            emit("failed", record_id=record_id, unit_id=unit_id, condition=condition, repetition=repetition, error_type=type(error).__name__)
                            error_record = {
                                "schema_version": RECORD_SCHEMA_VERSION,
                                "record_id": record_id,
                                "run_id": run_id,
                                "status": "failed",
                                "exploratory": True,
                                "step": "47A",
                                "unit_id": unit_id,
                                "architecture": architecture,
                                "condition": condition,
                                "repetition": repetition,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            }
                            if prompt is not None:
                                error_record.update(prompt_metadata)
                                error_record["raw_response"] = None
                                error_record["raw_response_sha256"] = None
                            handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                            handle.flush()
                            raise
                        finally:
                            progress.update(f"{unit_id} {condition} r{repetition}")
    finally:
        progress.close()
        emit("run_finished")
    return {
        "step": "47A",
        "status": "complete" if status_counts["failed"] == 0 else "failed",
        "exploratory": True,
        "provider": provider_name,
        "architecture": architecture,
        "model": client.model,
        "items": len(pool),
        "conditions": list(conditions),
        "repetitions": repetitions,
        "total_requested": total,
        "status_counts": status_counts,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
    }


def main() -> None:
    load_repo_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_TEST_EVIDENCE)
    parser.add_argument("--training-evidence", type=Path, default=DEFAULT_TRAIN_EVIDENCE)
    parser.add_argument("--prediction-file", type=parse_mapping, default=("loginsight_lstm", DEFAULT_PREDICTION))
    parser.add_argument("--attribution-file", type=parse_mapping)
    parser.add_argument("--architecture", default="loginsight_lstm")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-items", type=int, default=2, help="default 2 to prevent accidental large API runs")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    parser.add_argument("--endpoint", default=os.getenv("LOGINSIGHT_LLM_BASE_URL", DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=os.getenv("LOGINSIGHT_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("GROQ_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("GROQ_API_KEY is required; no API call was made")
    prediction_architecture, prediction_path = args.prediction_file
    if prediction_architecture != args.architecture:
        raise SystemExit("--prediction-file architecture must match --architecture")
    attribution_path = None
    if args.attribution_file:
        attribution_architecture, attribution_path = args.attribution_file
        if attribution_architecture != args.architecture:
            raise SystemExit("--attribution-file architecture must match --architecture")
    conditions = tuple(value.strip() for value in args.conditions.split(",") if value.strip())
    unknown = set(conditions) - set(DEFAULT_CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    client = GroqChatClient(
        api_key=args.api_key,
        endpoint=args.endpoint,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    report = run_pilot(
        pool_path=args.pool.resolve(),
        evidence_path=args.evidence.resolve(),
        training_evidence_path=args.training_evidence.resolve(),
        prediction_path=prediction_path,
        attribution_path=attribution_path,
        output_path=args.output.resolve(),
        config_path=args.config.resolve(),
        architecture=args.architecture,
        max_items=args.max_items,
        repetitions=args.repetitions,
        conditions=conditions,
        top_k=args.top_k,
        shuffle_seed=args.shuffle_seed,
        client=client,
        show_progress=not args.no_progress,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
