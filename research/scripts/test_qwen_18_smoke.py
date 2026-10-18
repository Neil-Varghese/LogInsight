#!/usr/bin/env python3
"""Run the matched 18-call Step 47B Qwen smoke test.

This is the Qwen counterpart to the completed 18-call Mistral smoke test. It
uses the same synthetic prompt, greedy decoding, 4-bit NF4 loading, and JSONL
record format so response validity and latency can be compared directly. No
project logs, attribution artifacts, or API keys are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from pilot_record import RECORD_SCHEMA_VERSION, canonical_json, prompt_messages, sha256_text
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.pilot_record import RECORD_SCHEMA_VERSION, canonical_json, prompt_messages, sha256_text


MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_CALLS = 18
MAX_NEW_TOKENS = 256
REQUIRED_KEYS = {"answer_status", "summary", "evidence_event_ids"}

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a careful log analyst. Return only one valid JSON object "
            "with exactly the keys answer_status, summary, and evidence_event_ids."
        ),
    },
    {
        "role": "user",
        "content": (
            "Use only this synthetic evidence. A failure event E1 is followed by "
            "a recovery event E2. Return the failure as the answer and cite E1."
        ),
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json(text: str) -> dict[str, Any]:
    """Extract one JSON object, tolerating a surrounding markdown fence."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        decoder = json.JSONDecoder()
        value = None
        for index, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
        if value is None:
            raise ValueError(f"model output was not valid JSON: {error}") from error

    if not isinstance(value, dict):
        raise ValueError("model output JSON was not an object")
    return value


def _valid_schema(value: dict[str, Any]) -> bool:
    return (
        set(value) == REQUIRED_KEYS
        and isinstance(value.get("answer_status"), str)
        and isinstance(value.get("summary"), str)
        and isinstance(value.get("evidence_event_ids"), list)
    )


def _runtime_metadata(torch: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        metadata["cuda_device"] = torch.cuda.get_device_name(0)
        metadata["cuda_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    return metadata


def _progress(call: int, total: int, *, response_seconds: float | None = None) -> None:
    suffix = ""
    if response_seconds is not None:
        suffix = f" | last {response_seconds:.2f}s"
    print(f"\rQwen 47B smoke test: {call}/{total}{suffix}", end="", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--calls", type=int, default=DEFAULT_CALLS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/tables/step47b_qwen_18_smoke_responses.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("research/results/tables/step47b_qwen_18_smoke_test.json"),
    )
    args, unknown = parser.parse_known_args()
    run_id = f"qwen_smoke:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    # ``colab exec -f`` executes the file through the Jupyter kernel launcher,
    # which appends its own ``-f /root/.../kernel-*.json`` argument. Accept
    # that launcher argument while still rejecting unexpected user options.
    if unknown and not (
        len(unknown) == 2
        and unknown[0] == "-f"
        and unknown[1].endswith(".json")
    ):
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.calls < 1:
        parser.error("--calls must be at least 1")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        print("Missing runtime dependency. Run the Colab setup first:", error)
        return 2

    print(f"Model: {MODEL}")
    print(f"Revision: {args.revision}")
    print(f"Calls: {args.calls}")
    print("Runtime:", json.dumps(_runtime_metadata(torch), sort_keys=True))
    if not torch.cuda.is_available():
        print("RESULT: FAIL — a CUDA GPU is required for the matched Colab test")
        return 1

    print("Loading model...")
    load_started = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=args.revision)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=args.revision,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization,
        )
    except Exception as error:  # pragma: no cover - exercised in Colab
        print("RESULT: FAIL — Qwen did not load")
        print(type(error).__name__ + ":", error)
        return 1
    load_seconds = time.monotonic() - load_started
    print(f"Model loaded in {load_seconds:.2f}s")

    inputs = tokenizer.apply_chat_template(
        MESSAGES,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    response_seconds: list[float] = []
    valid_json_calls = 0
    valid_schema_calls = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for call in range(1, args.calls + 1):
            started = time.monotonic()
            row: dict[str, Any] = {"call": call, "status": "failed"}
            try:
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw_response = tokenizer.decode(
                    output[0][prompt_tokens:], skip_special_tokens=True
                ).strip()
                parsed = _extract_json(raw_response)
                valid_json_calls += 1
                valid_schema = _valid_schema(parsed)
                valid_schema_calls += int(valid_schema)
                elapsed = time.monotonic() - started
                response_seconds.append(elapsed)
                row = {
                    "schema_version": RECORD_SCHEMA_VERSION,
                    "call": call,
                    "record_id": f"47B:qwen_smoke:call_{call}",
                    "run_id": run_id,
                    "step": "47B",
                    "status": "complete",
                    "condition": "synthetic_smoke",
                    "repetition": call,
                    "model": {"provider": "huggingface_local_or_colab", "model_name": MODEL, "endpoint": "local_transformers"},
                    "exact_prompt": MESSAGES[1]["content"],
                    "prompt_messages": MESSAGES,
                    "prompt_sha256": sha256_text(canonical_json(MESSAGES)),
                    "decoding": {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": MAX_NEW_TOKENS, "do_sample": False},
                    "raw_response": raw_response,
                    "parsed_json": parsed,
                    "valid_schema": valid_schema,
                    "response_seconds": elapsed,
                    "sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
                    "raw_response_sha256": sha256_text(raw_response),
                }
                _progress(call, args.calls, response_seconds=elapsed)
            except Exception as error:  # pragma: no cover - exercised in Colab
                elapsed = time.monotonic() - started
                row.update(
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "response_seconds": elapsed,
                    }
                )
                _progress(call, args.calls, response_seconds=elapsed)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
    print()

    report = {
        "created_at_utc": _utc_now(),
        "step": "47B",
        "smoke_test": "matched_18_call",
        "prompt_data": "synthetic_only",
        "model": MODEL,
        "revision": args.revision,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "quantization": "4-bit NF4 with double quantization",
        },
        "calls_requested": args.calls,
        "calls_completed": sum(row.get("status") == "complete" for row in rows),
        "valid_json_calls": valid_json_calls,
        "valid_schema_calls": valid_schema_calls,
        "load_seconds": load_seconds,
        "response_seconds_total": sum(response_seconds),
        "response_seconds_average": (
            sum(response_seconds) / len(response_seconds) if response_seconds else None
        ),
        "unique_response_hashes": len(
            {row["sha256"] for row in rows if "sha256" in row}
        ),
        "runtime": _runtime_metadata(torch),
        "responses_jsonl": str(args.output),
        "status": (
            "passed"
            if len(rows) == args.calls
            and valid_json_calls == args.calls
            and valid_schema_calls == args.calls
            else "failed"
        ),
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Responses: {args.output}")
    print(f"Report: {args.report}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
