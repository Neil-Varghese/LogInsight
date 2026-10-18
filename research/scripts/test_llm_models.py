"""Run Step 46 smoke tests for the selected Groq and Qwen models.

The Groq test uses ``GROQ_API_KEY`` from the environment and a synthetic log
prompt. The Qwen test is intended for local or Colab execution with the
public Hugging Face checkpoint and does not require gated-model approval. No
project log data or API secrets are embedded in the script or its report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
OPEN_WEIGHT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 512
INPUT_CONTEXT_CAP_TOKENS = 6500
LOCAL_SEED = 2026


def _load_dotenv() -> None:
    """Load simple local ``.env`` entries without overriding the shell."""

    script_path = Path(__file__).resolve()
    env_path = next(
        (candidate / ".env" for candidate in (script_path.parent, *script_path.parents)
         if (candidate / ".env").exists()),
        None,
    )
    if env_path is None:
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()

SMOKE_PROMPT = (
    "Return JSON with exactly these keys: answer_status, summary, and evidence_event_ids. "
    "Use only this synthetic log evidence. The evidence contains one failure event "
    "and one recovery event.\n"
    "event_id=E1 line_id=10 template=BlockReceiver failure raw=receiver failed\n"
    "event_id=E2 line_id=11 template=BlockReceiver recovery raw=receiver recovered"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_chat_completion_text(payload: dict[str, Any], provider: str) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{provider} response contained no choices")
    message = choices[0].get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{provider} response contained no text content")
    return content.strip()


def _parse_json_object(text: str) -> dict[str, Any]:
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
            raise RuntimeError(f"model output was not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("model output JSON was not an object")
    return value


def run_groq(api_key: str) -> dict[str, Any]:
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a log analyst. Return only valid JSON and do not invent evidence.",
            },
            {"role": "user", "content": SMOKE_PROMPT},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }
    request = Request(
        GROQ_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LogInsight-FV-Step46/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq API request failed ({error.code}): {detail[:500]}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the Groq API: {error.reason}") from error

    text = _extract_chat_completion_text(response_payload, "Groq")
    parsed = _parse_json_object(text)
    return {
        "status": "passed",
        "provider": "groq",
        "endpoint": GROQ_ENDPOINT,
        "requested_model": GROQ_MODEL,
        "reasoning_effort": "low",
        "response_json": parsed,
        "response_text_sha256": __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
    }


def _runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metadata["cuda_device"] = torch.cuda.get_device_name(0)
            metadata["cuda_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except ImportError:
        metadata["torch"] = None
        metadata["cuda_available"] = False
    return metadata


def run_qwen(*, revision: str, hf_token: str | None) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Qwen smoke test requires torch and transformers; run the Colab setup first"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        OPEN_WEIGHT_MODEL,
        revision=revision,
        token=hf_token,
    )
    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "token": hf_token,
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_pretrained(OPEN_WEIGHT_MODEL, **load_kwargs)
    messages = [
        {
            "role": "system",
            "content": "You are a log analyst. Return only one valid JSON object and do not invent evidence.",
        },
        {"role": "user", "content": SMOKE_PROMPT},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        inputs = tokenizer(SMOKE_PROMPT, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    torch.manual_seed(LOCAL_SEED)
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    parsed = _parse_json_object(generated)
    return {
        "status": "passed",
        "provider": "huggingface_local_or_colab",
        "requested_model": OPEN_WEIGHT_MODEL,
        "requested_revision": revision,
        "response_json": parsed,
        "response_text_sha256": __import__("hashlib").sha256(generated.encode("utf-8")).hexdigest(),
        "runtime": _runtime_metadata(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("groq", "qwen", "all"), default="all")
    parser.add_argument("--groq-key-env", default="GROQ_API_KEY")
    parser.add_argument("--qwen-revision", default=os.getenv("QWEN_MODEL_REVISION", "main"))
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/step46_llm_smoke_test.json"))
    args = parser.parse_args()

    results: dict[str, Any] = {
        "created_at_utc": _utc_now(),
        "step": 46,
        "prompt_data": "synthetic_only",
        "decoding": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "input_context_cap_tokens": INPUT_CONTEXT_CAP_TOKENS,
            "input_context_cap_policy": "deterministic_cap; record omitted event IDs; never silently truncate",
            "local_seed": LOCAL_SEED,
        },
        "models": {},
    }
    failures = 0
    if args.model in ("groq", "all"):
        key = os.getenv(args.groq_key_env)
        if not key:
            results["models"]["groq"] = {"status": "skipped", "reason": f"{args.groq_key_env} is not set"}
        else:
            try:
                results["models"]["groq"] = run_groq(key)
            except RuntimeError as error:
                results["models"]["groq"] = {"status": "failed", "error": str(error)}
                failures += 1
    if args.model in ("qwen", "all"):
        try:
            results["models"]["qwen"] = run_qwen(
                revision=args.qwen_revision,
                hf_token=os.getenv(args.hf_token_env),
            )
        except RuntimeError as error:
            results["models"]["qwen"] = {"status": "failed", "error": str(error)}
            failures += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": results["models"]}, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
