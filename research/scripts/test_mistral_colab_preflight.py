#!/usr/bin/env python3
"""Run a synthetic-only Mistral-7B Colab smoke test.

This script is intended for a Colab T4 runtime. It verifies that the public
Mistral checkpoint can be loaded with 4-bit bitsandbytes quantization and can
produce one parseable JSON explanation. No project data or API keys are used.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time


MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def _extract_json(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
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
    if not isinstance(value, dict):
        raise ValueError("model output did not contain a JSON object")
    return value


def main() -> int:
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        print("RESULT: FAIL — missing runtime dependency:", error)
        return 2

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("RESULT: FAIL — no CUDA GPU is attached")
        return 1

    properties = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory GiB:", round(properties.total_memory / (1024**3), 2))
    print("Compute capability:", f"{properties.major}.{properties.minor}")
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        ],
        check=False,
    )

    print("Loading:", MODEL)
    started = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization,
        )
    except Exception as error:
        print("RESULT: FAIL — model did not load")
        print(type(error).__name__ + ":", error)
        return 1
    print("Load seconds:", round(time.monotonic() - started, 2))

    messages = [
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
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    started = time.monotonic()
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        parsed = _extract_json(text)
    except Exception as error:
        print("RESULT: FAIL — generation or JSON parsing failed")
        print(type(error).__name__ + ":", error)
        return 1

    print("Response seconds:", round(time.monotonic() - started, 2))
    print("Output:", repr(text))
    print("Parsed JSON:", json.dumps(parsed, ensure_ascii=False, sort_keys=True))
    print("Output SHA256:", hashlib.sha256(text.encode("utf-8")).hexdigest())
    print("RESULT: PASS — Mistral loaded and generated parseable JSON locally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
