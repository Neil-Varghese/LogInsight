#!/usr/bin/env python3
"""Check a Colab GPU and run one tiny local GPT-OSS 20B response.

This script is intended to be uploaded to and run inside Google Colab. It
does not use project data, API keys, or provider quotas.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time


MODEL = "openai/gpt-oss-20b"


def main() -> int:
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    try:
        import torch
    except ImportError as error:
        print("ERROR: install torch before running this script:", error)
        return 2

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("RESULT: FAIL — no CUDA GPU is attached")
        return 1

    properties = torch.cuda.get_device_properties(0)
    memory_gib = properties.total_memory / (1024 ** 3)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory GiB:", round(memory_gib, 2))
    print("Compute capability:", f"{properties.major}.{properties.minor}")
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=False,
    )
    if memory_gib < 16:
        # The model documentation recommends 16 GiB, but Transformers can
        # sometimes place a small part of the model on CPU with device_map=
        # "auto". Keep testing on free Colab T4 runtimes rather than failing
        # before that fallback has a chance to run.
        print("WARNING: less than the recommended 16 GiB; allowing CPU offload")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        print("ERROR: install transformers and accelerate before running this script:", error)
        return 2

    print("Loading:", MODEL)
    started = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            offload_buffers=True,
        )
    except Exception as error:
        print("RESULT: FAIL — model did not load")
        print(type(error).__name__ + ":", error)
        return 1
    print("Load seconds:", round(time.monotonic() - started, 2))

    messages = [
        {"role": "system", "content": "Reasoning: low"},
        {"role": "user", "content": "Reply with exactly: COLAB_GPT_OSS_OK"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    print("Response seconds:", round(time.monotonic() - started, 2))
    print("Output:", repr(text))
    print("RESULT: PASS — GPT-OSS loaded and generated locally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
