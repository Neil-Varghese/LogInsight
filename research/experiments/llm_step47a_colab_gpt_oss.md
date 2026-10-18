> Historical document. GPT-OSS is no longer the selected research model.
> Use `llm_step46_colab_mistral.md` and `research/scripts/run_mistral_pilot.py`
> for the active Mistral workflow.

# Historical Step 47A GPT-OSS 20B Colab preflight

This is only a hardware/model-load test. It does not run the research
experiment and does not use Groq or Cloudflare API quota.

Open a new Google Colab notebook, select **Runtime → Change runtime type →
GPU**, and run the cells below in order. Do not upload the project data yet.

## Cell 1: record the free GPU

```python
import subprocess, sys, torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("No GPU is attached. Select Runtime -> Change runtime type -> GPU.")

gpu = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
memory_gb = props.total_memory / (1024 ** 3)
print("GPU:", gpu)
print("GPU memory GiB:", round(memory_gb, 2))
print("Compute capability:", f"{props.major}.{props.minor}")

subprocess.run([
    "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader"
], check=False)
```

## Cell 2: install the model runtime

```python
!pip -q install -U "transformers>=4.55" accelerate huggingface_hub kernels
```

Restart the runtime if Colab asks you to, then run Cell 1 again.

## Cell 3: load GPT-OSS and run one tiny response

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "openai/gpt-oss-20b"
print("Loading", MODEL)
started = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype="auto",
)
print("Loaded in seconds:", round(time.time() - started, 2))

messages = [{
    "role": "user",
    "content": "Reply with exactly: COLAB_GPT_OSS_OK",
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
inputs = {key: value.to(model.device) for key, value in inputs.items()}

started = time.time()
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )
text = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
).strip()
print("Response time seconds:", round(time.time() - started, 2))
print("Output:", repr(text))
```

## How to interpret the result

- **GPU is T4 / no model load:** stop; free Colab is not suitable for this
  GPT-OSS setup. Do not spend time uploading the project.
- **Model loads and returns `COLAB_GPT_OSS_OK`:** save the GPU name, memory,
  load time, and response time. Then we can build the resumable Colab runner.
- **GPU is L4/A100 but load fails:** copy only the error message; do not paste
  any tokens or private credentials.

GPT-OSS 20B is an open-weight model and OpenAI documents it as requiring
approximately 16 GB in its optimized format. Colab GPU availability is
dynamic, so a free session is not guaranteed to provide a suitable GPU.
