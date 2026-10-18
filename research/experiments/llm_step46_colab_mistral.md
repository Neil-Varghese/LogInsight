# Step 46 Colab Mistral smoke test

The active replacement for the historical hosted GPT-OSS model is
`mistralai/Mistral-7B-Instruct-v0.3`. It is a public Apache-2.0 checkpoint and
does not require an API key or gated-model approval. The smoke test uses
4-bit NF4 bitsandbytes loading on a Colab Tesla T4 and synthetic evidence only.

The reusable smoke-test script is:

```bash
python3 research/scripts/test_mistral_colab_preflight.py
```

The captured result passed. The exact checkpoint revision is
`c170c708c41dac9275d15a8fff4eca08d52bab71`. Runtime metadata and the response
hash are recorded in
`research/results/tables/step46_mistral_smoke_test.json`.

Observed runtime: Python 3.13.15, PyTorch 2.11.0+cu130, Tesla T4 with
14.56 GiB, and compute capability 7.5. Model loading took 751.22 seconds and
the synthetic response took 4.05 seconds. The response was valid JSON and
 correctly cited the synthetic failure event `E1`.

This document records only the Step 46 synthetic/model-load smoke gate. It is
not evidence that the full 47A pilot is runnable on a T4. The subsequent
2,520-call attempt failed with CUDA out-of-memory after three completed calls;
the local MPS attempt also stalled. The execution record is
`research/experiments/step47a_execution_failures.md`.
