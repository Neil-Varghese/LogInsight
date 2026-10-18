# LLM model selection

Step 45 selects two models for the LogInsight-FV explanation study:

1. Primary open-weight model: `mistralai/Mistral-7B-Instruct-v0.3`, run
   locally or in Colab with 4-bit loading; Colab GPU is the default runtime
   target.
2. Matched second model: `Qwen/Qwen2.5-7B-Instruct`, run locally or in Colab
   with 4-bit loading.

Mistral remains the selected replacement for the historical GPT-OSS path, but
the Step 46 model-load/synthetic smoke result must not be confused with full
pilot feasibility. The later full-run Colab T4 attempt loaded Mistral but
failed with CUDA OOM after 3 completed calls, and the local Apple-MPS smoke
test stalled before producing a record. See
`research/experiments/step47a_execution_failures.md`.
Mistral is publicly downloadable, Apache-2.0 licensed, instruction tuned, and
requires no API key or gated-model approval. Qwen remains the second model so
the study compares two different model families rather than repeating the
same checkpoint.

The Mistral and Qwen licenses and exact checkpoint revisions must be retained
with the experiment record.

The pair is frozen at the model-family and access-mode level. Step 46 recorded
the exact Mistral checkpoint revision
`c170c708c41dac9275d15a8fff4eca08d52bab71`, the exact Qwen checkpoint revision
`a09a35458c702b33eeacc393d103063234e8bc28`, the Colab T4 runtime, and the
context-length/token-budget settings. The synthetic smoke-test reports are
`research/results/tables/step46_mistral_smoke_test.json`,
`research/results/tables/step46_qwen_smoke_test.json`, and the historical API
smoke report `research/results/tables/step46_llm_smoke_test.json`.
API keys and other secrets must never be stored in the repository.

The request policy uses a deterministic 6,500-token input cap to leave room for
prompt overhead and output under practical provider limits. If a block exceeds
the cap, omitted event IDs must be recorded; evidence must never be silently
truncated.
