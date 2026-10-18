# Step 47A execution record

This document records execution outcomes after the OSF registration. It does
not alter the registered protocol, hypotheses, conditions, or analysis plan.
The full Step 47A target is 120 annotation units × 7 conditions × 3
repetitions = 2,520 generation calls.

## Status at 2026-10-18

The full Mistral 47A run is **not complete**. No confirmatory or pilot result
may be computed from the partial outputs below, and Step 48 parsing/analysis
must wait for a successful generation run.

### Historical API attempts

The historical GPT-OSS path was attempted through hosted APIs before the model
selection was frozen. One attempt failed at DNS/name resolution and another
returned HTTP 403 Forbidden. Existing GPT-OSS/Groq and Cloudflare JSONL files
are retained for provenance only. They are not Mistral 47A data and are not
evidence that the registered Mistral run completed.

### Google Colab T4 attempt

The active Mistral runner loaded `mistralai/Mistral-7B-Instruct-v0.3` on a
Tesla T4, but the full run failed during the fourth generation because the
14.56-GiB GPU ran out of memory. The run produced three completed records and
one failed record, then stopped because the runner raises on a generation
failure. The requested total was 2,520 calls; completed calls were 3 and
failed calls were 1. The remote Colab output/status must be downloaded and
preserved before the runtime is discarded. It is not currently treated as a
local completed artifact.

### Local Apple-MPS attempt

The model was downloaded once locally (three sharded files, approximately
14.50 GB) and loaded successfully with the MPS code path. A one-record smoke
test then stalled during the first generation for approximately 1 hour 46
minutes. The process was stopped by the user; the output JSONL contains zero
records and the status file records `stopped_by_user`. Physical memory usage
approached the Mac's 24-GB unified-memory limit. The full 47A run was never
started locally.

The local Hugging Face cache remains available, so a future attempt does not
need to redownload the model. Model download completion is not generation
success: loading the checkpoint completed, but the local Transformers/MPS
generation path was not operationally viable in this test.

## Operational lessons

1. A model-load smoke test is insufficient. The gate must include at least one
   real project-format generation and record generation latency, peak memory,
   output persistence, and process termination behavior.
2. A free T4 is adequate for the earlier synthetic smoke test but is not
   adequate for this full Mistral configuration under the observed prompt and
   generation settings.
3. The local MPS path must not be described as supported merely because the
   checkpoint loads. It needs a successful bounded generation before it can be
   used for 47A.
4. Every run must preserve its JSONL output and machine-readable status file;
   remote Colab files must be downloaded before the session ends.
5. Do not launch the 2,520-call run until a new runtime has passed a bounded
   generation test and the user explicitly authorizes the full run.

## Next runtime decision

The next attempt should use a larger CUDA runtime (preferably L4/A100) or a
validated Apple-native inference backend such as MLX or llama.cpp, with the
exact checkpoint, prompt format, quantization, and output schema recorded.
Changing the model, prompt, conditions, or analysis because of these runtime
failures would require a separate documented protocol decision; it must not be
silently folded into the registered study.
