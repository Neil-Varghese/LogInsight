# Research experiments

Store experiment configurations, run metadata, and pilot/main experiment
outputs here. Do not overwrite a completed run; create a new run directory.

`experiment_config.json` is the single source of truth for the current
protocol. The protocol is externally registered at
https://osf.io/qrvuy/overview. Values marked `TO_BE_*`, `null`, or with a
provisional status must not enter confirmatory output; resolve them through a
documented registration update or label the resulting run exploratory.

`experiment_ledger_schema.json` defines one record for one block/window and
one explanation condition. Keep the exact prompt, raw response, attribution
visibility, perturbation outputs, and separate outcome fields for every record.

`llm_explanation_schema.json` defines the structured payload inside an LLM
explanation response. It requires typed claims, event and line citations,
confidence values, and an explicit abstention state. Raw response parsing is
implemented by `research/scripts/parse_explanation_responses.py`; invalid
responses are classified rather than repaired.

`protocol_freeze_step62.json` freezes the registration-critical definitions:
event units, normal-block replacement alignment, evidence budget, empty-set
F1 handling, false-positive separation, primary strata, mixed-effects random
effects, and the primary shuffled-attribution method. It is the
registration-critical freeze referenced by the OSF record.

Step 63 preparation is recorded under `registration/`. Run
`research/scripts/prepare_registration.py` to hash the exact protocol files
and create `step63_preregistration_manifest.json`. The script does not contact
OSF. The current registered manifest records the OSF URL; use `--exploratory`
only for explicitly exploratory runs that are not presented as confirmatory
results.

- `pilot/`: minimal HDFS go/no-go pilot.
- `main/`: registered main experiments.

Step 47 is deliberately executed in two ordered parts: 47A runs Mistral first
and 47B runs Qwen second using the same frozen inputs and conditions. 47A is
currently blocked after failed API, Colab-T4, and local-MPS execution attempts;
see `step47a_execution_failures.md`. Results from 47A must not be used to
change the protocol before 47B. Historical GPT-OSS API records are retained
separately and are not part of the selected model pair.

Step 47A stores raw Mistral generation records in
`research/experiments/pilot/step47a_mistral_generation.jsonl`. Parse and
validate those records with `research/scripts/parse_explanation_responses.py`.
The runner's default two-item limit, progress bar, and machine-readable status
file make bounded local/Colab runs observable and recoverable. A model-load
success is not a generation success; preserve both JSONL and status artifacts.
The older GPT-OSS output remains in its original file only as a historical
provenance record.

Step 50 records the exact prompt messages, hashes of the prompt and raw
response, decoding settings, request/run identifiers, model metadata, and
provider payload alongside each completed generation.

Step 56 is documented in
`annotation_ethics_documentation.md`. It defines the consent, compensation,
anonymization, incident, retention, and institutional approval or exemption
records required before recruitment or core labeling. The document is
prepared, but its labeling gate remains blocked until the real institutional
decision and coordinator sign-off are recorded; no approval is implied by the
presence of the file.

The matched Step 47B smoke runner is
`research/scripts/test_qwen_18_smoke.py`. It runs 18 identical synthetic calls
with Qwen, uses the same greedy 4-bit NF4 setup as the Mistral comparison, and
writes raw responses to
`research/results/tables/step47b_qwen_18_smoke_responses.jsonl` plus a summary
report at `research/results/tables/step47b_qwen_18_smoke_test.json`.
