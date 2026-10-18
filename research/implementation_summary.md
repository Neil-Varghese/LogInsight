# LogInsight Implementation Summary

## Purpose of this document

This document summarizes the LogInsight research implementation completed so
far. It is intended for an independent reviewer who needs to understand what
was built, what was verified, which artifacts exist, and what remains before
the attribution and LLM-explanation experiments begin.

## Research objective

The project studies whether LLM-generated explanations of log anomalies are
faithful to the underlying detector and useful for recovering evidence from
the original logs.

The main protocol is based on:

- HDFS block-level anomaly detection using complete `BlockId` blocks;
- BGL evaluation using non-overlapping 60-second windows;
- a repository LogInsight LSTM detector;
- a LogRobust-style attention Bi-LSTM detector;
- Integrated Gradients as the only attribution method intended to be visible
  to the unblinded LLM;
- separate human-evidence-recovery and detector-faithfulness outcomes;
- leakage-controlled splits, preprocessing, retrieval, prompts, and evaluation.

The project deliberately separates the research benchmark from the existing
LogInsight dashboard/product code.

## Completed implementation

All implementation checklist items through Step 46, plus Steps 51–56 and 61–74, are marked
complete in `research/implementation_steps.txt`.

The Step 47A and 47B execution statuses are intentionally separate from
implementation completion. Their runners and recording machinery exist, but
the full Mistral generation and full repeated Qwen pilot are deferred. The
dated Mistral failure record is
`research/experiments/step47a_execution_failures.md`.

### Step 56 — Annotation ethics documentation

`research/experiments/annotation_ethics_documentation.md` records the
participant scope, consent requirements, compensation fields, data
minimization and anonymization controls, foreseeable risks, withdrawal and
incident procedures, artifact-retention rules, and the institutional
approval-or-exemption decision record. The documentation artifact is prepared,
but its labeling gate remains blocked until an actual institutional decision
and coordinator sign-off are recorded. No recruitment or core labeling should
start from the presence of this file alone.

### Step 61 — Parameterized TOST power analysis

`research/scripts/power_analysis_tost.py` implements the registered planning
calculation for the paired human-evidence-recovery comparison. With margin
`0.05`, alpha `0.05`, target power `0.80`, and paired SD values from `0.15`
through `0.35`, the required independent paired-item counts are `71`, `126`,
`197`, `283`, and `385`, respectively. The auditable output is stored in
`research/results/tables/step61_tost_power_analysis.json`; focused tests are in
`research/scripts/test_power_analysis_tost.py`. This is a planning artifact
only and does not generate LLM output or analyze pilot results.

### Step 62 — Registration-critical protocol freeze

`research/experiments/protocol_freeze_step62.json` freezes the definitions
needed before preregistration: complete HDFS `BlockId` blocks as the primary
unit, non-overlapping 60-second BGL windows as transfer units, top-5 evidence
events, position-aligned normal replacement with no fallback counterpart,
explicit empty-set F1 rules, separate detector false-positive reporting,
primary HDFS/LogInsight-LSTM/primary-LLM strata, a prediction-unit random
intercept for mixed-effects sensitivity analysis, and within-block attribution
permutation as the primary shuffle. The central configuration references this
artifact, and `validate_protocol_freeze.py` plus its focused tests validate the
freeze. Step 63 is complete: the protocol was archived at
https://osf.io/qrvuy/overview.

### Step 63 — OSF preregistration

The local preregistration packet is prepared in
`research/experiments/registration/step63_preregistration_packet.md` and the
exact-file manifest were submitted to OSF Registries and archived at
https://osf.io/qrvuy/overview. `research/scripts/prepare_registration.py`
records the OSF URL, timestamp, and frozen artifact hashes. Prior detector
development and limited model smoke tests remain disclosed as foreknowledge;
confirmatory LLM generation and human-outcome analysis proceed under the
registered protocol.

### Step 64 — Pluggable perturbation interface

`research/scripts/perturbation_interface.py` defines the common
`apply_perturbation` contract, immutable input context, named strategy
registry, output-shape validation, and per-input reference validation. The
existing single-event deletion operation is registered through this interface
and is now shared by the occlusion and leave-one-event-out diagnostics. The
replacement strategies and remaining masking, resampling, and matched-random
strategies use the same contract. Focused tests are in
`research/scripts/test_perturbation_interface.py`.

### Step 65 — Deletion perturbation

The registered `deletion` strategy removes all cited active event positions in
each prediction unit and returns one fixed-width sequence per unit. It records
the deleted positions and event IDs, preserves event order, and treats an
empty citation list as an explicit no-op. The existing
`single_event_deletion` strategy remains separate for the verification-only
occlusion and leave-one-event-out diagnostics.

### Step 66 — Normal-event replacement

The `normal_block_event_replacement` strategy aligns each cited anomalous
event to the same zero-based active position in its normal comparison block.
It replaces events in place, preserves sequence width and chronology, and
records unavailable counterparts without applying a fallback alignment.

### Step 67 — Unknown-token replacement

The `unknown_token_replacement` strategy replaces each cited active event with
the learned vocabulary unknown-event ID in place. It preserves sequence width
and positions, records original and replacement IDs, and rejects a padding ID
used as the unknown token.

### Step 68 — Normal-block resampling

The `normal_block_event_resampling` strategy samples donor events from the
paired normal block with a deterministic seed and writes them into cited
positions. It avoids padding donors, preserves sequence width and chronology,
uses replacement only when the donor pool is smaller than the cited set, and
records unavailable donors and sampling provenance.

### Step 69 — Parameter masking

The `parameter_masking` strategy keeps the detector event-ID sequence
unchanged while masking parsed dynamic parameters and their corresponding raw
line values for cited events. Event templates, positions, and sequence length
are preserved, and the result carries masked evidence plus parameter-count
metadata for parser and explanation checks.

### Step 70 — Matched random perturbations

The `apply_matched_random_perturbation` wrapper samples active positions with
the same per-unit budget as the cited set, then routes those positions through
the selected perturbation strategy. Selection uses the registered seed and
unit provenance, and records random positions, citation overlap, and seed
metadata for the matched control.

### Step 71 — Perturbation metadata

The common result validator now records per-output sequence-length change,
vocabulary change, chronology change, parser status and validity, and
out-of-distribution indicators with reasons. Evidence-bearing perturbations
are explicitly marked `preserved_unreparsed` until a parser re-run is added.

### Step 72 — Multi-strategy detector-faithfulness scoring

`research/scripts/detector_faithfulness.py` adds a detector-only scoring layer
for the deletion, replacement, unknown-token, resampling, masking, and matched
random strategies. It recomputes each detector score delta as
`detector_score_after - detector_score_before`, evaluates agreement with the
registered predicted effect using a numeric tolerance, and aggregates
counterfactual validity, mean score delta, predicted-direction accuracy, and
random-control delta both overall and by strategy. Human evidence labels and
evidence-recovery metrics are ignored and the report records
`human_evidence_metrics_used: false`, preserving the registered separation
between detector faithfulness and human evidence recovery. Focused tests are in
`research/scripts/test_detector_faithfulness.py`.

### Step 73 — Multi-strategy sufficiency checks

`research/scripts/sufficiency_scoring.py` evaluates whether cited events alone
preserve the detector decision after the complement is neutralized. It reports
the raw and normalized sufficiency gaps, groups results by claim and strategy,
and requires at least two distinct valid strategies including one core
non-deletion strategy. Deletion-only claims are excluded from sufficiency
acceptance, and agreement across at least two strategies is reported
separately. Human evidence labels are not read or combined with the detector
outcome. Focused tests are in `research/scripts/test_sufficiency_scoring.py`.

### Step 74 — Attribution-only reference and random floor

`research/scripts/reference_floor.py` defines matched interpretation levels
for every available detector, split, strategy, and event-budget stratum. The
attribution-only condition is summarized as an evidence-alignment reference,
explicitly not as an upper bound. Matched random-event selections form the
random floor and receive 95% confidence intervals. Target conditions report
their deltas from both references and a normalized position between them when
the two levels differ. Human evidence metrics are excluded. Focused tests are
in `research/scripts/test_reference_floor.py`.

### Step 51 — Evidence-relevance annotation guideline

`research/experiments/evidence_relevance_annotation_guideline.md` defines the
primary human evidence-relevance task. It specifies event-occurrence labels,
the distinction between direct anomaly evidence and routine context, treatment
of parameters, chronology, recovery and contrastive differences, explicit
uncertainty handling, the required audit fields, and adjudication rules. The
guideline keeps detector scores, attribution values, LLM outputs, and claim
support outside the evidence-relevance outcome.

### Step 52 — Claim-support annotation guideline

`research/experiments/claim_support_annotation_guideline.md` defines the
secondary claim-entailment task. It specifies `supported`, `unsupported`, and
`uncertain` labels; citation, scope, qualifier, contrastive, causal, and
detector-assertion rules; issue categories; claim-support denominator and
hallucination-rate calculations; and independent adjudication. It explicitly
keeps claim support separate from Step 51 event relevance, detector
faithfulness, and perturbation consistency.

### Step 53 — Annotation workflow

`research/experiments/annotation_workflow.md` defines the human-first
annotation workflow. It freezes guidelines and sampling before labeling,
creates blinded packets, assigns two independent annotators, locks evidence
relevance before LLM conditions, samples claim support separately, preserves
event- and claim-level disagreements for adjudication, and specifies agreement,
quality, audit, and correction rules. An optional LLM judge is exploratory only
and cannot replace human labels or adjudicate disagreements.

### Step 54 — Annotation packet and export format

`research/scripts/annotation_packet.py` builds the blinded evidence-relevance
packet and validates completed exports. It replaces source identifiers with
stable local event/line aliases, redacts common hosts, paths, block IDs, UUIDs,
and email addresses, enforces the configured 6,500-token deterministic context
cap, and records every omitted event when a packet is capped. Completed exports
must retain all event positions and required relevance fields while excluding
detector, attribution, condition, model, seed, and source-provenance metadata.
Focused tests are in `research/scripts/test_annotation_packet.py`.

### Step 55 — Annotation blinding and ID-memorization probe

`research/scripts/annotation_blinding.py` validates that annotation packets
contain no condition, detector, attribution, model, seed, or source-provenance
leakage. It also builds a deterministic balanced same/different probe using
fresh opaque aliases and equal-length event-only views, with a hidden answer
key and a scorer that reports accuracy against the 0.5 chance level. Focused
tests are in `research/scripts/test_annotation_blinding.py`.

### Project setup and protocol

Steps 1–7 established the research branch, audited the existing repository
branches, recorded local and historical Colab environment evidence, created the
research directory structure, added the central experiment configuration and
experiment ledger schema, and added deterministic seed utilities.

Important registered controls include:

- research branch: `research-implementation`;
- split seed: `2026`;
- detector seeds: `13`, `42`, `73`, `101`, and `202`;
- primary HDFS split: block-disjoint, label-stratified 80/10/10;
- temporal sensitivity splits for HDFS and BGL;
- training-only parser and vocabulary fitting;
- training-only normal-example retrieval;
- validation-only calibration and threshold selection.

### Dataset preparation and leakage controls

Steps 8–19 prepared and validated the datasets.

HDFS work included:

- verification of the official LogHub `HDFS_v1` source;
- inspection of the raw format and `BlockId` structure;
- construction of 575,061 labelled HDFS blocks from 11,175,629 log lines;
- preservation of raw lines, templates, parameters, event IDs, positions, and
  detector-compatible event sequences;
- creation and validation of block-disjoint primary splits;
- creation and validation of strict chronological sensitivity splits.

BGL work included:

- official download and checksum verification;
- removal of alert-label prefixes before parsing;
- retention of an audit artifact showing which prefixes were removed;
- construction of 27,670 non-overlapping 60-second evaluation windows;
- temporal split creation and validation.

The leakage audit passed for the primary and temporal split bundles. It found
no duplicate or misplaced prediction units, no BGL alert-prefix leakage in the
parser-safe input, and no violation of the configured training-only fitting
policies.

### Annotation-pool preparation

Step 20 implemented the annotation sampler and then generated the pilot pool
after detector predictions became available.

The completed HDFS pool contains:

- 240 annotation items;
- 120 items for the LogInsight LSTM;
- 120 items for the LogRobust-style detector;
- 239 unique HDFS blocks;
- deterministic sampling with seed `2026`;
- candidate metadata containing dataset, unit ID, label, sequence length,
  confidence, architecture, and detector-seed provenance.

Artifacts:

- `research/experiments/pilot/annotation_candidates.jsonl`;
- `research/experiments/pilot/annotation_pool.jsonl.gz`;
- `research/experiments/pilot/annotation_pool_manifest.json`.

The completed pool is HDFS-only. BGL has been prepared and validated, but it
has not been included in this completed pilot pool.

### Detector implementation and evaluation

Steps 21–28 implemented the detector pipeline.

The reusable LogInsight LSTM training script supports:

- frozen train/validation/test evidence splits;
- training-only event vocabulary fitting;
- an explicit unknown-event token;
- deterministic seeds and configuration recording;
- checkpoint saving and evaluation-only reuse;
- split-overlap rejection;
- shared prediction and metric serialization.

Calibration and threshold selection are separate validation-only steps:

- temperature scaling is fitted on validation predictions;
- thresholds are selected using validation F1;
- test labels are not used to select either value.

The LogRobust-style attention Bi-LSTM implementation includes a trainable event
embedding, masked additive temporal attention, a bidirectional LSTM, and a
dense classifier. Its deviations from the published method are documented,
including the shared trainable event embedding, the HDFS block protocol, and
post-padding for TensorFlow Metal/cuDNN compatibility.

The shared detector interface standardizes:

- input validation;
- probability validation;
- binary metrics;
- prediction serialization;
- model adapter behavior.

Architecture-specific model construction and attention behavior remain in the
individual detector modules.

### Multi-seed detector training

Step 29 completed the registered five-seed experiment for both detectors:

- LogInsight LSTM: seeds `13`, `42`, `73`, `101`, `202`;
- LogRobust-style attention Bi-LSTM: seeds `13`, `42`, `73`, `101`, `202`.

The LSTM runs were completed locally. The missing LogRobust runs were trained
on Google Colab and transferred back to the local repository. The resumable
orchestrator validated all ten architecture/seed targets and recorded the
result in:

- `research/results/tables/multiseed_training_manifest.json`.

The individual model checkpoints, histories, prediction files, and manifests
are under:

- `research/models/checkpoints/loginsight_lstm/`;
- `research/models/checkpoints/logrobust_attention_bilstm/`.

### Inference caching and batching

Step 30 added reusable inference infrastructure to
`research/scripts/detector_interface.py`.

`predict_probabilities_batched` performs bounded, ordered model evaluation.
`DetectorInferenceCache`:

- namespaces entries by detector and model identity;
- includes input dtype, shape, and bytes in the cache key;
- coalesces duplicate rows within a request;
- reuses probabilities across requests;
- reports requested, cache-hit, deduplicated, and model-evaluated rows;
- stores detector probabilities only, not attribution results.

Focused tests are in
`research/scripts/test_detector_inference_cache.py`.

### Integrated Gradients for the LogInsight LSTM

Step 31 implements the prompt-visible attribution method in
`research/scripts/integrated_gradients_lstm.py`. Since the detector input is
integer event IDs, gradients are integrated in embedding space from an
all-padding baseline. Embedding-dimension scores are summed into event-level
scores, while padding positions are retained for alignment but excluded from
ranking.

Each attribution record includes detector and baseline probabilities, signed
and absolute event scores, event ranks, input/sequence positions, and a
completeness error. The default configuration uses 64 trapezoidal integration
steps and writes compressed JSONL output plus a manifest under
`research/models/attributions/loginsight_lstm/`.

### Integrated Gradients for the LogRobust-style Bi-LSTM

Step 32 adds `research/scripts/integrated_gradients_logrobust.py`. It uses the
same event-level output contract as the LSTM implementation, but explicitly
passes the valid-event mask through the bidirectional LSTM and custom temporal
attention layer. The all-padding embedding baseline preserves the input mask
along the path, and the mask policy plus completeness diagnostics are recorded
in the output artifacts.

### Integrated Gradients validation

Step 33 adds `research/scripts/validate_integrated_gradients.py`. It validates
both detector attribution artifacts without TensorFlow, checking probability
arithmetic, completeness thresholds, padding and event-index alignment,
ranking, duplicate units, and manifest counts. It exits nonzero when an
artifact fails the default completeness threshold of `0.05`.

### Verification-only occlusion scores

Step 34 adds `research/scripts/occlusion_scores.py` for both detector
architectures. It performs single-event deletion, re-pads each sequence
according to the configured policy, and records `p(original) - p(occluded)`
in a separate compressed JSONL artifact and manifest. The scorer uses the
Step 30 inference cache and explicitly marks the results
`verification_only: true` and `prompt_visible: false`, so these values cannot
be treated as LLM-visible Integrated Gradients. Focused coverage is in
`research/scripts/test_occlusion_scores.py`.

### Verification-only leave-one-event-out scores

Step 35 adds `research/scripts/leave_one_event_out_scores.py`. It holds out
each retained event using the matched single-event deletion/re-padding rule,
records `p(original) - p(held_out)` in a separately named compressed JSONL
artifact and manifest, and reuses the Step 30 inference cache. The output is
explicitly marked `verification_only: true` and `prompt_visible: false` and
is never intended for LLM prompts or the attribution-only condition. Focused
coverage is in `research/scripts/test_leave_one_event_out_scores.py`.

### Attribution separation

Step 36 adds `research/scripts/prompt_attribution_boundary.py`, the strict
condition-aware boundary for research prompt construction. It loads the
registered visible/forbidden fields, permits only the selected condition's
allowlist, and rejects occlusion, leave-one-event-out, verification-only
metadata, and other unsupported fields before prompt rendering. Integrated
Gradients is accepted only in conditions that explicitly allow it. Focused
coverage is in `research/scripts/test_prompt_attribution_boundary.py`.

### Attribution-only baseline

Step 37 adds `research/scripts/attribution_only_baseline.py`. Given a
serialized Integrated Gradients artifact and an explicit top-k value, it
selects ranked non-padding events and writes a separate no-LLM baseline
artifact with source attribution provenance. It does not render prompts,
invoke an LLM, or copy the complete attribution vector into the baseline.
Focused coverage is in `research/scripts/test_attribution_only_baseline.py`.

### Matched random-event baseline

Step 38 adds `research/scripts/random_event_baseline.py`. It uses the same
explicit top-k budget as the attribution-only baseline, samples non-padding
events without replacement with a stable per-unit seed derived from the
registered perturbation seed, and excludes Integrated Gradients scores and
ranks from the output. It does not render prompts or invoke an LLM. Focused
coverage is in `research/scripts/test_random_event_baseline.py`.

### Separate attribution-vector storage

Step 39 adds `research/scripts/validate_attribution_storage.py`. It validates
that Integrated Gradients, event occlusion, and leave-one-event-out vectors
remain in separate method-specific files, hashes them, enforces unique units
and prompt/verification flags, rejects mixed files and reused paths, and writes
only a registry manifest. It never merges vectors, preventing accidental
circularity. Focused coverage is in
`research/scripts/test_validate_attribution_storage.py`.

### R1 — Exact event-sequence overlap audit

Reviewer gate R1 is complete. The audit in
`research/scripts/sequence_overlap_audit.py` uses exact ordered, unpadded
event-ID sequences. The primary HDFS split contains 17,884 unique training
sequences and 4,564 unique test sequences; 3,121 test sequence types are also
present in training. Therefore 1,459 of 57,506 test blocks (2.54%) form a
novel-sequence subset. Novel-subset metrics are recorded for the available
LSTM seed-13 and LogRobust-inspired seed-202 prediction artifacts.

The 240-item annotation pool contains 239 unique blocks and 102 unique event
sequences. The full report is
`research/results/tables/sequence_overlap_audit.json`, with focused tests in
`research/scripts/test_sequence_overlap_audit.py`.

### R2 — Shortcut baseline audit

R2 is complete. A training-vocabulary bag-of-event-template-count logistic
regression reached test F1 `0.9811` and average precision `0.9824`. A
sequence-length-only logistic regression reached test F1 `0.1032` and average
precision `0.3872`. The available detector artifacts reached F1 `0.9514` for
LogInsight LSTM seed 13 and `0.9979` for LogRobust-inspired seed 202 at their
stored thresholds.

The event-count baseline is close to the LSTM detector, so event-template
composition is a meaningful shortcut in this split. Later explanation
evaluation must not interpret detector performance alone as proof that the
LLM recovered uniquely relevant evidence. The full report is
`research/results/tables/shortcut_baseline_audit.json`, with focused tests in
`research/scripts/test_shortcut_baseline_audit.py`.

### R3 — Primary explanation seed registration

R3 is complete. `research/scripts/select_primary_explanation_seeds.py` ranks
the ten completed Step 29 run manifests by validation F1, with lower
validation loss and lower seed as deterministic tie-breakers. Selection was
recorded before explanation generation and does not use test metrics or
attribution outputs. The primary seeds are LogInsight LSTM seed 13
(validation F1 `0.990280`) and LogRobust-style attention Bi-LSTM seed 202
(validation F1 `0.997330`). The other four seeds per architecture are
registered as secondary attribution-stability analyses. The complete report is
`research/results/tables/primary_explanation_seed_selection.json`, with
focused tests in `research/scripts/test_select_primary_explanation_seeds.py`.

### Step 40 — Structured explanation schema

Step 40 is complete. `research/experiments/llm_explanation_schema.json` now
defines the strict LLM output contract: answer status, summary, typed claims,
event and line citations, per-claim and overall confidence, and explicit
abstention. `research/scripts/explanation_schema.py` validates decoded
payloads and enforces citation and abstention consistency. Step 48 is handled
by `research/scripts/parse_explanation_responses.py`, which preserves raw
responses and classifies invalid JSON, missing citations, malformed claims,
and abstentions. Focused tests are in
`research/scripts/test_parse_explanation_responses.py`.

### Step 41 — Training-only contrastive retrieval and blinded prompt

Step 41 is complete. `research/scripts/training_only_retrieval.py` fits an
`event_tfidf_cosine` index on normal training blocks only, uses deterministic
similarity/tie-breaking, and records source hashes, candidate provenance, and
the fact that query labels and attribution were not used. Zero-overlap queries
use the documented lowest-ID fallback. `research/scripts/blinded_prompt.py`
passes only anomalous evidence, a normal comparison block, and basic detector
output through the existing prompt boundary; unit labels and attribution
values are not prompt-visible. The method remains provisional until
registration freezes it. Focused tests are in
`research/scripts/test_training_only_retrieval.py` and
`research/scripts/test_blinded_prompt.py`.

### Step 42 — Unblinded prompt and contrastive factor

Step 42 is complete. `research/scripts/contrastive_prompt.py` implements the
2×3 factor of normal comparison present/absent crossed with attribution
none/Integrated Gradients/shuffled Integrated Gradients. The six cells map to
strict prompt conditions and require or reject their corresponding payloads
before rendering. The factor is provisional until registration freezes it; no
LLM calls are made. Focused tests are in
`research/scripts/test_contrastive_prompt.py`.

### Step 43 — Shuffled Integrated Gradients prompt

Step 43 is complete. `research/scripts/shuffled_integrated_gradients.py`
supports deterministic within-block attribution permutation and cross-block
donor values with equal active-event counts. It preserves event positions,
recomputes ranks, records source/donor provenance, and emits a prompt-safe
payload without padding or artifact identifiers. The primary shuffle type is
provisional until Step 62, and no LLM calls are made. Focused tests are in
`research/scripts/test_shuffled_integrated_gradients.py`.

### Step 44 — No-detector-context prompt

Step 44 is complete. `research/scripts/no_detector_context_prompt.py` builds
the registered `no_detector_context` condition from anomalous evidence only.
It excludes detector output, normal comparisons, attribution values, labels,
and storage identifiers from the prompt payload, while retaining event and
line citations needed by the structured explanation schema. The prompt is
deterministic and uses template ID `no_detector_context_v1`. Focused tests are
in `research/scripts/test_no_detector_context_prompt.py`.

### Step 45 — LLM model selection

Step 45 is complete at the model-family and access-mode level. The selected
pair is `mistralai/Mistral-7B-Instruct-v0.3` and
`Qwen/Qwen2.5-7B-Instruct` as publicly downloadable open-weight models run
locally or in Colab, with Colab GPU as the default target. Mistral is the
active replacement for the historical GPT-OSS slot because it loaded on the
free Colab T4 while GPT-OSS 20B did not fit.
The Mistral and Qwen licenses must be retained with the experiment record.
Step 46 then recorded the exact revisions, Colab runtime, decoding settings,
and passed synthetic smoke tests. The selection is recorded in
`research/experiments/experiment_config.json` and
`research/experiments/llm_model_selection.md`. Exact provider/checkpoint
revisions, runtime hardware, and decoding settings were captured in Step 46
before pilot generation.

## Main artifacts and documentation

### Documentation coverage audit

The main implementation record now contains an individual section for every
completed checklist item from Steps 1–46. The knowledge graph has a matching
architecture/provenance entry for those steps, and this summary provides the
shorter independent-review version. Future checklist items remain explicitly
pending; the documentation does not infer completion of the LLM, annotation,
ethics, perturbation, or pilot-experiment phases.

The primary implementation record is:

- `research/steps_1_to_20_documentation.md` — now updated to document Steps
  1–45.

Persistent project memory is maintained in:

- `research/project_knowledge_graph.md`.

The implementation checklist is:

- `research/implementation_steps.txt`.

Other important documentation includes:

- `research/research_proposal.txt` — research questions, conditions, outcomes,
  perturbations, statistics, and ethics/privacy constraints;
- `research/colab_environment_record.txt` — historical Colab evidence and
  environment-recording requirements;
- `research/experiments/logrobust_implementation_deviations.md` — deviations
  from the published LogRobust method;
- `research/scripts/README.md` — script-by-script usage and responsibilities;
- dataset, model, prompt, experiment, and results README files under
  `research/`.

## Verification status

The complete research test suite currently passes:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 -m unittest discover -s research/scripts -p 'test_*.py'
```

Result: 226 tests passed.

The tests cover preprocessing, split validation, leakage checks, vocabulary
behavior, detector models, calibration, threshold selection, baseline
reporting, the shared detector interface, multi-seed orchestration, inference
caching, Integrated Gradients, attribution validation, verification-only
perturbation diagnostics, prompt-boundary enforcement, matched baselines, and
separate attribution-vector storage. The R4 tests also cover audit hashing,
two-architecture coverage, step-count comparison, logit-space separation, and
randomized-model sanity reporting.

## Important caveats for review

1. The project has both local and Colab provenance. The LogRobust multi-seed
   runs were completed on Colab, so their environment must not be silently
   treated as identical to the local environment.

2. The historical Colab record does not contain every runtime version or GPU
   identifier. The limitation is recorded in
   `research/colab_environment_record.txt`.

3. The LogRobust implementation is a documented research-compatible variant,
   not a byte-for-byte reproduction of every published training detail.

4. The current completed annotation pool is HDFS-only and uses the available
   validated detector prediction artifacts. BGL preparation is complete, but a
   BGL annotation pool has not been finalized here.

5. The shared prediction files and model artifacts must continue to preserve
   detector architecture and seed provenance. Reviewers should check the
   manifests before treating aggregate results as a single definitive model.

6. The annotation pool is a pilot-scale pool. It is not yet the final human
   study result, and no claim about LLM explanation quality should be made from
   the detector metrics alone.

7. The completed R4 audit finds that LogInsight LSTM seed 13 is stable between
   16 and 64 integration steps, but LogRobust-inspired seed 202 is not stable
   on the 16-block audit sample and has large completeness errors. The
   all-padding baseline gives a near-certain anomaly probability for both
   models. The randomized-weight sanity comparison passes, but it does not
   resolve those convergence and baseline limitations. The full report is
   `research/results/tables/integrated_gradients_r4_audit.json`.

8. R5 verification tooling is implemented in
   `research/scripts/verify_colab_transfer.py`, but the six-split local
   inference command has not yet completed in this environment. It must pass
   at the registered `1e-6` tolerance before R5 is marked complete.

## Remaining planned work

Step 47A has a completed Mistral synthetic smoke test and a resumable
local/Colab runner in `research/scripts/run_mistral_pilot.py`, but its full run
is blocked. The historical API attempts encountered DNS and HTTP 403 failures;
the Colab T4 full attempt reached 3 complete and 1 failed call before CUDA
out-of-memory; and the local MPS one-record attempt stalled and was stopped
with 0 complete calls. The Mistral runner preserves resumable JSONL output and
live status, but no full-run result exists. The matched Step 47B Qwen smoke
test is also complete: 18/18
calls returned valid JSON and valid schema responses on a free Colab T4. Its
report is `research/results/tables/step47b_qwen_18_smoke_test.json` and its
raw responses are in `research/results/tables/step47b_qwen_18_smoke_responses.jsonl`.
The remaining planned work includes:

- selecting and validating a runtime that can complete Mistral generation;
- downloading/preserving all remote partial outputs before runtime shutdown;
- the full repeated Qwen pilot after the smoke gate;
- structured blinded, unblinded, shuffled-attribution, and no-detector-context
  prompts;
- full pilot generation where the required attribution artifacts are available;
- perturbation experiments, annotation QA, statistical analysis, and final
  reporting.

## Suggested questions for an independent reviewer

- Are the HDFS and BGL evaluation units appropriate for the stated research
  question?
- Are the split, parser, vocabulary, retrieval, calibration, and threshold
  controls sufficient to prevent leakage?
- Is the LogRobust-style deviation record detailed enough for reproducibility?
- Is the Colab-to-local artifact transfer sufficiently documented and
  verifiable?
- Does the completed annotation pool support the intended comparison, or does
  it need a BGL pool or additional stratification?
- Are the planned attribution and LLM conditions capable of separating detector
  faithfulness from general explanation quality?
- Which results should be treated as provisional until Steps 34 onward are
  complete?
