# Research scripts

Place deterministic preprocessing, training, attribution, perturbation,
evaluation, and validation scripts here. Scripts should accept explicit input
and output paths and record the configuration/run ID they use.

`power_analysis_tost.py` performs the Step 61 planning calculation for the
paired TOST primary endpoint. It uses the proposal's conservative two-sided
95% confidence-interval convention under a true difference of zero and
reports required independent paired items for a parameterized set of paired
standard deviations:

    python3 research/scripts/power_analysis_tost.py

The default sensitivity range is SD 0.15 through 0.35. The generated report
is written to `research/results/tables/step61_tost_power_analysis.json`.

`seed_utils.py` is the shared seed implementation. Call
`set_global_seed(seed, deterministic=True)` at process startup, pass explicit
`random_state` values to scikit-learn, and use `derive_seed` or
`make_numpy_rng` for named sub-runs. Record the returned seed report in the
experiment ledger. LLM providers may not honor a seed, so repeated-generation
indices and the provider/model configuration must also be recorded.

`build_hdfs_blocks.py` reads the Drain structured HDFS CSV, extracts BlockIds
from `Content`, sorts events chronologically by `LineId`, joins the official labels, and
writes one compressed JSONL record per complete block. Use `--max-rows` only for
smoke tests; the full dataset must be built without that option. The normal
run displays a dependency-free progress bar for both reading and writing:

    python3 research/scripts/build_hdfs_blocks.py

`build_hdfs_block_evidence.py` builds the explanation-ready companion artifact.
It preserves each block's event-ID sequence and stores, for every event, the
exact raw line, EventTemplate, parsed ParameterList, source LineId, EventId,
and zero-based position within the block. It also writes a schema/provenance
manifest. The original raw log must be aligned one-to-one with the structured
CSV; the default paths are the repository's HDFS files:

    python3 research/scripts/build_hdfs_block_evidence.py

Use `--max-rows` only for smoke tests. Do not use it for the final artifact.

`download_verify_bgl.py` downloads the official LogHub BGL archive, verifies
the published archive MD5 and size, extracts `BGL.log`, verifies the official
line count, and records archive/raw MD5 and SHA-256 checksums in a manifest.
The download is intentionally run from the user's terminal because it needs
network access:

    python3 research/scripts/download_verify_bgl.py

`strip_bgl_label_prefix.py` removes only the first BGL label token from each
line before parsing. It writes parser-safe `BGL_clean.log` and a separate
compressed audit stream containing the original line, removed prefix, and
derived alert indicator. The audit stream must not be sent to the detector or
LLM. The full run displays a progress bar:

    python3 research/scripts/strip_bgl_label_prefix.py

`build_bgl_windows.py` freezes the BGL transfer unit as non-overlapping
60-second windows anchored to Unix epoch time. A window is anomalous when any
source line in it had an alert prefix. It stores inclusive clean-log line
ranges, so it does not duplicate the large cleaned log:

    python3 research/scripts/build_bgl_windows.py

`validate_preprocessing.py` runs the shared artifact checks for both datasets:
HDFS evidence ordering, event alignment, labels, lengths, and raw evidence;
and BGL window ordering, disjoint line ranges, labels, durations, and lengths.
It also compares observed counts with the recorded manifests:

    python3 research/scripts/validate_preprocessing.py

`split_hdfs_blocks.py` creates the registered HDFS train/validation/test
partitions. It stratifies by Normal/Anomaly label using seed `2026`, assigns
each BlockId exactly once, preserves chronological order inside each split,
and records assignments, counts, and hashes in a manifest:

    python3 research/scripts/split_hdfs_blocks.py

`create_temporal_splits.py` creates a separate chronological sensitivity split
for both datasets. It orders HDFS blocks by the first timestamp in their raw
evidence and BGL windows by `WindowStartEpoch`. A unit is written only when
its complete interval fits inside one temporal partition; HDFS blocks crossing
a boundary are recorded as excluded instead of leaking future evidence into
an earlier split. The command writes per-dataset split files, assignment
streams, and `temporal_split_manifest.json`. It includes progress bars for
the metadata scan and split-file writing:

    python3 research/scripts/create_temporal_splits.py

Use `--force` only when intentionally regenerating the temporal artifacts.

`validate_leakage.py` audits the primary and temporal split bundles for
duplicate or misplaced prediction units, checks that configured normal-block
retrieval is training-only, verifies the parser/vocabulary fit policy, and
checks that every BGL parser input line begins with an epoch timestamp rather
than an alert label. It writes a leakage-audit manifest. Retrieval and parser
runtime ledgers can be supplied later with `--retrieval-ledger` and
`--parser-manifest`; until then they are explicitly recorded as pending:

    python3 research/scripts/validate_leakage.py

`build_annotation_candidates.py` converts one held-out prediction file per
detector architecture into the registered candidate metadata and preserves the
detector seed and probability as provenance. It rejects duplicate
dataset/unit/architecture records:

    python3 research/scripts/build_annotation_candidates.py \
        --prediction-file research/models/checkpoints/loginsight_lstm/predictions_test.jsonl.gz \
        --prediction-file research/models/checkpoints/logrobust_attention_bilstm/predictions_test.jsonl.gz

`sample_annotation_pool.py` draws the registered 200--300-item annotation
pool after detector inference has produced candidate metadata. Each candidate
must include `dataset`, `unit_id`, `split`, `label`, `sequence_length`,
`confidence`, and `architecture`. The sampler uses the registered test split,
length bins, confidence bins, dataset, label, and architecture; it uses seed
`2026` by default and refuses missing detector metadata:

    python3 research/scripts/sample_annotation_pool.py \
        --candidate-metadata research/experiments/pilot/annotation_candidates.jsonl

The default target is 240 annotation items. It writes the compressed pool to
`research/experiments/pilot/annotation_pool.jsonl.gz` and its provenance to
`research/experiments/pilot/annotation_pool_manifest.json`. Because detector
confidence and architecture are required strata, the final pool cannot be
drawn honestly until the research detector outputs exist.

`train_loginsight_lstm.py` trains and evaluates the leakage-free research
LogInsight LSTM. It reads the HDFS train/validation/test evidence splits,
consumes a vocabulary fitted on the training split only, and refuses duplicate
prediction units across splits. The default model mirrors the repository
notebook: embedding, two LSTM layers, batch normalization, dropout, and a
sigmoid output. The script records the seed report, explicit configuration,
class weights, best/final checkpoints, split-wise predictions, and fixed-
threshold evaluation metrics. It also includes an evaluation-only mode for
reusing a checkpoint in later calibration and attribution steps.

First create the training-only vocabulary:

    python3 research/scripts/build_event_vocabulary.py \
      --training-sequences research/data/processed/hdfs_block_evidence_train.jsonl.gz \
      --output research/models/vocabularies/event_vocabulary.json

Then run training explicitly:

    python3 research/scripts/train_loginsight_lstm.py \
      --config research/experiments/lstm_training_config.json

To evaluate an existing checkpoint without retraining:

    python3 research/scripts/train_loginsight_lstm.py \
      --config research/experiments/lstm_training_config.json \
      --evaluate-only \
      --checkpoint research/models/checkpoints/loginsight_lstm/best_seed_13.keras

`train_logrobust_attention_bilstm.py` trains the second detector: a
LogRobust-style masked additive-attention Bi-LSTM classifier. It uses the same
training-only vocabulary and split files as the LogInsight LSTM, records the
attention architecture in its manifest, and supports evaluation-only
checkpoint reuse. The external semantic-embedding choice in the published
method is intentionally replaced by the shared trainable event vocabulary;
the deviation is recorded for Step 27.

Run training explicitly:

    python3 research/scripts/train_logrobust_attention_bilstm.py \
      --config research/experiments/logrobust_attention_bilstm_training_config.json

To evaluate an existing checkpoint without retraining:

    python3 research/scripts/train_logrobust_attention_bilstm.py \
      --config research/experiments/logrobust_attention_bilstm_training_config.json \
      --evaluate-only \
      --checkpoint research/models/checkpoints/logrobust_attention_bilstm/best_seed_13.keras

The complete Step 27 deviation record is maintained at:

    research/experiments/logrobust_implementation_deviations.json
    research/experiments/logrobust_implementation_deviations.md

`detector_interface.py` defines the shared detector contract used by both
trainers. It validates integer sequence batches and unique unit metadata,
standardizes Keras probability output, computes the common binary metrics, and
writes the common prediction schema consumed by calibration and threshold
selection. Model construction remains architecture-specific.

Step 30 adds `predict_probabilities_batched` and `DetectorInferenceCache` to
the same module. The batched predictor evaluates bounded chunks in input order.
The cache deduplicates identical sequence requests, reuses probabilities across
calls, and namespaces keys by detector and model identity so different seeds or
checkpoints cannot collide:

    cache = DetectorInferenceCache("loginsight_lstm", "checkpoint-sha256")
    probabilities = cache.predict(model, perturbed_inputs, batch_size=256)

The cache is intentionally in-process and stores probabilities only; labels,
thresholds, and attribution metadata remain outside the inference layer.

`integrated_gradients_lstm.py` computes the Step 31 prompt-visible attribution
for the LogInsight LSTM. It integrates gradients in embedding space from an
all-padding baseline, sums embedding-dimension scores to event positions, and
records signed/absolute scores, ranks, sequence alignment, detector
probabilities, and completeness diagnostics. Padding positions remain in the
artifact for alignment but are excluded from event ranking. The default is 64
integration steps and the output is a compressed JSONL file plus a manifest:

    python3 research/scripts/integrated_gradients_lstm.py \
      --config research/experiments/lstm_training_config.json \
      --checkpoint research/models/checkpoints/loginsight_lstm/best_seed_13.keras \
      --split test

The default output is under
`research/models/attributions/loginsight_lstm/`. This artifact contains only
Integrated Gradients values; occlusion and leave-one-event-out diagnostics are
stored separately by their verification scripts and are not mixed into this
output.

`integrated_gradients_logrobust.py` applies the same output contract to the
LogRobust-style attention Bi-LSTM. It integrates through the trainable event
embedding and passes the detector's valid-event timestep mask explicitly to
the bidirectional LSTM and temporal-attention layer. Because this detector
uses `mask_zero=true`, the input mask is held fixed along the path; the
baseline therefore records both the PAD embedding and its mask policy:

    python3 research/scripts/integrated_gradients_logrobust.py \
      --config research/experiments/logrobust_attention_bilstm_training_config.json \
      --checkpoint research/models/checkpoints/logrobust_attention_bilstm/best_seed_13.keras \
      --split test

The default output is under
`research/models/attributions/logrobust_attention_bilstm/`.

`validate_integrated_gradients.py` validates the serialized Integrated
Gradients artifacts without importing TensorFlow. It checks probability and
attribution arithmetic, completeness thresholds, padding and event-index
alignment, ranking, duplicate units, and manifest counts. Repeat
`--attribution-file` to validate both detector outputs together:

    python3 research/scripts/validate_integrated_gradients.py \
      --attribution-file research/models/attributions/loginsight_lstm/seed_13_test_integrated_gradients.jsonl.gz \
      --attribution-file research/models/attributions/logrobust_attention_bilstm/seed_13_test_integrated_gradients.jsonl.gz

The default maximum allowed completeness error is `0.05`. The command exits
nonzero when any artifact fails validation, unless `--allow-failures` is used
for diagnostic reporting only.

`perturbation_interface.py` implements Step 64's shared perturbation
interface. `apply_perturbation` validates the integer sequence matrix, padding
policy, strategy output width, and per-input references for every registered
strategy. Strategies are registered by name with
`register_perturbation_strategy`; the existing `single_event_deletion` path is
registered as the first strategy and is used by both occlusion and
leave-one-event-out diagnostics. Later replacement, masking, resampling, and
matched-random strategies must use the same function rather than defining
separate input/output contracts. Focused tests are in
`test_perturbation_interface.py`.

Step 65 adds the registered `deletion` strategy. It accepts one cited-position
list per prediction unit, removes all cited active events together, preserves
the configured sequence width and event order, and returns one perturbed
sequence per input unit. An empty cited list is an explicit no-op. The
single-event diagnostic remains separately registered as
`single_event_deletion` for the Step 34/35 verification paths.

Step 66 adds `normal_block_event_replacement`. It aligns each cited anomalous
event to the same zero-based active position in its normal comparison block,
replaces the token in place, and preserves sequence length and chronology. If
the normal block has no active counterpart, the target is left unchanged and
the reference records `replacement_available: false`; no fallback alignment
is performed.

Step 67 adds `unknown_token_replacement`. It replaces each cited active event
with the vocabulary's learned unknown-event ID in place, preserves sequence
width and positions, and rejects attempts to use the padding ID as the unknown
token. The replacement records the original and replacement event IDs.

Step 68 adds `normal_block_event_resampling`. It samples donor events from the
paired normal block with a deterministic registered seed and writes them into
the cited positions while preserving sequence width and chronology. Sampling
is without replacement when the normal pool is large enough, switches to
replacement when necessary, and records unavailable donors rather than using
padding as a donor event.

Step 69 adds `parameter_masking`. It keeps the detector event-ID sequence
unchanged while masking the parsed dynamic parameters and corresponding raw
line values for cited events. Event templates, positions, and sequence length
remain unchanged; the result carries the masked evidence and parameter-count
metadata for parser and explanation checks.

Step 70 adds `apply_matched_random_perturbation`. It samples active positions
with the same per-unit budget as the cited set, derives the selection from the
registered seed and unit provenance, and sends those positions through the
chosen deletion, replacement, unknown-token, resampling, or masking strategy.
The result records the random positions, citation overlap, and seed so the
control is auditable.

Step 71 adds common metadata recording to every perturbation result. Each
result now records per-output sequence-length change, vocabulary change,
chronology change, parser status/validity, and out-of-distribution indicators
with reasons. Evidence-bearing perturbations are marked
`preserved_unreparsed` until a parser re-run is performed.

`detector_faithfulness.py` implements Step 72's detector-only scoring layer.
`score_perturbation_record` recomputes the detector score delta as
`after - before`, classifies the observed direction with a registered numeric
tolerance, and compares it with the explanation's predicted effect. The
`score_detector_faithfulness` report aggregates validity, mean score delta,
predicted-direction accuracy, and random-control delta overall and by
perturbation strategy. It ignores human evidence labels and emits an explicit
`human_evidence_metrics_used: false` boundary. A report can be generated
from a JSON array, JSONL stream, or object containing `perturbations`:

    python3 research/scripts/detector_faithfulness.py \
      --input research/results/tables/perturbation_scores.jsonl \
      --output research/results/tables/detector_faithfulness.json

Focused tests are in `test_detector_faithfulness.py`.

`sufficiency_scoring.py` implements Step 73. It scores the registered
sufficiency gap, `p(original) - p(retained)`, and its normalized form
`1 - p(retained) / p(original)` when the original score is positive. Results
are grouped by claim and strategy. A claim is eligible only when at least two
distinct strategies are valid and at least one is a core non-deletion strategy
(normal replacement, unknown-token replacement, normal resampling, or
parameter masking). Deletion-only claims are explicitly excluded, and the
report records whether conclusions agree across at least two strategies. It
does not read human evidence labels:

    python3 research/scripts/sufficiency_scoring.py \
      --input research/results/tables/sufficiency_scores.jsonl \
      --output research/results/tables/sufficiency_report.json

Focused tests are in `test_sufficiency_scoring.py`.

`reference_floor.py` implements Step 74's interpretation references. It keeps
the attribution-only condition as an evidence-alignment reference (never a
theoretical upper bound) and summarizes matched random-event selections as a
random floor with 95% confidence intervals. Results are stratified by dataset,
detector architecture/seed, split, perturbation strategy, and event budget
when those fields are present. Each target condition reports its difference
from both reference levels and its normalized position between them:

    python3 research/scripts/reference_floor.py \
      --input research/results/tables/faithfulness_condition_scores.jsonl \
      --output research/results/tables/reference_floor_report.json \
      --metric-field consistency_score

The report does not use human evidence metrics. Focused tests are in
`test_reference_floor.py`.

`annotation_packet.py` implements Step 54's blinded annotation packet and
export boundary. `build_annotation_packet` replaces source unit, event, and
line identifiers with study-local aliases, redacts common block IDs, hosts,
paths, UUIDs, and email addresses, and applies the registered deterministic
6,500-token context cap. If events do not fit, the packet records omitted
positions and aliases and marks itself `annotation_eligible: false`; no event
is silently truncated. `validate_annotation_export` checks the required
position, alias, relevance, role, reason, and contrastive fields and rejects
detector, attribution, condition, model, seed, and source-provenance fields.
Focused tests are in `test_annotation_packet.py`.

`annotation_blinding.py` implements Step 55. `validate_blinded_packet` rejects
condition, detector, attribution, model, seed, and source-provenance leakage
before a packet enters annotation. `build_id_memorization_probe` creates
balanced same/different pairs with equal event counts, fresh opaque aliases,
and no log content; the hidden answer key stays with the coordinator.
`score_id_memorization_probe` reports answered items, accuracy, invalid
responses, and an above-chance indicator against the 0.5 chance level. Focused
tests are in `test_annotation_blinding.py`.

`occlusion_scores.py` computes the Step 34 verification-only diagnostic for
both detectors. It deletes each retained event one at a time, re-pads the
remaining sequence according to the configured padding policy, and records
`p(original) - p(occluded)`. The output is a separate compressed JSONL file
and manifest; it is marked `verification_only` and `prompt_visible: false`
and must never be passed to an LLM prompt. The Step 30 inference cache is
used for duplicate perturbations:

    python3 research/scripts/occlusion_scores.py \
      --config research/experiments/lstm_training_config.json \
      --checkpoint research/models/checkpoints/loginsight_lstm/best_seed_13.keras \
      --split test

    python3 research/scripts/occlusion_scores.py \
      --config research/experiments/logrobust_attention_bilstm_training_config.json \
      --checkpoint research/models/checkpoints/logrobust_attention_bilstm/best_seed_13.keras \
      --split test

The focused tests are in `test_occlusion_scores.py`. Production runs are
intentionally explicit because the workload scales with the number of retained
events.

`leave_one_event_out_scores.py` computes the Step 35 verification-only
leave-one-event-out diagnostic. Under the current registered protocol it uses
the matched single-event deletion/re-padding operation, but writes a distinct
artifact and records `p(original) - p(held_out)` as
`leave_one_event_out_score`. Results are marked `verification_only` and
`prompt_visible: false`; they must not be passed to an LLM prompt or the
attribution-only condition. The Step 30 inference cache is reused:

    python3 research/scripts/leave_one_event_out_scores.py \
      --config research/experiments/lstm_training_config.json \
      --checkpoint research/models/checkpoints/loginsight_lstm/best_seed_13.keras \
      --split test

    python3 research/scripts/leave_one_event_out_scores.py \
      --config research/experiments/logrobust_attention_bilstm_training_config.json \
      --checkpoint research/models/checkpoints/logrobust_attention_bilstm/best_seed_13.keras \
      --split test

The focused tests are in `test_leave_one_event_out_scores.py`. Production
runs are intentionally explicit because the workload scales with retained
events. Step 36 enforces the prompt-boundary separation in the consumer path.

`prompt_attribution_boundary.py` implements Step 36. Every research prompt
consumer must call `build_prompt_payload` or `render_prompt_payload` with a
registered condition. The boundary reads the condition allowlist from
`research/experiments/experiment_config.json`, rejects unsupported fields, and
recursively rejects occlusion, leave-one-event-out, `verification_only`, and
`prompt_visible: false` fields. Only Integrated Gradients is allowed where the
selected condition explicitly permits it:

    from research.scripts.prompt_attribution_boundary import render_prompt_payload

    prompt_text = render_prompt_payload(
        "unblinded",
        {
            "anomalous_block_evidence": evidence,
            "normal_comparison_block": normal_block,
            "detector_output": detector_output,
            "integrated_gradients": integrated_gradients,
        },
    )

The focused tests are in `test_prompt_attribution_boundary.py`. Verification
attributions are rejected before prompt text is rendered, not merely omitted
afterward. Step 49 adds `test_prompt_leakage.py`, which exercises every
registered condition and the six normal-comparison/attribution cells,
including nested cross-condition leakage attempts.

`attribution_only_baseline.py` creates the Step 37 no-LLM baseline from a
serialized Integrated Gradients artifact. It requires an explicit `--top-k`,
selects ranked non-padding events, preserves source attribution provenance,
and writes only the selected event set to a separate compressed JSONL artifact
and manifest. It does not render a prompt or invoke an LLM:

    python3 research/scripts/attribution_only_baseline.py \
      --attribution-file research/models/attributions/loginsight_lstm/seed_13_test_integrated_gradients.jsonl.gz \
      --top-k 5

The default output is `seed_13_test_attribution_only_k_5.jsonl.gz` next to the
source artifact. The focused tests are in `test_attribution_only_baseline.py`.
The evidence budget remains explicit until the registered top-k value is
frozen.

`random_event_baseline.py` creates the Step 38 matched random-event baseline.
It takes the same explicit `--top-k` budget, samples non-padding events
without replacement using the registered perturbation seed, derives a stable
per-unit selection seed, and excludes Integrated Gradients scores and ranks
from the output. It does not render a prompt or invoke an LLM:

    python3 research/scripts/random_event_baseline.py \
      --attribution-file research/models/attributions/loginsight_lstm/seed_13_test_integrated_gradients.jsonl.gz \
      --top-k 5 \
      --seed 2026

The default output is `seed_13_test_random_event_k_5.jsonl.gz` next to the
source artifact. The focused tests are in `test_random_event_baseline.py`.
The top-k budget remains explicit until registration freezes it.

`validate_attribution_storage.py` implements Step 39. It registers separate
method-specific Integrated Gradients, event-occlusion, and leave-one-event-out
JSONL artifacts. It hashes each file, rejects mixed-method files, duplicate
unit IDs, reused paths, and verification fields in Integrated Gradients, then
writes only a non-merging registry manifest:

    python3 research/scripts/validate_attribution_storage.py \
      --integrated-gradients-file research/models/attributions/loginsight_lstm/seed_13_test_integrated_gradients.jsonl.gz \
      --occlusion-file research/models/attributions/loginsight_lstm/seed_13_test_occlusion.jsonl.gz \
      --leave-one-event-out-file research/models/attributions/loginsight_lstm/seed_13_test_leave_one_event_out.jsonl.gz

The default manifest is
`research/models/attributions/attribution_vector_store.manifest.json`.
The focused tests are in `test_validate_attribution_storage.py`. Steps 40–46
and the Step 47 smoke-gate machinery are complete; the full repeated Mistral
pilot is currently blocked by runtime failures documented in
`research/experiments/step47a_execution_failures.md`.

`test_mistral_colab_preflight.py` verifies the public Mistral checkpoint on a
Colab GPU using synthetic evidence only. The captured passing result is
`research/results/tables/step46_mistral_smoke_test.json`.

`run_mistral_pilot.py` implements the selected Step 47A local/Colab runner. It
loads Mistral once with the available backend, reuses the frozen
prompt and retrieval pipeline, writes resumable raw JSONL records, and displays
the same completed-calls/rate/ETA progress bar as the API runner. A safe first
smoke run without attribution artifacts is:

    python3 research/scripts/run_mistral_pilot.py \
      --max-items 2 \
      --conditions blinded,detector_only,no_detector_context

The default output is
`research/experiments/pilot/step47a_mistral_generation.jsonl`. The full
seven-condition exploratory run additionally requires a matching
Integrated-Gradients artifact through `--attribution-file`.

`test_qwen_18_smoke.py` implements the matched Step 47B gate. It uses the same
synthetic prompt, 18-call count, greedy decoding, 4-bit NF4 loading, and
JSONL record format as the Mistral comparison:

    python3 research/scripts/test_qwen_18_smoke.py

The default outputs are
`research/results/tables/step47b_qwen_18_smoke_responses.jsonl` and
`research/results/tables/step47b_qwen_18_smoke_test.json`. This is only the
18-call smoke test; the full repeated Qwen pilot remains separate.

### Historical GPT-OSS runners

The following runners are retained only for reproducibility and provenance.
GPT-OSS is not part of the active model pair; use `run_mistral_pilot.py` for
new Step 47A work.

`run_gpt_oss_pilot.py` implements the historical hosted-provider version of
Step 47A. It is resumable, writes raw
prompt/response records, retries transient provider failures, and displays a
dependency-free progress bar with completed calls, condition, rate, and ETA.
The default is deliberately limited to two items. It reads `GROQ_API_KEY`
from the repository `.env` when present, or from the process environment. It
also honors Groq `429` retry delays and streams only the retrieved normal
records from the large training artifact. A safe first smoke run without
attribution artifacts is:

    python3 research/scripts/run_gpt_oss_pilot.py \
      --max-items 2 \
      --conditions blinded,detector_only,no_detector_context

The full seven-condition exploratory run additionally requires an
Integrated-Gradients artifact covering the selected pool items:

    python3 research/scripts/run_gpt_oss_pilot.py \
      --max-items 2 \
      --attribution-file loginsight_lstm=PATH_TO_MATCHING_IG_JSONL_GZ

The runner uses three repetitions by default, resumes from its JSONL output,
and stores raw responses with `parsing_status: deferred_to_step48`. Parse the
completed output without changing the raw response with:

    python3 research/scripts/parse_explanation_responses.py \
      --input research/experiments/pilot/step47a_mistral_generation.jsonl \
      --output research/experiments/pilot/step47a_mistral_generation_parsed.jsonl \
      --summary research/experiments/pilot/step47a_mistral_generation_parsed.summary.json

The same parser accepts the Step 47B JSONL container. It classifies malformed
responses and valid abstentions; it does not repair or score explanations.

`pilot_record.py` defines the Step 50 reproducibility contract. Completed
pilot records preserve the exact prompt messages, prompt and response hashes,
raw response, decoding settings, request/run identifiers, model metadata, and
provider payload. Validate a record in Python with
`validate_reproducibility_record`; focused coverage is in
`test_pilot_record.py`.

For the full Mistral 47A run, use `run_mistral_pilot.py --full` only after a
bounded real-record generation succeeds on the chosen runtime and explicit
authorization is given. The runner reports each preflight and generation
transition to stderr and writes a live JSON status file beside the generation
output (override it with `--status-file`). A model-load success is not enough;
preserve/download the output and status files before a remote runtime ends.

`run_gpt_oss_cloudflare_pilot.py` runs the historical GPT-OSS smoke test through
Cloudflare Workers AI's OpenAI-compatible endpoint. It reads
`CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` from `.env`, writes a
separate output file, and uses provider-specific record IDs so Groq and
Cloudflare results cannot be mixed accidentally. A matched 18-request replay
is:

    python3 research/scripts/run_gpt_oss_cloudflare_pilot.py \
      --max-items 2 \
      --conditions blinded,detector_only,no_detector_context

Cloudflare uses `reasoning_effort=low` and a provider-specific 2048-token
completion budget because the shorter Groq-style 512-token budget can be
consumed by GPT-OSS reasoning before visible JSON is returned on long prompts.

The default output is
`research/experiments/pilot/step47a_cloudflare_gpt_oss_generation.jsonl`.

`sequence_overlap_audit.py` implements reviewer gate R1. It hashes each exact
ordered, unpadded HDFS `Sequence`, compares train and test sequence sets,
reports the novel-test subset, calculates detector metrics on that subset for
each supplied prediction file, and counts unique sequences in the annotation
pool. It writes:

    python3 research/scripts/sequence_overlap_audit.py \
      --prediction-file research/models/checkpoints/loginsight_lstm/predictions_test.jsonl.gz \
      --prediction-file research/models/checkpoints/logrobust_attention_bilstm/predictions_test.jsonl.gz

The report is
`research/results/tables/sequence_overlap_audit.json`. The focused tests are
in `test_sequence_overlap_audit.py`.

`audit_integrated_gradients.py` implements reviewer gate R4 over serialized
Integrated Gradients artifacts. It reports baseline and completeness
distributions, compares integration-step counts, records logit-space results
separately, and evaluates trained-versus-randomized attribution similarity per
architecture. The completed two-architecture report is
`research/results/tables/integrated_gradients_r4_audit.json`; its coverage
status is distinct from attribution quality because no quality threshold was
pre-registered. Focused tests are in
`test_audit_integrated_gradients.py`.

`verify_colab_transfer.py` implements reviewer gate R5. It hashes the copied
primary-seed checkpoints and prediction files, re-runs local inference for
train/validation/test without overwriting stored predictions, and compares
probabilities and metadata by prediction-unit ID against the registered
`1e-6` numeric tolerance. It prints architecture/split progress with rows,
rate, and ETA and writes
`research/results/tables/r5_colab_transfer_verification.json`:

    python3 research/scripts/verify_colab_transfer.py \
      --repo-root . --architecture both \
      --split train validation test \
      --output research/results/tables/r5_colab_transfer_verification.json

The full command is intentionally a long local inference run. A nonzero exit
code or a report status of `fail` leaves R5 open; do not mark the checklist
item complete from a bounded smoke run.

`shortcut_baseline_audit.py` implements reviewer gate R2. It fits two
training-only baselines on the full unpadded HDFS `Sequence`: a
bag-of-event-template-count logistic regression and a sequence-length-only
logistic regression. Both use class-balanced fitting and a fixed 0.5 threshold
without test tuning, then are compared with supplied detector prediction
files:

    python3 research/scripts/shortcut_baseline_audit.py \
      --prediction-file research/models/checkpoints/loginsight_lstm/predictions_test.jsonl.gz \
      --prediction-file research/models/checkpoints/logrobust_attention_bilstm/predictions_test.jsonl.gz

The report is
`research/results/tables/shortcut_baseline_audit.json`. The focused tests are
in `test_shortcut_baseline_audit.py`. The event-count result is a shortcut
diagnostic, not a replacement for detector or explanation evaluation.

`select_primary_explanation_seeds.py` implements reviewer gate R3. It reads
the completed multi-seed run manifests and selects one primary seed per
architecture using validation F1 only. Ties are broken by lower validation
loss and then lower seed. Test metrics and explanation/attribution outputs are
not used for selection; all remaining seeds are recorded for secondary
attribution-stability analysis:

    python3 research/scripts/select_primary_explanation_seeds.py

The report is
`research/results/tables/primary_explanation_seed_selection.json`. Focused
tests are in `test_select_primary_explanation_seeds.py`.

`research/experiments/llm_explanation_schema.json` implements Step 40. It
defines the strict, machine-readable LLM payload: answer status, summary,
typed claims, cited event IDs, cited line IDs, per-claim and overall
confidence, and explicit abstention status and reason. The companion
`explanation_schema.py` validates already-decoded payloads, including
cross-field citation and abstention consistency. The Step 48 parser
`parse_explanation_responses.py` decodes raw JSON (including JSON code fences),
preserves the original response, and records a deterministic parsing status.

Run the focused schema tests with:

    python3 -m unittest research.scripts.test_explanation_schema

`training_only_retrieval.py` and `blinded_prompt.py` implement Step 41.
The retriever fits an `event_tfidf_cosine` index on normal training blocks
only, never reads query labels for ranking, uses a deterministic unit-ID tie
break, and records a deterministic lowest-ID fallback for zero-overlap
queries. Retrieval records and their manifest preserve source hashes,
similarities, candidate provenance, and the fact that attribution was not
used:

    python3 research/scripts/training_only_retrieval.py \
      --training-evidence research/data/processed/hdfs_block_evidence_train.jsonl.gz \
      --query-evidence research/data/processed/hdfs_block_evidence_test.jsonl.gz \
      --query-split test \
      --output research/experiments/pilot/normal_retrieval_test.jsonl.gz

The blinded prompt exposes only anomalous evidence, the retrieved normal
comparison evidence, and basic detector output through the existing prompt
boundary. It does not accept Integrated Gradients, occlusion, or
leave-one-event-out values. Focused tests are in
`test_training_only_retrieval.py` and `test_blinded_prompt.py`.

`contrastive_prompt.py` implements Step 42. It exposes the full provisional
normal-comparison `{true, false}` × attribution `{none, integrated_gradients,
shuffled_integrated_gradients}` factor. The six cells map to strict prompt
conditions; each requires or rejects the corresponding normal and attribution
payloads before rendering. Run its focused tests with:

    python3 -m unittest research.scripts.test_contrastive_prompt

`shuffled_integrated_gradients.py` implements Step 43. It creates deterministic
prompt-visible shuffled Integrated Gradients artifacts using either
`within_block_permutation` or `cross_block_donor`. The latter requires a
distinct donor artifact with the same number of active events. Both modes
preserve event alignment, recompute attribution ranks, and record provenance;
the prompt payload removes padding and artifact identifiers. The shuffle
registration remains provisional until Step 62:

    python3 research/scripts/shuffled_integrated_gradients.py \
      --attribution-file research/models/attributions/attributions.jsonl.gz \
      --shuffle-type within_block_permutation \
      --output research/experiments/pilot/shuffled_integrated_gradients.jsonl.gz

Focused tests are in `test_shuffled_integrated_gradients.py`.

`no_detector_context_prompt.py` implements Step 44. It builds the registered
`no_detector_context` condition from anomalous evidence only. Detector output,
normal comparison evidence, attribution values, labels, and storage metadata
are not passed to the prompt; event and line identifiers remain available for
structured citations. The prompt uses template ID `no_detector_context_v1`:

    python3 -m unittest research.scripts.test_no_detector_context_prompt

Step 45 selected `mistralai/Mistral-7B-Instruct-v0.3` and
`Qwen/Qwen2.5-7B-Instruct` as separately evaluated, publicly downloadable
open-weight models, with Colab GPU as the default runtime target. The Mistral
and Qwen licenses must be recorded. Exact checkpoint revisions and runtime
capture were completed in Step 46 and are recorded in the smoke-test reports
below.

Step 46 is complete. The exact Mistral and Qwen revisions and Colab runtime
are recorded in `research/results/tables/step46_mistral_smoke_test.json` and
`research/results/tables/step46_qwen_smoke_test.json`. Steps 47A and 47B have
now passed their 18-call smoke gates. The OSF registration is archived at
https://osf.io/qrvuy/overview; confirmatory generation must use the frozen
protocol and cannot be altered based on pilot results. The Groq
GPT-OSS smoke record remains historical in
`research/results/tables/step46_llm_smoke_test.json`.

`train_multiseed_detectors.py` runs the registered Step 29 seed set for both
detectors. It reuses complete seed-13 artifacts, trains missing seeds
sequentially, validates every run, and writes a resumable manifest:

    python3 research/scripts/train_multiseed_detectors.py --dry-run
    python3 research/scripts/train_multiseed_detectors.py

The actual run is intentionally long-running and must be started explicitly.

The configured threshold remains the provisional 0.5 reference threshold;
validation-only threshold selection is a later registered step and must not be
performed on the test split here.

`calibrate_detector.py` applies validation-only temperature scaling to detector
prediction files. It fits one positive temperature on validation labels,
reports NLL, Brier score, and expected calibration error before and after
calibration, and writes calibrated predictions plus a manifest. It does not
select an alert threshold; that is Step 23.

After a detector run has produced prediction files, run:

    python3 research/scripts/calibrate_detector.py \
      --validation-predictions research/models/checkpoints/loginsight_lstm/predictions_validation.jsonl.gz \
      --train-predictions research/models/checkpoints/loginsight_lstm/predictions_train.jsonl.gz \
      --test-predictions research/models/checkpoints/loginsight_lstm/predictions_test.jsonl.gz \
      --output-dir research/models/checkpoints/loginsight_lstm/calibrated

`select_threshold.py` chooses the alert threshold from validation predictions
only. The registered rule maximizes validation F1, breaks ties by choosing the
threshold closest to 0.5, then chooses the lower threshold. It applies the
frozen threshold to all splits, writes thresholded prediction files, and saves
`threshold_manifest.json`. Test labels are used only for the final descriptive
metrics and never for threshold selection.

After calibration (or directly from raw predictions when calibration is not
being used), run:

    python3 research/scripts/select_threshold.py \
      --validation-predictions research/models/checkpoints/loginsight_lstm/calibrated/calibrated_predictions_validation.jsonl.gz \
      --train-predictions research/models/checkpoints/loginsight_lstm/calibrated/calibrated_predictions_train.jsonl.gz \
      --test-predictions research/models/checkpoints/loginsight_lstm/calibrated/calibrated_predictions_test.jsonl.gz \
      --output-dir research/models/checkpoints/loginsight_lstm/thresholded

`record_lstm_baseline.py` consolidates the raw, calibrated, and thresholded
artifacts into a reproducible baseline report. It records average precision as
the PR-AUC definition, thresholded precision/recall/F1, false-positive counts,
and raw/calibrated NLL, Brier score, and ECE for every split. Run it after
threshold selection:

    python3 research/scripts/record_lstm_baseline.py \
      --run-dir research/models/checkpoints/loginsight_lstm \
      --output research/results/tables/lstm_baseline_seed_13.json
