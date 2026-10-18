# LogInsight Implementation Record: Steps 1–46

## Purpose

This document records what was implemented for Steps 1–46 of the LogInsight research plan, what was verified, and what remains pending. It is intended to make the project recoverable if a future Codex conversation starts without the earlier chat context.

The research target is block-level log anomaly detection with LSTM/LogRobust-style detectors and LLM-generated explanations. The primary explanation question is whether exposing Integrated Gradients improves explanation quality while remaining faithful to the detector and useful to a human evaluator.

## Current project state

- Repository: `/Users/neilvargheseabraham/Downloads/LogInsight`
- Research branch: `research-implementation`
- Primary prediction unit: complete HDFS `BlockId` block
- Transfer prediction unit: non-overlapping BGL 60-second window
- Primary dataset: LogHub HDFS_v1
- Transfer dataset: LogHub BGL
- Registered split seed: `2026`
- Primary split: HDFS block-disjoint, label-stratified 80/10/10 split
- Sensitivity split: strict chronological split with boundary-crossing HDFS blocks removed
- Primary model: repository LogInsight LSTM, to be retrained under the research protocol
- Second model: published LogRobust-style attention Bi-LSTM, implemented and trained in Step 26
- Final annotation pool: generated for HDFS with 240 items, 120 per detector architecture; BGL remains outside this completed pilot pool

## Status summary

| Step | Status | Main result |
|---|---|---|
| 1 | DONE | Research branch created |
| 2 | DONE | Both repository development lines audited |
| 3 | DONE | Local and historical Colab environment recorded |
| 4 | DONE | Research directory structure created |
| 5 | DONE | Central experiment configuration created |
| 6 | DONE | Experiment ledger schema created |
| 7 | DONE | Global seed utilities created and tested |
| 8 | DONE | HDFS_v1 source and provenance verified |
| 9 | DONE | HDFS format and timestamp/block structure inspected |
| 10 | DONE | HDFS block construction completed |
| 11 | DONE | Explanation-ready HDFS evidence completed |
| 12 | DONE | BGL downloaded and checksum-verified |
| 13 | DONE | BGL alert-label prefixes removed from parser input |
| 14 | DONE | BGL 60-second evaluation windows created |
| 15 | DONE | Training-only unknown-event vocabulary support implemented |
| 16 | DONE | Shared HDFS/BGL preprocessing validation passed |
| 17 | DONE | HDFS block-disjoint split created and verified |
| 18 | DONE | HDFS/BGL temporal sensitivity splits created and verified |
| 19 | DONE | Automated leakage checks passed |
| 20 | DONE | 240-item stratified HDFS annotation pool generated |
| 21 | DONE | Reusable LogInsight LSTM training/evaluation entry point |
| 22 | DONE | Detector probability calibration |
| 23 | DONE | Validation-based threshold selection |
| 24 | DONE | Leakage-free LogInsight LSTM training |
| 25 | DONE | LSTM baseline results recorded |
| 26 | DONE | LogRobust-style attention Bi-LSTM integrated and trained |
| 27 | DONE | LogRobust implementation deviations documented |
| 28 | DONE | Shared detector interface created |
| 29 | DONE | Both detectors trained and validated across five seeds |
| 30 | DONE | Detector inference batching and caching added |
| 31 | DONE | Integrated Gradients implemented for the LogInsight LSTM |
| 32 | DONE | Integrated Gradients implemented for the LogRobust-style Bi-LSTM |
| 33 | DONE | Integrated Gradients completeness and alignment validator added |
| 34 | DONE | Verification-only single-event occlusion scores implemented |
| 35 | DONE | Verification-only leave-one-event-out scores implemented |
| 36 | DONE | Attribution separation enforced before prompt rendering |
| 37 | DONE | Top-k Integrated Gradients attribution-only baseline implemented |
| 38 | DONE | Matched random-event baseline implemented |
| 39 | DONE | Separate attribution-vector storage registry and validator implemented |
| 40 | DONE | Structured LLM explanation schema and decoded-payload validator implemented |
| 41 | DONE | Training-only normal retrieval and blinded contrastive prompt implemented |
| 42 | DONE | Unblinded prompt and normal-comparison × attribution factor implemented |
| 43 | DONE | Shuffled Integrated Gradients generation and prompt payload implemented |
| 44 | DONE | No-detector-context prompt implemented |
| 45 | DONE | API and public open-weight LLM models selected |
| 46 | DONE | Model revisions, decoding settings, and Colab smoke test recorded |

## Documentation coverage audit

The completed checklist items through Step 46 are individually documented in
this record: every completed item from Steps 1–46 has a corresponding
`### Step N` section below. The implementation summary provides the shorter
reviewer-facing version, while `research/project_knowledge_graph.md` provides
the persistent architecture and provenance index.

The documentation does not claim that future checklist items are complete.
Step 47 and later remain pending unless explicitly marked in
`research/implementation_steps.txt`. In particular, the LLM explanation
pipeline, human annotation collection, ethics/registration work, and the
remaining perturbation and pilot experiments have not been silently inferred
from the detector and attribution implementations.

## Step-by-step record

### Step 1 — Create the research implementation branch

A dedicated branch named `research-implementation` was created so the research pipeline could evolve without changing the product branches. The branch was pushed to the remote during the early implementation work.

The research changes are kept separate from the existing dashboard/product line. Later changes should continue on this branch unless the project explicitly changes its branch strategy.

### Step 2 — Audit both repository branches

The repository was inspected before implementation. The important distinction is:

- The main/research foundation contains the current LSTM detector, Drain-based parsing, HDFS processing, model loading, and basic prediction path.
- `origin/llm_integration_and_dashboard` is a separate product-development line containing authentication, Next.js dashboard work, live monitoring, provider/API integration, and expanded server behavior.

The research decision was to use the main branch’s detector foundation and keep product dashboard behavior outside the core scientific benchmark. The existing LLM interface is treated as repository context, not as the research contribution.

The architecture audit is summarized in `research/project_knowledge_graph.md`.

### Step 3 — Freeze environment provenance

The project records two environments rather than pretending the historical Colab model was trained in the local environment.

Local evidence:

- Python: `3.9.6`
- TensorFlow: `2.20.0`
- Keras: `3.10.0`
- Direct dependency pins: `research/requirements-local.txt`

Historical Colab evidence:

- Python: `3.12.13`
- Saved model archive reports Keras `3.13.2`
- TensorFlow and scikit-learn versions were not recorded in the original run
- The Colab runtime image/GPU identifier was not recorded

This limitation is documented in `research/colab_environment_record.txt`. A capture cell was added there for the next Colab training run. Every future model result must record an environment ID.

### Step 4 — Create the research directory structure

The research layout was created to separate source data, processed data, models, prompts, experiments, results, and scripts:

```text
research/
├── data/{raw,processed,manifests}
├── models/{checkpoints,attributions,vocabularies}
├── prompts/
├── experiments/{pilot,main}
├── results/{tables,figures}
└── scripts/
```

Existing root-level model and notebook artifacts were not silently moved. Their original locations remain part of the repository provenance.

### Step 5 — Create the central experiment configuration

`research/experiments/experiment_config.json` is the central protocol configuration. It records:

- HDFS and BGL data paths and prediction units
- split fractions and seeds
- parser configuration and training-only vocabulary rules
- LogInsight LSTM and LogRobust detector identities
- prompt conditions: blinded, unblinded, shuffled Integrated Gradients, attribution-only, random-event, and no-detector-context
- the rule that Integrated Gradients is the only attribution signal visible to the unblinded LLM
- verification-only occlusion and leave-one-event-out signals
- perturbation strategies
- annotation outcomes and blinding
- TOST margin, power assumptions, and reporting requirements
- temporal splitting and leakage-control policies
- annotation-pool sampling rules

The configuration remains provisional until the protocol is formally registered on OSF or AsPredicted.

### Step 6 — Create the experiment ledger schema

`research/experiments/experiment_ledger_schema.json` defines the record that must be saved for each evaluated block/window and condition. It separates:

- human-evidence recovery;
- claim support;
- detector faithfulness;
- perturbation consistency;
- detector/model provenance;
- parser and vocabulary provenance;
- split identity;
- prompt and LLM settings;
- perturbation strategy and seed;
- latency, token, and cost information;
- quality checks and leakage status.

This prevents plausibility, human usefulness, and detector faithfulness from being collapsed into one unsupported score.

### Step 7 — Add global seed utilities

`research/scripts/seed_utils.py` provides:

- Python, NumPy, and TensorFlow seeding;
- stable derived seeds for named sub-runs;
- deterministic-operation reporting;
- explicit recording of the `PYTHONHASHSEED` limitation.

It is intended for detector training, annotation sampling, perturbation experiments, and repeated LLM runs. Tests are in `research/scripts/test_seed_utils.py`.

### Step 8 — Verify the HDFS dataset source and version

The local HDFS data was identified as the official LogHub `HDFS_v1` variant. The provenance references:

- LogHub HDFS source;
- the user-provided Zenodo HDFS_v1 archive link;
- the local raw-file checksum and artifact checksums.

The provenance record is `research/data/manifests/hdfs_v1_manifest.json`.

The source contains 11,175,629 log rows and approximately 575,061 labelled HDFS blocks.

### Step 9 — Inspect the HDFS format

The format report confirmed:

```text
<Date> <Time> <Pid> <Level> <Component>: <Content>
```

The inspection confirmed:

- every raw row contains an HDFS `BlockId` pattern;
- the source contains timestamps;
- labels are supplied at the block level through `anomaly_label.csv`;
- parsed records contain `LineId`, raw fields, `EventId`, `EventTemplate`, and `ParameterList`;
- 575,061 BlockIds are covered by the official labels.

The detailed report is `research/data/manifests/hdfs_v1_format_report.json`.

### Step 10 — Build HDFS blocks

`research/scripts/build_hdfs_blocks.py` groups all events with the same `BlockId`, orders events by `LineId`, joins the official block label, and writes one compressed JSONL record per block.

The full user-run result was:

- input rows: 11,175,629;
- output blocks: 575,061;
- labelled blocks: 575,061;
- rows with BlockId: 11,175,629;
- rows without BlockId: 0;
- partial input: 0.

Output: `research/data/processed/hdfs_block_sequences.jsonl.gz`.

The script includes a progress bar and accepts `--max-rows` only for smoke tests.

### Step 11 — Preserve explanation-ready HDFS evidence

`research/scripts/build_hdfs_block_evidence.py` creates the explanation-ready HDFS artifact. It retains the detector-compatible event-ID sequence and adds per-event evidence:

- `Position`;
- `LineId`;
- `EventId`;
- `EventTemplate`;
- parsed `ParameterList`;
- `ParameterListRaw`;
- exact `RawLine`.

The full user-run result was:

- evidence events: 11,175,629;
- labelled blocks: 575,061;
- rows without BlockId: 0;
- partial input: 0.

Outputs:

- `research/data/processed/hdfs_block_evidence.jsonl.gz`
- `research/data/manifests/hdfs_block_evidence_manifest.json`

### Step 12 — Download and verify BGL

`research/scripts/download_verify_bgl.py` downloaded the official LogHub BGL archive and recorded provenance/checksums.

Verified values:

- archive MD5: `4452953c470f2d95fcb32d5f6e733f7a`;
- raw line count: 4,747,963;
- raw-log SHA-256: `666130b15ef44eb32fd02bd053e6c6e007c37696b5e7e8b9d8e45b729876a5d2`.

Outputs:

- `research/data/raw/BGL.zip`
- `research/data/raw/BGL.log`
- `research/data/manifests/bgl_manifest.json`

### Step 13 — Remove BGL alert-label prefixes

`research/scripts/strip_bgl_label_prefix.py` removes the first BGL alert-label token before any parsing, detector input, retrieval, attribution, or LLM input.

The original label information is retained only in the audit stream, which is not permitted as model/LLM evidence.

Verified values:

- processed lines: 4,747,963;
- alert lines: 348,460;
- normal lines: 4,399,503;
- distinct prefixes: 42;
- clean output contains label prefix: false.

Outputs:

- parser-safe log: `research/data/processed/BGL_clean.log`;
- audit-only labels: `research/data/processed/BGL_label_audit.jsonl.gz`;
- manifest: `research/data/manifests/bgl_label_strip_manifest.json`.

### Step 14 — Define and build the BGL evaluation unit

`research/scripts/build_bgl_windows.py` freezes BGL as non-overlapping 60-second Unix-epoch windows:

```text
window_start = floor(epoch_timestamp / 60) * 60
```

A window is anomalous if any source line in it has a non-`-` alert prefix. The prefix itself is not included in the parser-safe evidence.

Verified values:

- total windows: 27,670;
- anomalous windows: 1,838;
- normal windows: 25,832;
- source lines: 4,747,963;
- overlap: 0;
- partial input: 0.

Outputs:

- `research/data/processed/BGL_windows.jsonl.gz`;
- `research/data/manifests/bgl_window_manifest.json`.

### Step 15 — Implement the unknown-event token

`research/scripts/event_vocabulary.py` and `research/scripts/build_event_vocabulary.py` implement research vocabulary handling.

Rules:

- `__PAD__` is always ID 0;
- `__UNKNOWN_EVENT__` is always present as a learned row;
- the vocabulary is fitted on training sequences only;
- unseen validation, test, and perturbation events map to the unknown-event row;
- the old engineering `research/event2idx.pkl` is not silently modified.

The actual training-only vocabulary generation and detector retraining occur in later detector steps.

### Step 16 — Build shared preprocessing validation

`research/scripts/preprocessing_validation.py` contains shared validation rules. `research/scripts/validate_preprocessing.py` runs them over the full HDFS and BGL artifacts.

The full validation passed with:

- HDFS: 575,061 blocks and 11,175,629 events;
- BGL: 27,670 windows and 4,747,963 lines;
- valid labels, ordering, lengths, raw evidence, line ranges, and manifest counts;
- no malformed records detected.

Tests cover malformed records and schema violations.

### Step 17 — Create block-disjoint HDFS splits

`research/scripts/split_hdfs_blocks.py` creates a deterministic label-stratified HDFS split using seed `2026`.

The registered fractions and verified counts are:

| Split | Blocks | Anomaly | Normal |
|---|---:|---:|---:|
| Train | 460,049 | 13,470 | 446,579 |
| Validation | 57,506 | 1,684 | 55,822 |
| Test | 57,506 | 1,684 | 55,822 |

An independent check confirmed 575,061 unique BlockIds and no BlockId in more than one split.

Outputs:

- `research/data/processed/hdfs_block_evidence_train.jsonl.gz`;
- `research/data/processed/hdfs_block_evidence_validation.jsonl.gz`;
- `research/data/processed/hdfs_block_evidence_test.jsonl.gz`;
- `research/data/manifests/hdfs_block_assignments.jsonl.gz`;
- `research/data/manifests/hdfs_split_manifest.json`.

### Step 18 — Create temporal sensitivity splits

`research/scripts/create_temporal_splits.py` creates a separate chronological sensitivity split for HDFS and BGL.

Policy:

- order units by their first timestamp;
- use 80/10/10 chronological partitions;
- retain a unit only if its complete time interval fits inside one partition;
- exclude HDFS blocks crossing a temporal boundary instead of leaking future events into an earlier split;
- record all exclusions and boundary timestamps.

Verified HDFS result:

- train: 394,619 blocks;
- validation: 55,784 blocks;
- test: 57,505 blocks;
- excluded boundary-crossing blocks: 67,153;
- strict temporal ordering: true.

Verified BGL result:

- train: 22,136 windows;
- validation: 2,767 windows;
- test: 2,767 windows;
- excluded windows: 0;
- strict temporal ordering: true.

The manifest is `research/data/manifests/temporal_split_manifest.json`.

The primary block-disjoint split remains the main benchmark split. The temporal split is a generalization/sensitivity analysis.

### Step 19 — Create leakage-detection checks

`research/scripts/validate_leakage.py` audits:

1. duplicated or misplaced prediction units across primary and temporal split bundles;
2. the policy that normal retrieval candidates must come from training only;
3. the policy that parser and vocabulary fitting must use training only;
4. BGL parser-safe input for alert-label leakage;
5. optional runtime retrieval ledgers;
6. optional parser/vocabulary fit manifests.

The completed audit passed:

- primary HDFS split: 575,061 assignments, zero duplicates, zero misplaced records;
- temporal HDFS split: 507,908 assigned blocks, 67,153 explicitly excluded, zero duplicates, zero misplaced records;
- temporal BGL split: 27,670 assigned windows, zero duplicates, zero misplaced records;
- BGL parser-safe input: 4,747,963 lines, zero invalid first tokens;
- parser policy: training-only and frozen;
- retrieval policy: training-only source split.

The audit manifest is `research/data/manifests/leakage_check_manifest.json`.

The retrieval-ledger and parser-runtime checks are marked pending because those artifacts are produced by later detector/retrieval implementation steps. The checker is already able to validate them when supplied.

### Step 20 — Implement the annotation-pool sampler

`research/scripts/sample_annotation_pool.py` and `research/experiments/annotation_candidate_schema.json` implement the sampling protocol.

Each detector-produced candidate must include:

- `dataset` (`HDFS` or `BGL`);
- `unit_id`;
- held-out `split`;
- gold `label`;
- `sequence_length`;
- detector `confidence` in `[0, 1]`;
- detector `architecture`.

The sampler stratifies by:

- dataset;
- Normal/Anomaly label;
- length bin: 0–10, 11–50, 51+;
- confidence bin: low, medium, high;
- detector architecture.

The registered defaults are:

- eligible split: `test`;
- target pool: 240 annotation items;
- allowed range: 200–300;
- sampling seed: 2026;
- annotation item: dataset + prediction unit + architecture.

The same prediction unit may appear once per architecture so the annotation study can compare the LSTM and LogRobust-style detector explanations. The manifest will report both item count and unique block/window count.

The sampler was unit-tested, including deterministic repeatability, missing-metadata rejection, bin boundaries, and target-size validation. The candidate builder in `research/scripts/build_annotation_candidates.py` converts the held-out detector prediction files into the required metadata and rejects duplicate dataset/unit/architecture records. The completed HDFS pool contains 240 items, 120 per architecture, and 239 unique blocks; its provenance is recorded in `research/experiments/pilot/annotation_pool_manifest.json`.

### Step 21 — Refactor the LogInsight LSTM into a reusable training/evaluation script

`research/scripts/train_loginsight_lstm.py` now provides the research detector
entry point. It consumes the frozen HDFS train/validation/test evidence splits,
loads a vocabulary manifest fitted on the training split only, maps unseen
events to the learned unknown-event row, pre-pads/truncates sequences to the
configured maximum length, and rejects duplicate prediction units across
splits.

The default architecture preserves the repository LSTM notebook: embedding
dimension 128; LSTM layers with 128 and 64 units; batch normalization after
each LSTM; dropout 0.3, 0.3, and 0.2; a dense ReLU layer with 32 units; and a
sigmoid binary output.

The script records explicit configuration and seed metadata, uses balanced
class weights when enabled, saves best and final Keras checkpoints, writes
split-wise prediction metadata, and reports fixed-threshold metrics. The
threshold remains the provisional 0.5 reference value; validation-only
threshold selection is deferred to Step 23. An `--evaluate-only` mode reuses a
checkpoint without retraining.

Configuration is stored in `research/experiments/lstm_training_config.json`.
The vocabulary-generation and training commands are documented in
`research/scripts/README.md`. Full model training was intentionally not run by
the agent because it is a long-running experiment; the user should run those
commands in the terminal.

### Step 22 — Add detector probability calibration

`research/scripts/calibrate_detector.py` implements validation-only temperature
scaling for detector prediction files. It fits one positive temperature on
validation labels by minimizing validation negative log-likelihood, then
applies that temperature to train, validation, and test probabilities. The
transformation preserves score ranking while making probabilities more
comparable across detector architectures and seeds.

The script reports negative log-likelihood, Brier score, and fixed-bin expected
calibration error before and after calibration. It writes calibrated
prediction files and `calibration_manifest.json`, including the fit split,
temperature bounds, and input/output paths. It deliberately removes the
provisional thresholded prediction from calibrated outputs; threshold selection
is reserved for Step 23 and must use validation labels only.

The calibration configuration is recorded for both detector architectures in
`research/experiments/experiment_config.json`. After the user completed Step
24, calibration was run on the seed-13 HDFS prediction files. The fitted
temperature was 1.272477891888822; validation NLL changed from 0.007232 to
0.006461, and the calibrated files and manifest are under
`research/models/checkpoints/loginsight_lstm/calibrated/`.

### Step 23 — Add validation-based threshold selection

`research/scripts/select_threshold.py` implements the frozen alert-threshold
protocol. It enumerates thresholds that cover every distinct predicted-positive
set, maximizes validation F1, breaks ties by choosing the threshold closest to
the provisional 0.5 reference, and then chooses the lower threshold. The
selected threshold is applied unchanged to train, validation, and test
prediction files.

The script writes thresholded predictions and `threshold_manifest.json`. The
manifest records the selection rule, selected threshold, validation metrics,
per-split descriptive metrics, and `test_labels_used_for_selection: false`.
Test labels may appear in the final descriptive test metrics, but cannot affect
the selected threshold. The script accepts either raw detector predictions or
the calibrated prediction files from Step 22.

Threshold selection was executed after the user completed Step 24; no test
labels were used to choose the threshold.

### Step 24 — Train the leakage-free LogInsight LSTM

The user trained the HDFS detector with seed 13 using the M4 Pro GPU and the
research configuration. Early stopping stopped after epoch 4 and restored the
best weights from epoch 1. The run produced the checkpoint, split-wise
predictions, and manifest under
`research/models/checkpoints/loginsight_lstm/`.

Using the provisional 0.5 threshold before calibration/selection, the test
metrics were accuracy 0.999374, precision 0.981308, recall 0.997625, and F1
0.989399, with 32 false positives and 4 false negatives. After Step 22
calibration and Step 23 validation-only threshold selection, the selected
threshold was 0.35207340985154245. Its descriptive test metrics were accuracy
0.999391, precision 0.981319, recall 0.998219, and F1 0.989697, with 32 false
positives and 3 false negatives. The test labels did not influence threshold
selection.

### Step 25 — Record the LSTM baseline results

`research/scripts/record_lstm_baseline.py` consolidates the raw, calibrated,
and thresholded prediction artifacts into one reproducible baseline result
file. It records precision, recall, F1, average precision (the explicitly
documented PR-AUC definition), false-positive counts/rates, and calibration
metrics (NLL, Brier score, and 10-bin ECE) for train, validation, and test.
The report also records the checkpoint paths, calibration temperature,
selected threshold, source manifests, and the fact that test labels were not
used to select the threshold.

For the completed seed-13 HDFS run, the report is
`research/results/tables/lstm_baseline_seed_13.json`. The held-out test split
has PR-AUC 0.994214, calibrated NLL 0.006706, calibrated Brier score
0.000627, calibrated ECE 0.001457, and thresholded precision 0.981319,
recall 0.998219, and F1 0.989697 (32 false positives and 3 false negatives).
The selected validation-only threshold is 0.35207340985154245.

### Step 26 — Integrate the LogRobust-style attention Bi-LSTM

`research/scripts/train_logrobust_attention_bilstm.py` implements the second
detector entry point. Its model is an event embedding followed by a
bidirectional LSTM, additive temporal attention, a dense classifier, and a
sigmoid anomaly probability. The custom attention layer supports sequence
masks, so padding cannot contribute to the pooled representation, and it
is saved with the Keras checkpoint for later evaluation and attribution work.

The trainer reuses the same training-only vocabulary, HDFS block-disjoint
splits, unit-overlap checks, class-weight policy, seed reporting,
checkpointing, fixed-threshold metrics, and evaluation-only mode as the
LogInsight LSTM. Configuration is stored in
`research/experiments/logrobust_attention_bilstm_training_config.json`.

The implementation is deliberately a LogRobust-style integration rather than
a claim of byte-for-byte reproduction: the published method uses external
semantic embeddings, while this study uses a trainable embedding over the
frozen research event vocabulary so the two detectors receive identical
leakage-controlled inputs. These and the protocol-specific split/training
choices are registered for the Step 27 deviation record. No long-running
training was performed by the agent; the user completed the seed-13 training
run separately.

The Bi-LSTM configuration uses post-padding because TensorFlow Metal/cuDNN
requires right-padded masks. This is a model-input representation choice only:
the parser, training-only vocabulary, prediction units, and frozen splits are
unchanged, and the attention layer still consumes the padding mask so padded
events cannot affect the pooled representation.

The focused model/configuration tests and the complete research test suite
pass (51 tests). The user-run training command is documented in
`research/scripts/README.md`.

The user then trained seed 13 successfully. Early stopping stopped at epoch
15 and restored the best weights from epoch 12. The run manifest,
checkpoints, split-wise predictions, and history are under
`research/models/checkpoints/logrobust_attention_bilstm/`. At the provisional
0.5 threshold, the held-out HDFS test results were accuracy 0.999513,
precision 0.985346, recall 0.998219, and F1 0.991740, with 25 false positives
and 3 false negatives.

### Step 27 — Document LogRobust implementation deviations

The complete deviation record is stored in both machine-readable and
human-readable form:

- `research/experiments/logrobust_implementation_deviations.json`;
- `research/experiments/logrobust_implementation_deviations.md`.

The record distinguishes the preserved architectural idea—semantic event
representations followed by attention-based Bi-LSTM classification—from the
study-specific choices: the shared training-only event vocabulary, HDFS
BlockId units and fixed block-disjoint splits, maximum sequence length 50,
post-padding for TensorFlow Metal mask compatibility, shared class-weight and
early-stopping policy, validation-only calibration/thresholding, and the rule
that attention weights are not treated as faithful explanations. It also
records the exact seed-13 manifest/checkpoint and warns against direct
numerical comparison with published scores from different data protocols.

The experiment registry points to both deviation artifacts, and Step 27 is
complete.

### Step 28 — Create the shared detector interface

`research/scripts/detector_interface.py` defines the architecture-neutral
contract used by both detector trainers. `DetectorBatch` validates integer
event-sequence inputs, binary labels, unique prediction-unit metadata, and
split identity. `DetectorContract` standardizes detector IDs and prediction
fields. `KerasDetectorAdapter` and `predict_probabilities` standardize model
construction/loading and finite probability output. The shared metrics and
prediction writer ensure both architectures emit the same schema for
calibration, threshold selection, attribution, and reporting.

The LogInsight LSTM now delegates its probability validation, binary metrics,
and prediction serialization to the interface. The LogRobust trainer uses the
same path through the existing shared trainer helpers. Architecture-specific
model construction and attention details remain in their respective trainer
modules. Step 28 is complete; the focused interface tests remain included in
the full research suite described below.

### Step 29 — Train both detectors with multiple seeds

The registered detector seeds are 13, 42, 73, 101, and 202. HDFS runs for all
five seeds are now present for both detectors. The LogInsight LSTM runs were
completed locally; the missing LogRobust attention Bi-LSTM runs were completed
on Colab and transferred back to the local repository.

`research/scripts/train_multiseed_detectors.py` is the resumable orchestrator.
It reuses and validates complete existing seed artifacts, invokes each
architecture-specific trainer with the current Python interpreter, validates
the run manifest/checkpoints/history/predictions after every run, and writes a
progress manifest after every target. Completion requires a manifest with
every architecture/seed target validated.

Preview the eight missing training runs:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
source .venv-metal/bin/activate
export PYTHONNOUSERSITE=1
python research/scripts/train_multiseed_detectors.py --dry-run
```

After reviewing the plan, run the actual long experiment:

```bash
python research/scripts/train_multiseed_detectors.py
```

The report is written to
`research/results/tables/multiseed_training_manifest.json`. The completed
report shows all ten targets complete or reused and the per-seed artifacts
passing validation.

The detector metadata and annotation pool were then generated from the
completed prediction artifacts. The candidate metadata and pool are stored at:

- `research/experiments/pilot/annotation_candidates.jsonl`;
- `research/experiments/pilot/annotation_pool.jsonl.gz`;
- `research/experiments/pilot/annotation_pool_manifest.json`.

The pool contains 240 HDFS annotation items, balanced at 120 per architecture,
and covers 239 unique HDFS blocks. The detector seed used for each candidate is
retained in the candidate metadata so the pool provenance is auditable.

### Step 30 — Add detector inference caching and batching

`research/scripts/detector_interface.py` now provides bounded batched
inference through `predict_probabilities_batched`. It preserves input order,
validates the same detector input contract as single-call inference, and
evaluates only bounded chunks so later attribution and perturbation workloads
do not need to submit an unbounded matrix to a model.

`DetectorInferenceCache` provides an in-process probability cache for repeated
inference requests. Cache keys include the detector namespace, model key,
input dtype, input shape, and input bytes. Within a request, duplicate rows are
coalesced; across requests, previously computed probabilities are reused. The
cache exposes requested-row, cache-hit, deduplicated-row, and model-evaluated-
row statistics. It stores detector probabilities only; attribution results
remain separate and are not silently reused.

The focused tests are in
`research/scripts/test_detector_inference_cache.py` and cover empty input,
batch ordering, duplicate coalescing, cache reuse, and namespace isolation.
Step 30 is complete.

### Step 31 — Implement Integrated Gradients for the LogInsight LSTM

`research/scripts/integrated_gradients_lstm.py` implements the registered
prompt-visible attribution method for the research LSTM. Because detector
inputs are integer event IDs, the method integrates gradients in embedding
space rather than differentiating the integer IDs. The default baseline is an
all-padding sequence using the training vocabulary's `__PAD__` row, and the
default path uses 64 trapezoidal integration steps.

For each prediction unit, the output records the anomaly probability,
all-padding baseline probability, signed attribution, absolute attribution,
ranked non-padding events, padded/input positions, retained sequence indices,
and the completeness error between the summed attribution and the model
output difference. Padding positions remain in the artifact for alignment but
are excluded from ranking. Attribution output is kept separate from future
occlusion and leave-one-event-out verification artifacts.

Run the test-split attribution pass explicitly with:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 research/scripts/integrated_gradients_lstm.py \
  --config research/experiments/lstm_training_config.json \
  --checkpoint research/models/checkpoints/loginsight_lstm/best_seed_13.keras \
  --split test
```

The default compressed JSONL output and manifest are written under
`research/models/attributions/loginsight_lstm/`. The focused tests are in
`research/scripts/test_integrated_gradients_lstm.py`; they cover event ranking,
manifest naming, and completeness on a small research-LSTM model.
Step 31 is complete.

### Step 32 — Implement Integrated Gradients for the LogRobust-style attention Bi-LSTM

`research/scripts/integrated_gradients_logrobust.py` applies the same
prompt-visible Integrated Gradients output contract to the LogRobust-style
attention Bi-LSTM. It integrates through the trainable event embedding, passes
the valid-event mask explicitly through the bidirectional LSTM and custom
temporal-attention layer, aggregates embedding dimensions to event-position
scores, and preserves the same ranked event and sequence-alignment fields as
the LSTM artifact.

Because this detector uses `mask_zero=true`, the all-padding embedding baseline
preserves the input event mask during interpolation. This avoids introducing a
discrete all-false mask jump that gradients cannot explain. The baseline mask
policy is written into every attribution manifest and record. Completeness
errors are also recorded per prediction unit so Step 33 can assess numerical
convergence before any attribution is used in prompts.

Run the test-split attribution pass explicitly with:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 research/scripts/integrated_gradients_logrobust.py \
  --config research/experiments/logrobust_attention_bilstm_training_config.json \
  --checkpoint research/models/checkpoints/logrobust_attention_bilstm/best_seed_13.keras \
  --split test
```

The default compressed JSONL output and manifest are written under
`research/models/attributions/logrobust_attention_bilstm/`. Focused tests are
in `research/scripts/test_integrated_gradients_logrobust.py`, including a
masked-attention completeness test and input validation. Step 32 is complete.

### Step 33 — Validate Integrated Gradients outputs

`research/scripts/validate_integrated_gradients.py` validates the serialized
Integrated Gradients artifacts from Steps 31 and 32 without requiring
TensorFlow. It checks probability/baseline arithmetic, attribution-sum and
completeness-error consistency, the default maximum completeness error of
`0.05`, duplicate prediction units, manifest counts, padding flags, retained
sequence indices, input-position ordering, and complete non-padding ranks.

Validation can cover both architectures in one command:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 research/scripts/validate_integrated_gradients.py \
  --attribution-file research/models/attributions/loginsight_lstm/seed_13_test_integrated_gradients.jsonl.gz \
  --attribution-file research/models/attributions/logrobust_attention_bilstm/seed_13_test_integrated_gradients.jsonl.gz
```

The command exits nonzero when an artifact fails validation. `--allow-failures`
is available only for diagnostic reports. The focused tests are in
`research/scripts/test_validate_integrated_gradients.py` and cover valid
records, probability/alignment failures, and completeness-threshold failures.
Step 33 is complete. A production attribution run must still be generated and
passed through this validator before its events are used in later prompts.

### Step 34 — Implement occlusion scores

`research/scripts/occlusion_scores.py` implements the registered
verification-only single-event occlusion diagnostic for both detector
architectures. For each retained event, it deletes that event, re-pads the
remaining event sequence using the detector's configured pre/post-padding
policy, reruns the detector, and records:

```text
occlusion_score = p(original) - p(occluded)
```

The output is deliberately separate from Integrated Gradients. Every record
and event is marked `verification_only: true` and `prompt_visible: false`,
and the script writes a compressed JSONL artifact plus a manifest under
`research/models/attributions/<architecture>/`. Repeated perturbation inputs
reuse the Step 30 `DetectorInferenceCache`; the cache stores probabilities,
not attribution values.

Run the LSTM diagnostic explicitly with:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 research/scripts/occlusion_scores.py \
  --config research/experiments/lstm_training_config.json \
  --checkpoint research/models/checkpoints/loginsight_lstm/best_seed_13.keras \
  --split test
```

Run the LogRobust-style diagnostic with:

```bash
python3 research/scripts/occlusion_scores.py \
  --config research/experiments/logrobust_attention_bilstm_training_config.json \
  --checkpoint research/models/checkpoints/logrobust_attention_bilstm/best_seed_13.keras \
  --split test
```

The focused tests are in `research/scripts/test_occlusion_scores.py` and
cover deletion/re-padding, retained-sequence alignment, score arithmetic,
and cache reuse. Step 34 is complete. Full production runs remain explicit
user-run workloads because they require one detector inference per retained
event.

### Step 35 — Implement leave-one-event-out scores

`research/scripts/leave_one_event_out_scores.py` implements the registered
leave-one-event-out detector-faithfulness diagnostic for both architectures.
Under the current protocol, holding out one event uses the same matched
single-event deletion and re-padding operation as Step 34. It is nevertheless
stored as a separate method and artifact so the two verification diagnostics
remain distinguishable:

```text
leave_one_event_out_score = p(original) - p(held_out)
```

The output defaults to
`research/models/attributions/<architecture>/seed_<seed>_<split>_leave_one_event_out.jsonl.gz`
with a separate manifest. Records use `method: leave_one_event_out`, mark
`verification_only: true` and `prompt_visible: false`, and never belong in an
LLM prompt or attribution-only condition. The Step 30 inference cache is used
for duplicate perturbation inputs.

Run the LSTM diagnostic explicitly with:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 research/scripts/leave_one_event_out_scores.py \
  --config research/experiments/lstm_training_config.json \
  --checkpoint research/models/checkpoints/loginsight_lstm/best_seed_13.keras \
  --split test
```

Run the LogRobust-style diagnostic with:

```bash
python3 research/scripts/leave_one_event_out_scores.py \
  --config research/experiments/logrobust_attention_bilstm_training_config.json \
  --checkpoint research/models/checkpoints/logrobust_attention_bilstm/best_seed_13.keras \
  --split test
```

The focused tests are in `research/scripts/test_leave_one_event_out_scores.py`
and cover deletion/re-padding, score arithmetic, and cache reuse. Step 35 is
complete. Full production runs remain explicit user-run workloads.

### Step 36 — Enforce attribution separation

`research/scripts/prompt_attribution_boundary.py` provides the strict boundary
for research prompt construction. It loads the registered `prompt_conditions`
from `research/experiments/experiment_config.json`, permits only fields listed
as visible for the selected condition, and rejects unsupported fields rather
than silently dropping them. It also recursively rejects occlusion fields,
leave-one-event-out fields, `verification_only` metadata, and records marked
`prompt_visible: false` before JSON prompt rendering.

The boundary therefore permits Integrated Gradients only where the registered
condition allows it, while blinded, random, and no-detector-context conditions
cannot receive attribution values. Occlusion and leave-one-event-out values
remain verification-only and cannot enter an LLM prompt or the attribution-only
condition. Future research prompt consumers must call `build_prompt_payload`
or `render_prompt_payload` before sending any prompt text.

The focused tests are in
`research/scripts/test_prompt_attribution_boundary.py`. They cover the
unblinded Integrated Gradients allowlist, blinded-condition rejection, nested
occlusion/leave-one-out rejection, verification metadata rejection, and
rendered-payload field allowlisting. Step 36 is complete.

### Step 37 — Create the attribution-only baseline

`research/scripts/attribution_only_baseline.py` creates the registered
attribution-only reference from a serialized Integrated Gradients artifact.
The caller must provide an explicit `--top-k` value. For each prediction unit,
the script selects the highest-ranked non-padding Integrated Gradients events,
preserves their event positions, IDs, scores, ranks, and source-file
provenance, and writes a separate compressed JSONL artifact plus manifest.

This baseline does not invoke an LLM, render a prompt, or use occlusion or
leave-one-event-out values. The complete attribution vector remains in the
source Integrated Gradients artifact; the baseline stores only the selected
event set to avoid conflating the reference with the full attribution output.

Run it after generating and validating an Integrated Gradients artifact:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 research/scripts/attribution_only_baseline.py \
  --attribution-file research/models/attributions/loginsight_lstm/seed_13_test_integrated_gradients.jsonl.gz \
  --top-k 5
```

The default output is next to the source artifact as
`seed_13_test_attribution_only_k_5.jsonl.gz`, with a matching manifest. The
focused tests are in
`research/scripts/test_attribution_only_baseline.py`; they cover rank-based
non-padding selection, no-LLM/provenance metadata, source-method rejection,
and artifact writing. Step 37 is complete. The top-k value remains an
explicit experimental choice because the registered evidence budget is not
yet frozen.

### Step 38 — Create the random-event baseline

`research/scripts/random_event_baseline.py` creates the matched random-event
reference from the same serialized Integrated Gradients event positions used
by Step 37. It requires the same explicit `--top-k` value and samples
non-padding events uniformly without replacement. The master seed defaults to
the registered perturbation seed `2026`; a stable per-unit selection seed is
derived from the perturbation seed, architecture, detector seed, split,
prediction-unit ID, and top-k value, so selection does not depend on file
iteration order.

The random baseline does not use Integrated Gradients scores or ranks. Its
separate compressed JSONL artifact stores only selected event positions/IDs,
sampling provenance, detector metadata, and the no-LLM flags. It never renders
a prompt or invokes an LLM. Run it with:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 research/scripts/random_event_baseline.py \
  --attribution-file research/models/attributions/loginsight_lstm/seed_13_test_integrated_gradients.jsonl.gz \
  --top-k 5 \
  --seed 2026
```

The default output is next to the source artifact as
`seed_13_test_random_event_k_5.jsonl.gz`, with a matching manifest. The
focused tests are in `research/scripts/test_random_event_baseline.py`; they
cover reproducibility, matched budget, exclusion of attribution scores/ranks,
source validation, and artifact writing. Step 38 is complete. The top-k value
remains an explicit experimental choice until the evidence budget is frozen.

### Step 39 — Store all attribution vectors separately

`research/scripts/validate_attribution_storage.py` validates and registers
method-specific attribution vector files without combining them. It accepts
separate Integrated Gradients, event-occlusion, and leave-one-event-out paths,
requires one exact method per file, unique `unit_id` values within each file,
and the expected prompt/verification flags. It hashes each file and writes
only a registry manifest with file references, counts, and provenance.

The validator rejects mixed-method files, reusing one path under multiple
methods, duplicate prediction units, and verification fields inside an
Integrated Gradients file. It therefore keeps the prompt-visible Integrated
Gradients vector physically separate from verification vectors and prevents
accidental circularity through an implicit merged artifact.

Run it after the method-specific production artifacts exist:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 research/scripts/validate_attribution_storage.py \
  --integrated-gradients-file research/models/attributions/loginsight_lstm/seed_13_test_integrated_gradients.jsonl.gz \
  --integrated-gradients-file research/models/attributions/logrobust_attention_bilstm/seed_13_test_integrated_gradients.jsonl.gz \
  --occlusion-file research/models/attributions/loginsight_lstm/seed_13_test_occlusion.jsonl.gz \
  --leave-one-event-out-file research/models/attributions/loginsight_lstm/seed_13_test_leave_one_event_out.jsonl.gz
```

The default registry is
`research/models/attributions/attribution_vector_store.manifest.json`.
The focused tests are in
`research/scripts/test_validate_attribution_storage.py`. Step 39 is complete;
the validator does not fabricate missing production vector files.

### Step 40 — Define the structured explanation schema

Step 40 is complete. The machine-readable LLM payload is defined in
`research/experiments/llm_explanation_schema.json` (JSON Schema draft
2020-12). Every payload must declare its answer status, optional summary,
typed claims, cited event IDs, cited line IDs, per-claim confidence, overall
confidence, and an explicit abstention object. Claims may be marked abstained;
non-abstained claims must cite at least one event or line in the companion
validator.

`research/scripts/explanation_schema.py` validates already-decoded payloads
with standard-library checks for required fields, supported enums, confidence
ranges, unique citations, root/claim citation consistency, and agreement
between answer and abstention status. Raw JSON extraction, malformed-response
handling, and repair remain Step 48 work. Focused coverage is in
`research/scripts/test_explanation_schema.py`.

### Step 41 — Training-only contrastive retrieval and blinded prompt

Step 41 is complete. `research/scripts/training_only_retrieval.py` fits an
`event_tfidf_cosine` index using only normal HDFS training blocks. It never
uses query labels or attribution values for ranking, breaks equal scores by
the lowest candidate unit ID, and records a deterministic lowest-ID fallback
when a query has zero vocabulary overlap with the normal training index.

The retrieval ledger records the query and candidate IDs, similarity,
sequence lengths, source split, source-file hash, candidate normal-label
restriction, and explicit no-label/no-attribution provenance. It writes a
separate manifest and does not place retrieval metadata in the LLM prompt.

`research/scripts/blinded_prompt.py` converts evidence records into
prompt-visible event objects without BlockId or LabelId, then uses the
registered prompt-attribution boundary to expose only anomalous evidence, the
normal comparison block, and basic detector output. It rejects unregistered
attribution fields before rendering. The current similarity method is
provisional before registration and is now frozen by the OSF record at
https://osf.io/qrvuy/overview. Focused coverage
is in `research/scripts/test_training_only_retrieval.py` and
`research/scripts/test_blinded_prompt.py`.

### Step 42 — Unblinded prompt and contrastive factor

Step 42 is complete. `research/scripts/contrastive_prompt.py` implements the
registered 2×3 factor: normal comparison present/absent crossed with
attribution none/Integrated Gradients/shuffled Integrated Gradients. The six
cells map to the existing blinded, unblinded, and shuffled conditions plus
three explicit no-normal conditions. Each cell is passed through the strict
prompt boundary, so it cannot receive fields outside its registered
allowlist. Integrated Gradients and shuffled payloads are required only in
their corresponding cells; the none cell rejects attribution payloads.

The factor is recorded in `experiment_config.json` as provisional until
registration freezes it. No LLM calls are made. Focused coverage is in
`research/scripts/test_contrastive_prompt.py`.

### Step 43 — Shuffled Integrated Gradients prompt

Step 43 is complete. `research/scripts/shuffled_integrated_gradients.py`
creates prompt-visible shuffled attribution artifacts from Integrated
Gradients records. The provisional `within_block_permutation` mode moves
attribution values across active event positions while preserving event IDs
and positions; `cross_block_donor` transfers values from a distinct donor
block with the same active-event count. Both modes recompute absolute scores
and ranks, preserve padding alignment, and record source/donor provenance.

`build_shuffled_prompt_payload` strips artifact provenance and padding before
the shuffled values are supplied to the Step 42 prompt factor. The artifact is
marked prompt-visible and not verification-only; it remains separate from
occlusion and leave-one-event-out vectors. The primary shuffle type is
provisional until Step 62. Focused coverage is in
`research/scripts/test_shuffled_integrated_gradients.py`.

### Step 44 — No-detector-context prompt

Step 44 is complete. `research/scripts/no_detector_context_prompt.py` builds
the registered `no_detector_context` condition using only the anomalous raw
evidence. It does not accept detector output, normal comparison evidence, or
attribution inputs, and the evidence conversion removes storage metadata such
as `BlockId` and `Label` while retaining event and line citations required by
the structured explanation schema. The final payload passes through the
condition-aware prompt boundary and renders deterministically with template
ID `no_detector_context_v1`. Focused tests are in
`research/scripts/test_no_detector_context_prompt.py`.

### Step 45 — LLM model selection

Step 45 is complete at the model-family and access-mode level. The selected
pair is `mistralai/Mistral-7B-Instruct-v0.3` and
`Qwen/Qwen2.5-7B-Instruct`, both run locally or in Colab with Colab GPU as the
default runtime target. Mistral is the active replacement for the historical
GPT-OSS path because it loaded on the free Colab T4, while GPT-OSS 20B did not
fit that runtime. The Mistral and Qwen
licenses must be recorded with the experiment. The selection rationale and
required freeze fields are recorded in
`research/experiments/llm_model_selection.md` and
`research/experiments/experiment_config.json`. Exact checkpoint revisions,
runtime hardware, and decoding settings were captured in Step 46 before pilot
generation.

### Step 46 — Model and decoding capture

Step 46 is complete. Mistral and Qwen both passed synthetic-only smoke tests
on a Colab Tesla T4 using 4-bit loading. The Mistral checkpoint is pinned to
`c170c708c41dac9275d15a8fff4eca08d52bab71` and Qwen is pinned to
`a09a35458c702b33eeacc393d103063234e8bc28`. The reports record Python,
PyTorch, CUDA, GPU-memory, decoding, and response-hash metadata in
`research/results/tables/step46_mistral_smoke_test.json` and
`research/results/tables/step46_qwen_smoke_test.json`. No project logs or
secrets were sent to either smoke test. The prior GPT-OSS API smoke test is
historical only.

## Reviewer-required pre-pilot checks

### R1 — Measure exact event-sequence overlap

`research/scripts/sequence_overlap_audit.py` implements reviewer gate R1. It
defines a sequence as the exact ordered, unpadded event-ID `Sequence` list,
hashes those sequences, compares the registered HDFS train and test sets, and
reports detector metrics on the novel-test subset. It also counts annotation
items, unique blocks, and unique event sequences in the annotation pool.

The completed report is
`research/results/tables/sequence_overlap_audit.json`. Results are:

- Training: 460,049 blocks and 17,884 unique event sequences.
- Test: 57,506 blocks and 4,564 unique event sequences.
- Shared unique train/test sequences: 3,121, or 68.38% of test sequence types.
- Test blocks with sequences unseen in training: 1,459, or 2.54% of test blocks.
- Annotation pool: 240 items, 239 unique blocks, and 102 unique event sequences.

On the 1,459 novel-sequence test blocks, the available prediction artifacts
reported F1 0.8779 and average precision 0.9744 for LogInsight LSTM seed 13;
the LogRobust-inspired seed 202 artifact reported F1 0.9928 and average
precision 0.9993. These are descriptive novel-sequence metrics, not a claim
that the two artifacts are the registered primary seeds for the explanation
study.

The two focused tests are in
`research/scripts/test_sequence_overlap_audit.py`. R1 is complete; R2–R4 are
documented below. R5–R7 are deferred final reviewer checks in the
implementation checklist.

### R2 — Run shortcut baselines

`research/scripts/shortcut_baseline_audit.py` implements reviewer gate R2.
The event-count baseline is a logistic regression over training-vocabulary
bag-of-event-template counts. The length-only baseline is a logistic
regression over the number of events in the full unpadded sequence. Both fit
on the training split only, use class-balanced fitting, and use a fixed 0.5
threshold without test tuning.

The completed report is
`research/results/tables/shortcut_baseline_audit.json`. Test results are:

- Event-template-count logistic regression: F1 `0.9811`, average precision
  `0.9824`.
- Length-only logistic regression: F1 `0.1032`, average precision `0.3872`.
- LogInsight LSTM seed 13 artifact: F1 `0.9514`, average precision `0.9926`.
- LogRobust-inspired seed 202 artifact: F1 `0.9979`, average precision `0.9999`.

The event-count baseline is close to the LSTM detector, showing that event
composition alone is a strong predictive shortcut in this split. Explanation
claims must therefore distinguish evidence attributable to specific events
from performance that can be obtained from aggregate event-template counts.
R2 is complete; R3 and R4 are documented below. R5–R7 are deferred final
reviewer checks.

### R3 — Register primary explanation seeds

`research/scripts/select_primary_explanation_seeds.py` implements reviewer
gate R3. It reads all ten completed Step 29 run manifests and selects one
primary explained seed per architecture using validation F1 before any
explanation or attribution outputs are viewed. Ties are resolved by lower
validation loss and then lower seed. Test metrics are intentionally excluded
from the selection rule.

The completed report is
`research/results/tables/primary_explanation_seed_selection.json`:

- LogInsight LSTM: primary seed `13`, validation F1 `0.990280`; secondary
  stability seeds `73`, `101`, `42`, and `202`.
- LogRobust-style attention Bi-LSTM: primary seed `202`, validation F1
  `0.997330`; secondary stability seeds `101`, `42`, `13`, and `73`.

The selected primary checkpoint is the saved `best` checkpoint from each
corresponding run manifest. The four non-primary seeds for each architecture
remain available only for the registered secondary attribution-stability
analysis. Focused coverage is in
`research/scripts/test_select_primary_explanation_seeds.py`; R3 is complete.

### R4 — Audit Integrated Gradients stability and sanity

`research/scripts/audit_integrated_gradients.py` implements reviewer gate R4.
It consumes serialized Integrated Gradients artifacts without loading a model,
hashes every input artifact, reports baseline-probability and completeness-error
distributions, compares 16 versus 64 integration steps, keeps logit-space
attributions separate from probability-space attributions, and compares each
trained model with a deterministically randomized-weight control. It rejects
mixed prediction units or mismatched architecture, seed, split, target, or
output-space pairs. Occlusion and leave-one-event-out artifacts are not
accepted.

The completed report is
`research/results/tables/integrated_gradients_r4_audit.json`. It covers 16
matched HDFS test blocks for both registered primary explanation models:

- LogInsight LSTM seed 13 has strong 16-to-64-step stability: mean signed
  cosine `0.99993`, mean signed Spearman `0.99927`, and mean top-5 Jaccard
  `1.0`. Its mean probability-space completeness error falls from `0.05876`
  at 16 steps to `0.00261` at 64 steps.
- LogRobust-inspired seed 202 is not step-stable on this audit sample: mean
  signed cosine `0.35646`, mean signed Spearman `0.56853`, mean top-5 Jaccard
  `0.15179`, and median top-5 Jaccard `0.0`. Its mean probability-space
  completeness error is `0.30314` at 64 steps and its mean logit-space error
  is `2.33543`.
- The all-padding baseline is saturated for both models: mean baseline anomaly
  probability `1.0` for the LSTM and `0.99144` for LogRobust-inspired. This is
  a substantive limitation of the chosen baseline, not a passing quality
  result.
- Both randomized-model comparisons pass the descriptive sanity rule because
  trained-versus-randomized signed cosine is below same-model step stability.
  This does not override the LogRobust completeness and stability findings.

No attribution-quality threshold was pre-registered, so the report separates
audit coverage from scientific validity and preserves all architecture-specific
distributions. R4 is complete as an audit. Downstream interpretation must not
describe the LogRobust Integrated Gradients ranking as stable on the basis of
this sample, and the saturated all-padding baseline must be reported as a
limitation or addressed through a documented protocol decision.

Focused coverage is in
`research/scripts/test_audit_integrated_gradients.py`.

## Verification command

The current research test suite can be run with:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 -m unittest discover -s research/scripts -p 'test_*.py'
```

The current suite contains 170 passing tests, including the three Step 31
Integrated Gradients tests, the two Step 32 attribution tests, the four Step 30
cache tests, the three Step 29 orchestrator tests, the five Step 28 interface
tests, the three Step 26 model tests, the two Step 25 unit tests, and the four
Step 23 unit tests, the three Step 33 validation tests, the four Step 34
occlusion tests, the three Step 35 leave-one-event-out tests, and the six
Step 36 attribution-boundary tests, and the four Step 37 attribution-only
baseline tests, the four Step 38 random-event baseline tests, and the five
Step 39 attribution-storage tests, the two R1 sequence-overlap tests, the two
R2 shortcut-baseline tests, the three R3 seed-selection tests, the four R4
Integrated Gradients audit tests, and the four
Step 40 explanation-schema tests, the six Step 41 retrieval/prompt tests, the
six Step 42 contrastive-factor tests, the five Step 43 shuffle tests, and the
five Step 44 no-detector-context prompt tests.

## Important research rules preserved so far

- The unit of anomaly detection is the complete HDFS block, not an individual log line.
- BGL is evaluated as a documented fixed window, not as an HDFS block.
- The primary human outcome is human-evidence recovery.
- Detector faithfulness is a separate outcome.
- Only Integrated Gradients may be visible to the unblinded LLM.
- Occlusion and leave-one-out values remain verification-only.
- BGL alert prefixes never enter parser, detector, retrieval, attribution, or LLM input.
- Parser and vocabulary fitting must be training-only.
- Normal comparison retrieval must be training-only.
- Test thresholds must not be tuned on the test set.
- Split manifests, hashes, seeds, prompt conditions, and environment IDs must be recorded.
- The LLM explanation must not block the initial detector alert; verification is asynchronous/on-demand.

## Next implementation target

The next available implementation target is Step 51, the evidence-relevance
annotation guideline. Steps 47A and 47B are deferred execution tasks: the
Mistral full run remains blocked, while the Qwen smoke test is complete but its
full repeated pilot is pending. A suitable GPU runtime must pass bounded
real-record generation before those pilots are resumed. Confirmatory LLM
generation must use the exact protocol archived at
https://osf.io/qrvuy/overview.
