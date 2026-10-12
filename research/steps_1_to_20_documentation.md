# LogInsight-FV Implementation Record: Steps 1–20

## Purpose

This document records what was implemented for Steps 1–20 of the LogInsight-FV research plan, what was verified, and what remains pending. It is intended to make the project recoverable if a future Codex conversation starts without the earlier chat context.

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
- Final annotation pool: not generated yet because detector confidence and architecture metadata do not exist until detector steps are completed

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
| 20 | IMPLEMENTED, POOL PENDING | Sampler and schema exist; final pool awaits detector metadata |

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

The sampler was unit-tested, including deterministic repeatability, missing-metadata rejection, bin boundaries, and target-size validation. The complete final pool is intentionally not generated yet because the research detector outputs do not exist. Creating placeholder confidence or architecture values would compromise the planned strata.

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
modules. Five focused interface tests and the complete 56-test research suite
pass. Step 28 is complete.

### Step 29 — Train both detectors with multiple seeds

The registered detector seeds are 13, 42, 73, 101, and 202. Seed 13 already
has completed HDFS runs for both detectors. The remaining four seeds for each
architecture must be trained before this step can be marked complete.

`research/scripts/train_multiseed_detectors.py` is the resumable orchestrator.
It reuses and validates complete existing seed artifacts, invokes each
architecture-specific trainer with the current Python interpreter, validates
the run manifest/checkpoints/history/predictions after every run, and writes a
progress manifest after every target. It never marks Step 29 complete by
itself; completion requires a manifest with every architecture/seed target
validated.

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
`research/results/tables/multiseed_training_manifest.json`. Step 29 remains
pending until that report shows all ten targets complete or reused and the
per-seed artifacts pass validation.

Once detector candidates exist, run:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 research/scripts/sample_annotation_pool.py \
  --candidate-metadata research/experiments/pilot/annotation_candidates.jsonl
```

Expected outputs:

- `research/experiments/pilot/annotation_pool.jsonl.gz`;
- `research/experiments/pilot/annotation_pool_manifest.json`.

## Verification command

The current research test suite can be run with:

```bash
cd /Users/neilvargheseabraham/Downloads/LogInsight
python3 -m unittest discover -s research/scripts -p 'test_*.py'
```

At the time this record was written, all 59 tests passed, including the three
new Step 29 orchestrator tests, the five
new Step 28 interface tests, the three Step 26 model tests, the two Step 25
unit tests, and the four Step 23 unit tests.

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

The next checklist target is Step 29: train both detectors with multiple
seeds. After both detector candidates exist, return to the Step 20 command to
create the final annotation pool.
