# LogInsight Project Knowledge Graph

This file is the persistent project memory for future Codex chats. Read it before making research implementation changes.

Audit date: 2026-10-11

Current working branch: `research-implementation`

Current working commit: `3375f87 Add Gemini anomaly explanations`

## Branch graph

```text
3fdd974  Stop tracking node_modules
   |
   +-- main / origin/main / research-implementation
   |     3375f87 Add Gemini anomaly explanations
   |
   +-- origin/llm_integration_and_dashboard
         4a08558 Gemini explanations, live mode and Next.js dashboard
         3e9fc30 real-time monitor and production dashboard
         aca7ffc fake Docker HDFS cluster as a live log source
         b27e0a1 traffic feed, executive tab, severity ranking, dashboard theme
         8a64141 landing page, real login and Executive/dashboard polish
```

### Branch roles

| Branch | Role | Important contents |
|---|---|---|
| `main` | Stable batch implementation | HDFS BlockId parsing, LSTM prediction, static frontend, LLM explanations |
| `origin/llm_integration_and_dashboard` | Expanded product branch | Authentication, Next.js dashboard, live monitoring, traffic/simulation views, expanded server API |
| `research-implementation` | Research development branch | Currently starts from `main`; research code and artifacts should be added here selectively |
| `backup-before-history-cleanup` | Historical backup | Common ancestor-era repository state |

The two major implementation branches share ancestor `3fdd974`. The dashboard branch is not a small commit on top of the current `main`; it is a separate product-development line with approximately 68 changed paths and about 14,000 added lines relative to `main`.

## Core code knowledge graph

```text
test sets/*.log
      |
      v
frontend/app.js  --HTTP-->  backend/server.py
                                  |
                                  +--> backend/predictor.py
                                  |       |
                                  |       +--> Drain parser
                                  |       +--> BlockId grouping
                                  |       +--> event2idx.pkl
                                  |       +--> models/lstm_model(v2).keras
                                  |       +--> HDFS.log_templates.csv
                                  |
                                  +--> backend/runs/<job>.sqlite3
                                  |       +--> blocks table
                                  |       +--> explanations table
                                  |
                                  +--> backend/explainer.py
                                          |
                                          +--> Gemini API when GEMINI_API_KEY exists
                                          +--> OpenAI-compatible API fallback
```

### Main branch nodes

| Node | Type | Responsibility | Important details |
|---|---|---|---|
| `backend/predictor.py` | Detector pipeline | Parses raw HDFS logs and scores complete BlockId sequences | Uses Drain, `MAX_SEQUENCE_LENGTH = 50`, threshold `0.5`, and the current LSTM |
| `backend/explainer.py` | LLM adapter | Generates structured anomaly explanations | Uses Gemini first; supports OpenAI-compatible/Groq fallback |
| `backend/server.py` | Batch API/server | Runs prediction jobs and serves block results | Creates SQLite run databases and exposes `/api/run`, job, block, and explanation routes |
| `frontend/app.js` | Batch frontend | Starts runs, polls progress, lists blocks, requests explanations | Talks to the main server through `/api/...` |
| `research/preprocessing_pipeline.ipynb` | Research notebook | Existing preprocessing investigation | Source of current event parsing and dataset preparation knowledge |
| `research/LSTM.ipynb` | Research notebook | Existing LSTM modelling work | Source of current model/training assumptions |
| `research/event2idx.pkl` | Artifact | Event-template-to-index mapping | Must become compatible with the research unknown-event token |
| `research/output/HDFS.log_templates.csv` | Artifact | Master event templates | Used by prediction-time event resolution |
| `models/lstm_model(v2).keras` | Model artifact | Current deployed LSTM | Engineering baseline, not yet the research detector protocol |

## Current main prediction flow

1. The frontend lists files inside `test sets/`.
2. The user starts `/api/run` with a selected log file.
3. `backend/server.py` creates a job and SQLite database.
4. `backend/predictor.py` parses the file with Drain.
5. Log lines are grouped by `BlockId` and ordered by `LineId`.
6. Event IDs are mapped through `event2idx.pkl`.
7. Unknown event variants are currently skipped when no mapping is found.
8. Sequences are padded to length 50 and scored by the LSTM.
9. Block scores, labels, event IDs, and raw evidence are stored in SQLite.
10. The frontend requests an explanation for an anomalous block.
11. `backend/explainer.py` sends the block evidence to the configured LLM and returns structured JSON.

## Current LLM behavior

### Main branch

- Environment variables include `GEMINI_API_KEY`, `GEMINI_MODEL`, `LOGINSIGHT_LLM_BASE_URL`, and `LOGINSIGHT_LLM_MODEL`.
- Gemini is preferred when configured.
- The fallback uses an OpenAI-compatible Responses-style API.
- The current schema contains `explanation`, `possible_cause`, and `evidence`.
- The current prompt receives anomaly score, event IDs, and raw logs.
- It does not yet receive Integrated Gradients, a normal comparison block, claim IDs, uncertainty fields, or abstention fields.

### Dashboard branch

- `backend/explainer.py` is a simpler Gemini-only adapter.
- It masks IPv4 addresses before sending raw lines.
- Its schema contains `summary`, `likely_cause`, `confidence`, `incomplete_sequence`, and `suggested_checks`.
- Its prompt explicitly treats raw logs as untrusted evidence.
- It does not yet implement the registered blinded/unblinded/IG-only conditions.

Research implementation decision: use the main branch’s model-loading and provider flexibility as the starting point, then replace the explanation interface with the registered research schema. Do not merge dashboard behavior into the research pipeline unless a later target explicitly requires it.

## Dashboard branch knowledge graph

```text
dashboard Next.js UI
        |
        +--> /api/auth/* ------> backend/auth.py ------> auth.sqlite3
        |
        +--> /api/run, /api/jobs/* ------> backend/server.py
        |
        +--> /api/monitor/* ------> backend/monitor.py ------> monitor.sqlite3
        |
        +--> /api/sim/* ------> backend/simulate_live.py / docker-compose.yml
```

The dashboard branch adds:

- cookie/session authentication and SQLite-backed accounts;
- a built Next.js dashboard under `dashboard/`;
- live file monitoring with incremental block scoring;
- traffic and IP activity summaries;
- replay and Docker simulation modes;
- protected API routes and public landing-page statistics.

The live monitor matches new lines against stored templates and rescoring only changed blocks. It is useful product infrastructure, but it is not the primary research evaluation path. The research paper should first establish offline, leakage-free, block/window-level results.

## Research implementation constraints

These are the rules future implementation chats must preserve:

1. HDFS prediction unit: complete `BlockId` block, not individual log lines.
2. BGL prediction unit: explicitly documented session or fixed window, not an HDFS block.
3. Primary human outcome: human-evidence recovery.
4. Separate detector outcome: detector faithfulness under matched perturbations.
5. Unblinded LLM input: Integrated Gradients only.
6. Occlusion and leave-one-out: stored for verification only; never sent to the LLM.
7. Core conditions: blinded, unblinded, shuffled-IG, attribution-only, random, and no-detector-context.
8. Unknown-event token: present during detector training before test or perturbation use.
9. Data splitting: block-disjoint and temporal checks where possible.
10. Retrieval: normal comparison examples must come from training data only.
11. BGL label prefixes: removed before parsing, detector input, retrieval, attribution, or LLM input.
12. Registration: H1, TOST margin, power, prompts, exclusions, and analysis must be registered before pilot LLM output.

## Known gaps to implement

- Current predictor skips unknown event variants; replace this with the learned unknown-event token.
- Current predictor uses one deployed LSTM; add the published LogRobust-style attention Bi-LSTM.
- Current LLM prompts are not condition-configurable; implement the registered prompt matrix.
- Current explanations do not use contrastive normal-block retrieval.
- Current explanations do not have independent human-evidence labels.
- Current run databases do not contain the full experiment ledger required by the proposal.
- Current verification does not have deletion, replacement, masking, resampling, sufficiency, or ROAR-style checks.
- The primary HDFS evaluation split is block-disjoint, and the separate strict temporal sensitivity artifacts are now complete for both HDFS and BGL.
- BGL preprocessing is complete. Automated leakage checks are implemented in `research/scripts/validate_leakage.py`; the runtime retrieval and parser-manifest checks remain pending until those later artifacts exist.

## Context recovery instructions for future Codex chats

Before changing code:

1. Read this file.
2. Read `research/implementation_steps.txt`.
3. Check the current branch and working-tree status.
4. Identify the target number being implemented.
5. Modify only the files needed for that target.
6. Add or update tests where applicable.
7. Record the result, files changed, and verification command in the target’s final response.
8. Update this knowledge graph when the architecture or experiment protocol changes.

Current completed implementation targets:

- Persistent implementation record for Steps 1–20: `research/steps_1_to_20_documentation.md`.

- Phase 1, Step 1: research branch created and pushed.
- Phase 1, Step 2: repository branches and architecture audited.
- Phase 1, Step 3: environment provenance recorded. Local direct dependencies are pinned in `research/requirements-local.txt`; the historical Colab training evidence and missing-version limitations are recorded in `research/colab_environment_record.txt`.
- Phase 1, Step 4: research directories created for data, models, prompts, experiments, results, and scripts. Existing root-level artifacts were intentionally not moved yet.
- Phase 1, Step 5: central experiment configuration created at `research/experiments/experiment_config.json`. It records datasets, leakage constraints, detectors, seeds, prompt conditions, attribution visibility, perturbation strategies, annotation outcomes, and provisional registration values.
- Phase 1, Step 6: experiment ledger schema created at `research/experiments/experiment_ledger_schema.json`. It separates human-evidence recovery, claim support, detector faithfulness, and perturbation consistency, while recording provenance, prompts, perturbations, timing, cost, and quality checks.
- Phase 1, Step 7: shared reproducibility utilities created at `research/scripts/seed_utils.py`, with tests in `research/scripts/test_seed_utils.py`. The utility seeds Python, NumPy, and TensorFlow, derives stable sub-run seeds, and reports deterministic-operation status.
- Phase 2, Step 8: HDFS was identified as the official LogHub `HDFS_v1` variant using the official 11,175,629-line/1.47 GiB description. The user confirmed the original sources as the LogHub repository and its Zenodo HDFS_v1 archive link. The local raw file checksum and auxiliary artifact checksums are recorded in `research/data/manifests/hdfs_v1_manifest.json`.
- Phase 2, Step 9: HDFS format inspection completed. The raw log matches `<Date> <Time> <Pid> <Level> <Component>: <Content>`, all 11,175,629 lines contain a `blk_-?\\d+` BlockId, 575,061 BlockIds are covered by `anomaly_label.csv`, and the parsed representation contains LineId, raw fields, EventId, EventTemplate, and ParameterList. Details are in `research/data/manifests/hdfs_v1_format_report.json`.
- Phase 2, Step 10: reusable HDFS block construction completed and verified by the user in `research/scripts/build_hdfs_blocks.py`. The full run processed 11,175,629 input rows, found BlockIds in every row, created 575,061 labelled blocks, and reported zero partial input. It extracts BlockIds from structured `Content`, sorts events by `LineId`, joins labels, and writes one compressed JSONL record per block. The script includes a dependency-free progress bar for long terminal runs.
- Phase 2, Step 11: explanation-ready evidence construction completed and verified by the user in `research/scripts/build_hdfs_block_evidence.py`. The full run processed 11,175,629 rows, preserved evidence for 11,175,629 events, and created 575,061 labelled blocks with zero missing BlockIds and zero partial input. It keeps the detector-compatible event-ID `Sequence` and adds per-event `RawLine`, `EventTemplate`, parsed `ParameterList`, `ParameterListRaw`, `LineId`, `EventId`, and zero-based `Position`. The compressed artifact passed gzip validation and is accompanied by `hdfs_block_evidence_manifest.json`.
- Phase 2, Step 12: official BGL download and verification completed by the user with `research/scripts/download_verify_bgl.py`. The Zenodo v8 archive was downloaded with MD5 `4452953c470f2d95fcb32d5f6e733f7a`, extracted to `research/data/raw/BGL.log`, and verified at 4,747,963 lines. The extracted raw-log SHA-256 is `666130b15ef44eb32fd02bd053e6c6e007c37696b5e7e8b9d8e45b729876a5d2`. Full provenance is recorded in `research/data/manifests/bgl_manifest.json`.
- Phase 2, Step 13: BGL label-prefix stripping completed and verified by the user with `research/scripts/strip_bgl_label_prefix.py`. All 4,747,963 lines were processed; 348,460 alert lines and 4,399,503 normal lines were identified across 42 prefixes. The parser-safe `BGL_clean.log` contains no label prefixes, while `BGL_label_audit.jsonl.gz` retains each original line, removed prefix, and alert indicator. Hashes and leakage-control metadata are recorded in `research/data/manifests/bgl_label_strip_manifest.json`.
- Phase 2, Step 14: BGL evaluation-unit construction completed and verified by the user with `research/scripts/build_bgl_windows.py`. The frozen rule produced 27,670 non-overlapping 60-second Unix-epoch windows from all 4,747,963 lines: 1,838 alert windows and 25,832 normal windows, with zero partial input. The compressed window artifact passed validation and the complete rule is recorded in `research/data/manifests/bgl_window_manifest.json` and the experiment configuration.
- Phase 2, Step 15: explicit unknown-event vocabulary support implemented in `research/scripts/event_vocabulary.py` and `research/scripts/build_event_vocabulary.py`. Research vocabularies reserve `__PAD__` at ID 0 and always include a learned `__UNKNOWN_EVENT__` row; unseen validation, test, and perturbation events map to that row instead of being skipped. The existing engineering `research/event2idx.pkl` is intentionally unchanged. Training-split vocabulary generation and model retraining will be exercised in later detector steps.
- Phase 2, Step 16: shared preprocessing validation completed and verified by the user with `research/scripts/validate_preprocessing.py`. The full run passed for 575,061 HDFS blocks and 27,670 BGL windows, covering 11,175,629 HDFS events and 4,747,963 BGL lines. It validates HDFS evidence ordering, event/sequence alignment, labels, lengths, and raw evidence fields; BGL window ordering, non-overlap, duration, line ranges, labels, and lengths; and manifest count consistency. Malformed-record tests are included.
- Phase 2, Step 17: deterministic HDFS split generation completed and verified by the user with `research/scripts/split_hdfs_blocks.py`. Using seed 2026 and registered 80/10/10 fractions, it created 460,049 training blocks, 57,506 validation blocks, and 57,506 test blocks. Label counts were train 13,470 Anomaly/446,579 Normal, validation 1,684/55,822, and test 1,684/55,822. Gzip validation and an independent cross-file check confirmed exactly 575,061 unique BlockIds with no cross-split duplicates. Assignment/output hashes are recorded in `research/data/manifests/hdfs_split_manifest.json`.
- Phase 2, Step 18: temporal splits completed and independently verified with `research/scripts/create_temporal_splits.py`. The strict 80/10/10 chronological policy produced HDFS train/validation/test counts of 394,619/55,784/57,505, excluding 67,153 boundary-crossing blocks; BGL produced 22,136/2,767/2,767 windows with no exclusions. HDFS and BGL split files contain unique prediction units, match their assignment manifests, and maintain strict temporal ordering. Cutoffs, label counts, hashes, and exclusion records are stored in `research/data/manifests/temporal_split_manifest.json`.
- Phase 2, Step 19: automated leakage audit completed and passed with `research/scripts/validate_leakage.py`. Primary HDFS, temporal HDFS, and temporal BGL split bundles had zero duplicate or misplaced units; HDFS temporal exclusions matched the 67,153 boundary-crossing assignments; all 4,747,963 BGL parser-safe lines had valid epoch-first input and no alert-prefix leakage; and the configured parser/vocabulary and normal-retrieval policies require training-only fitting/source data. The manifest is `research/data/manifests/leakage_check_manifest.json`. Optional runtime retrieval-ledger and parser-manifest checks are recorded as pending until those later artifacts are created.
- Phase 2, Step 20: reproducible annotation-pool sampler implemented in `research/scripts/sample_annotation_pool.py`, with candidate schema `research/experiments/annotation_candidate_schema.json`. It requires held-out candidate metadata containing dataset, prediction-unit ID, label, sequence length, detector confidence, and architecture; stratifies by dataset, label, length bin, confidence bin, and architecture; and uses seed 2026 with a default 240-item target. The final pool is intentionally pending until the research detectors produce valid confidence/architecture metadata; the sampler refuses to fabricate those values.
- Phase 3, Step 21: reusable leakage-free LogInsight LSTM training/evaluation implemented in `research/scripts/train_loginsight_lstm.py`, configured by `research/experiments/lstm_training_config.json`. It consumes split-specific HDFS evidence and a training-only vocabulary manifest, maps unseen events to the learned unknown-event row, rejects cross-split prediction-unit overlap, preserves the notebook's embedding/two-LSTM/batch-normalization/dropout architecture, records deterministic seed and configuration metadata, saves best/final checkpoints, emits split-wise prediction metadata, and supports evaluation-only checkpoint reuse. Full training remains a user-run long command; no fabricated detector confidence or architecture metadata has been created.
- Phase 3, Step 22: validation-only temperature scaling implemented and executed with `research/scripts/calibrate_detector.py`. It fits one positive temperature on validation predictions only, applies it to train/validation/test prediction files, reports NLL, Brier score, and ECE before/after calibration, writes calibrated predictions and a calibration manifest, and deliberately leaves threshold selection to Step 23. The seed-13 HDFS temperature was 1.272477891888822 and the artifacts are under `research/models/checkpoints/loginsight_lstm/calibrated/`.
- Phase 3, Step 23: validation-only threshold selection implemented and executed with `research/scripts/select_threshold.py`. It maximizes validation F1, breaks ties by closeness to the 0.5 reference and then the lower threshold, applies the frozen threshold to all splits, writes thresholded predictions and a manifest, and records that test labels were not used for selection. The seed-13 selected threshold was 0.35207340985154245.
- Phase 3, Step 24: the user trained the leakage-free HDFS LogInsight LSTM with seed 13 using the M4 Pro GPU. Early stopping restored epoch-1 weights after epoch 4. The provisional 0.5 test F1 was 0.989399; after validation-only calibration and threshold selection at 0.35207340985154245, descriptive test F1 was 0.989697 with 32 false positives and 3 false negatives. Artifacts are under `research/models/checkpoints/loginsight_lstm/`.
- Phase 3, Step 25: baseline reporting implemented in `research/scripts/record_lstm_baseline.py` and verified with two focused tests plus the full 48-test suite. The completed seed-13 HDFS baseline is recorded in `research/results/tables/lstm_baseline_seed_13.json`, including average precision as PR-AUC, precision, recall, F1, false-positive rate, and raw/calibrated NLL, Brier, and ECE across train, validation, and test. The held-out test PR-AUC is 0.994214 and thresholded F1 is 0.989697; the threshold was selected on validation labels only.
- Phase 3, Step 26: LogRobust-style attention Bi-LSTM integration implemented in `research/scripts/train_logrobust_attention_bilstm.py`, configured by `research/experiments/logrobust_attention_bilstm_training_config.json`. The model uses the frozen training-only event vocabulary, a trainable event embedding, masked additive temporal attention over a bidirectional LSTM, a dense classifier, and the shared deterministic/checkpoint/evaluation protocol. The published-method deviations are explicit: external semantic embeddings are not used; the study uses the shared trainable event embedding for leakage control and architecture comparability; and post-padding is used for TensorFlow Metal/cuDNN mask compatibility. The implementation passed a real masked one-epoch smoke fit in the project environment, its focused tests, and the full 51-test suite. The user completed seed-13 training: early stopping restored epoch-12 weights at epoch 15, and the provisional 0.5 HDFS test F1 was 0.991740 with 25 false positives and 3 false negatives. Artifacts are under `research/models/checkpoints/logrobust_attention_bilstm/`.
- Phase 3, Step 27: all LogRobust implementation deviations are recorded in `research/experiments/logrobust_implementation_deviations.json` and `research/experiments/logrobust_implementation_deviations.md`, with registry pointers in `research/experiments/experiment_config.json`. The record separates the preserved attention-Bi-LSTM architecture from the study-specific event vocabulary, HDFS block/split protocol, post-padding, training, calibration/threshold, and explanation-interpretation choices, and explicitly prohibits direct numerical comparison with published scores from incompatible protocols.
- Phase 3, Step 28: shared detector interface implemented in `research/scripts/detector_interface.py`. Both detector paths now use the same architecture-neutral batch validation, probability validation, binary metrics, and prediction serialization contract; model construction remains architecture-specific. Five interface tests were added, and the full research suite passes with 56 tests.
- Phase 3, Step 29: multi-seed orchestration scaffold implemented in `research/scripts/train_multiseed_detectors.py` with three tests. It registers seeds 13, 42, 73, 101, and 202 for both detectors, reuses complete seed-13 artifacts, trains missing targets sequentially with the current interpreter, validates checkpoints/manifests/predictions, and writes a resumable report. The step remains pending until all ten architecture/seed targets are validated by the user-run experiment.

## Research directory layout

```text
research/
├── data/{raw,processed,manifests}
├── models/{checkpoints,attributions}
├── prompts/
├── experiments/{pilot,main}
├── results/{tables,figures}
└── scripts/
```

## Reproducibility warning

The original LSTM notebook was executed in Google Colab with Python 3.12.13 and the saved `models/lstm_model(v2).keras` archive reports Keras 3.13.2. The notebook does not record the TensorFlow or scikit-learn versions, nor the GPU/runtime image. The current local environment is Python 3.9.6 with TensorFlow 2.20.0 and Keras 3.10.0. These must be treated as separate environment IDs in later experiments. Before any new Colab training run, execute the capture cell in `research/colab_environment_record.txt` and save the complete `pip freeze` output.
