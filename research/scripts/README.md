# Research scripts

Place deterministic preprocessing, training, attribution, perturbation,
evaluation, and validation scripts here. Scripts should accept explicit input
and output paths and record the configuration/run ID they use.

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
