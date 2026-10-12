# LogRobust implementation deviations

This is the registered Step 27 deviation record for the second LogInsight-FV
detector. It documents the implementation as a LogRobust-style architecture
check, not as a byte-for-byte reproduction of the published experiment.

Reference: Xu Zhang et al., “Robust Log-Based Anomaly Detection on Unstable Log
Data,” ESEC/FSE 2019, DOI
[10.1145/3338906.3338931](https://doi.org/10.1145/3338906.3338931). The
publication’s central architectural contribution is semantic log-event
representation followed by an attention-based Bi-LSTM classifier. The
official publication summary is available from
[Microsoft Research](https://www.microsoft.com/en-us/research/publication/robust-log-based-anomaly-detection-on-unstable-log-data/).

The machine-readable record is
`research/experiments/logrobust_implementation_deviations.json`.

## What is preserved

The implementation preserves the published architectural idea: event-level
representations are processed by a bidirectional LSTM, additive temporal
attention pools the sequence, and a sigmoid classifier produces a binary
anomaly probability. The attention layer remains a model component, but its
weights are not treated as faithful explanations.

## Deviations and rationale

| Area | Published/reference expectation | LogInsight-FV implementation | Why it differs | Comparison consequence |
|---|---|---|---|---|
| Event representation | Semantic vectors for log events | Trainable embedding over the frozen training-only event vocabulary | Both detectors must receive the same leakage-controlled parser output; no external embedding corpus is introduced | Architecture-style comparison, not exact reproduction |
| Prediction unit | The paper’s Hadoop/service-log data construction | Labelled HDFS `BlockId` units | The registered endpoint is block-level evidence recovery | Published scores are contextual, not directly comparable |
| Splits | Publication-specific data and robustness experiments | Fixed block-disjoint train/validation/test splits with training-only vocabulary fitting | Prevents unit overlap and test leakage | Evaluation protocol is study-specific |
| Sequence shape | Publication-specific sequence handling | Maximum length 50, post-padding, pre-truncation | Bounded tensors and valid right-padded masks for TensorFlow Metal/cuDNN | Padding choice is explicit and reproducible |
| Training | Publication-specific training environment | Adam, balanced class weights, deterministic seed, validation early stopping, Keras checkpoints | Shared research runtime and reproducibility | Hyperparameters are not claimed to be paper-exact |
| Evaluation | Publication-specific threshold/reporting | Validation-only calibration and threshold selection, then test reporting | Same no-test-tuning protocol for both detectors | Final comparison uses common artifacts |
| Explanation | Attention can indicate learned predictive weighting | Integrated Gradients is the only prompt-visible attribution; attention is not declared faithful | Separates attention plausibility from faithfulness | Attention weights do not define the human explanation endpoint |

## Run-specific record

The seed-13 HDFS run used the configuration in
`research/experiments/logrobust_attention_bilstm_training_config.json` and
produced the manifest and checkpoint under
`research/models/checkpoints/logrobust_attention_bilstm/`. Early stopping
restored epoch-12 weights at epoch 15. Before calibration and validation-only
threshold selection, the test results at threshold 0.5 were accuracy 0.999513,
precision 0.985346, recall 0.998219, and F1 0.991740, with 25 false positives
and 3 false negatives.

These numbers are a detector-run record only. They must not be presented as a
direct reproduction of the published paper’s reported scores.

## Reproducibility pointers

- Trainer: `research/scripts/train_logrobust_attention_bilstm.py`
- Configuration: `research/experiments/logrobust_attention_bilstm_training_config.json`
- Experiment registry: `research/experiments/experiment_config.json`
- Run manifest: `research/models/checkpoints/logrobust_attention_bilstm/run_manifest_seed_13.json`
- Shared calibration: `research/scripts/calibrate_detector.py`
- Shared threshold selection: `research/scripts/select_threshold.py`
