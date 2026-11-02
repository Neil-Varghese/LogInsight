# Step 75 — bounded ROAR-style retraining check

`research/scripts/roar_retraining.py` implements the limited remove-and-retrain
diagnostic described in the proposal. It targets the registered primary
LogInsight LSTM and deliberately does not claim to reproduce the full ROAR
benchmark.

For each selected fraction, the script applies the same neutral replacement to
the training, validation, and test sequences. The Integrated-Gradients variant
replaces the highest-absolute-attribution events. The random-event control
replaces the same number of active positions using a deterministic per-unit
seed. The unmodified model is trained as the reference. The default bounded
run uses 256 train, 128 validation, and 128 test units, with fractions 0, 0.10,
0.30, and 0.50; all values are recorded in the output report.

The neutral representation is the training vocabulary's
`__UNKNOWN_EVENT__` token, which preserves sequence length and avoids the
padding shortcut. Attribution artifacts are required for all three splits and
must align with the encoded event IDs and positions. The report stores input
hashes, subset limits, seeds, removed-event counts, per-split metrics, and
test-F1 changes from the unmodified reference.

The result is an OOD diagnostic. It must not be interpreted as a standalone
faithfulness endpoint, and a failed or incomplete run cannot support ROAR-level
claims.

Example command after train/validation/test Integrated-Gradients artifacts are
available:

```bash
python3 research/scripts/roar_retraining.py \
  --repo-root . \
  --train-attribution research/models/attributions/loginsight_lstm/seed_13_train_integrated_gradients.jsonl.gz \
  --validation-attribution research/models/attributions/loginsight_lstm/seed_13_validation_integrated_gradients.jsonl.gz \
  --test-attribution research/models/attributions/loginsight_lstm/seed_13_test_integrated_gradients.jsonl.gz \
  --output research/results/tables/step75_roar_check.json
```

The current repository has the annotation-pool test attribution artifact but
does not yet have the train and validation attribution files required by this
command. Generate those bounded artifacts first, then run this check in a
TensorFlow environment.
