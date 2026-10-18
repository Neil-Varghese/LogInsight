# LogInsight-FV preregistration packet — Step 63

Status: local protocol freeze retained; an OSF preregistration draft has been
created and is pending submission. Confirmatory execution remains blocked until
the OSF registration is submitted and its permanent URL is recorded.

## Primary hypothesis and endpoint

The primary comparison is the paired difference in macro human-evidence-recovery
F1 between the unblinded Integrated-Gradients condition and the blinded
condition. The registered equivalence margin is 0.05 absolute F1. A secondary
comparison repeats the test against the shuffled-Integrated-Gradients control.

Human-evidence recovery and detector faithfulness are separate outcomes. The
primary endpoint is evaluated on independently annotated evidence and is not
replaced by detector perturbation scores.

## Study design

- Primary dataset/unit: HDFS complete `BlockId` blocks.
- Transfer dataset/unit: BGL non-overlapping 60-second windows.
- Primary detector/LLM stratum: LogInsight LSTM with `llm_primary`.
- Secondary architecture/model checks: LogRobust-style attention Bi-LSTM and
  `llm_open_weight`, reported separately.
- Conditions: blinded, unblinded Integrated Gradients, shuffled Integrated
  Gradients, attribution-only, random-event, and no-detector-context.
- Primary evidence budget: top 5 non-padding events; shorter units use all
  available events.
- Primary shuffle: within-block attribution permutation with seed 2026.
- Normal replacement: same active event position in the retrieved normal block;
  if no counterpart exists, leave the target unchanged and record unavailable.

## Statistical plan

The primary analysis uses paired TOST with alpha 0.05 and equivalence margin
0.05. The planning calculation assumes paired SD 0.15 and 80% power, requiring
71 independent pairs; at least 100 usable annotated items are targeted.
Per-unit evidence F1 uses 1.0 when both prediction and reference are empty and
0.0 when exactly one is empty. Detector false positives are reported as a
separate detector outcome and are not added to human evidence F1.

When strata are combined for sensitivity analysis, condition, detector, LLM,
dataset, and their registered interactions are fixed effects with a random
intercept for prediction-unit block/window ID. Primary HDFS results are not
pooled with BGL in the primary TOST.

## Exclusions and quality controls

Invalid or unparseable explanations are excluded from the primary metric only
when the predeclared quality-exclusion rule applies; every exclusion and its
reason are recorded. Occlusion and leave-one-event-out values are verification-
only and cannot enter an LLM prompt. Outputs from the exploratory pilot must
not be used for confirmatory claims or presented as preregistered results.

## Frozen artifact manifest

The exact file hashes and current code revision are recorded in
`step63_preregistration_manifest.json`. The manifest must be regenerated after
the final commit and attached to the external OSF or AsPredicted record.

## Registration record

- Platform: OSF
- Draft status: prepared, not yet submitted
- Draft identifier: `6ad4a425ec9694945ececcab`
- URL: pending permanent OSF registration URL
- Registered timestamp: pending submission
- Registration version: `step63-v1`

The repository contains prior detector development and limited exploratory
model smoke tests. These are disclosed as foreknowledge and are not treated as
confirmatory results. No confirmatory LLM generation, independent human-
evidence outcome analysis, or confirmatory perturbation-faithfulness analysis
will begin until the submitted OSF registration URL and timestamp are recorded
in the manifest.
