# LogInsight-FV annotation workflow

**Workflow version:** 1.0  
**Primary human outcome:** evidence relevance  
**Secondary human outcome:** claim support  
**Default sampling seed:** 2026  
**Required independent annotators:** 2 per core unit or claim

This workflow operationalizes the evidence-relevance and claim-support
guidelines. Human labels are primary. Evidence relevance is collected before
any LLM condition is assigned to the core units. Claim support is collected
later on a separate, pre-specified sample of generated explanations. The two
outcomes are stored, adjudicated, and analyzed separately.

## Roles

- **Study coordinator:** freezes the guideline and protocol version, creates
  anonymized packets, runs the sampling and randomization scripts, maintains
  the restricted mapping, and records all workflow events.
- **Evidence annotators:** independently label every event occurrence and the
  optional contrastive assessment for assigned core units.
- **Claim-support annotators:** independently label claims in the selected
  explanation sample using the claim-support guideline.
- **Adjudicator:** resolves disagreements after independent labels are locked;
  the adjudicator sees the evidence and written guidelines, not condition or
  attribution metadata.
- **Data steward:** maintains immutable exports, hashes, access controls, and
  the mapping between study-local aliases and source identifiers.
- **Optional LLM judge:** performs an exploratory comparison only after human
  adjudication. It cannot define a primary label, replace an annotator, or
  adjudicate disagreements.

One person may hold more than one operational role only when that person does
not see blinded condition metadata during annotation and the separation is
recorded. The adjudicator must not use detector scores or attribution values to
resolve a human disagreement.

## Workflow gates and order

### 1. Freeze the materials

Before labeling begins, the coordinator records:

- the exact guideline versions for evidence relevance and claim support;
- the protocol/configuration revision and sampling seed;
- the annotation-packet builder version;
- the target pool manifest and stratum definitions;
- the consent, compensation, anonymization, and approval or exemption status
  required by Step 56; and
- the export schema and storage location.

No core labels are collected while a guideline or packet format is changing.
Practice items may be revised, but they are never silently promoted into the
core analysis set.

### 2. Sample the core pool before explanations

Use the registered test split and the existing annotation-pool sampler. The
target is the configured 200–300 blocks/windows, with the current planning
target of 240 when all strata are available. Stratify by dataset, normal versus
anomalous label, sequence length, detector confidence, and architecture where
sample size permits. Include naturally occurring and pre-specified synthetic
cases.

The sampling manifest must record the seed, eligible source files, stratum
counts, selected unit IDs, and hashes. No LLM output, explanation citation,
attribution ranking, or perturbation score may influence pool membership.

### 3. Build blinded evidence packets

For each selected unit, create an annotator-facing packet containing the
chronological event occurrences, stable event positions, templates, parameters,
raw lines, and any permitted normal comparison. Replace source identifiers
with study-local aliases where needed. Remove condition names, detector
architecture and seed identifiers, detector probabilities and thresholds,
Integrated Gradients, occlusion and leave-one-out values, LLM outputs,
experiment filenames, and hypothesis text.

The restricted provenance ledger may retain the original identifiers and
condition metadata, but that ledger is inaccessible to annotators and is never
joined into the label export before blinding checks pass.

### 4. Calibrate annotators on practice items

Give each annotator the written guideline, a short orientation, and practice
packets that are not in the core pool. Practice items must include clear
positive cases, routine context, repeated templates, ambiguous/truncated lines,
and contrastive differences caused only by identifiers. Annotators submit
practice labels independently. The coordinator discusses rule interpretation
without revealing model outputs or study conditions, then records the final
training version and any clarified wording.

Practice agreement is a training diagnostic and is not reported as core
inter-annotator agreement. Annotators who cannot apply the rules consistently
are retrained before receiving core units.

### 5. Collect independent evidence-relevance labels

Assign every core unit to two annotators. Assignments are balanced across
annotators and strata using the registered seed. Annotators work independently
and must not discuss individual units before submitting their records.

Each annotator labels every event occurrence as `relevant`, `not_relevant`, or
`uncertain`, adds the permitted role/reason, and completes the contrastive
assessment when a comparison is present. Empty relevant sets are valid. The
export must include all event occurrences, not just selected events.

The coordinator runs a completeness check after submission: one label per
event position, valid line/event references, no duplicate position records,
valid guideline version, and no forbidden condition or attribution fields.
Incomplete records are returned for completion without exposing the other
annotator's labels.

### 6. Lock, measure, and adjudicate evidence labels

After both independent submissions are locked, calculate the pre-specified
agreement statistic at event level and for the contrastive label where
applicable. The statistic, minimum threshold, and remediation rule are frozen
in Step 59; this workflow does not change them after seeing results.

Retain disagreements by event position. The adjudicator reviews only the
blinded packet, the two labels, and the evidence-relevance guideline, then
records a final binary label, disagreement category, and short rationale.
`uncertain` labels are resolved here or receive an explicit exclusion reason;
they are never silently converted to `relevant` or `not_relevant`.

The finalized evidence set is locked before any core unit is assigned to an
LLM condition. The same locked units are then paired across all primary
conditions.

### 7. Generate and randomize explanation conditions

After evidence labels are locked, run the registered LLM conditions on the
same core units. Keep the anomalous unit, normal comparison, detector, LLM
model, decoding configuration, and explanation schema fixed within each paired
comparison. Randomize condition and repetition order with the registered seed
for execution-order balance; order randomization must not alter condition
assignment or evidence labels.

The ledger records the condition, repetition, unit, detector stratum, prompt
template, and randomization seed. Annotators never receive these fields.

### 8. Sample and annotate claim support separately

Select the claim-support sample before inspecting claim quality or outcomes.
Use a pre-specified stratified sampling rule over dataset, anomaly label,
condition, detector architecture, confidence, explanation status, and block or
window length where sample size permits. The sample is drawn from completed
generated explanations, not from the evidence-relevance label set, and must
include the registered conditions represented in the target sample.

Create a separate support packet containing the anonymized explanation, atomic
claims, cited lines/events, target evidence, normal comparison when relevant,
and the recorded detector output only when needed to check a detector-specific
claim. Remove condition, attribution, model/seed, and experiment identifiers.

Two claim-support annotators independently label each included atomic claim as
`supported`, `unsupported`, or `uncertain`, with a support basis, issue
category, and rationale. Abstentions are recorded separately and are not
forced into the factual-claim denominator. Run completeness checks and lock
the independent labels before adjudication.

### 9. Adjudicate and lock claim-support labels

The adjudicator reviews disagreements using only the support packet and the
claim-support guideline. They record the final label and issue category while
preserving both original labels. Report total, supported, unsupported, and
uncertain claims, support rate, hallucination rate, abstention counts, and the
number and categories of adjudicated disagreements.

Claim support remains a secondary outcome. It is never used to change the
primary evidence-relevance labels, select the annotation pool, alter the
prompt conditions, or rescue a failed primary endpoint.

## Optional LLM judge

An LLM judge may be run only as an exploratory secondary comparison after the
human labels and adjudications are locked. Before running it, freeze and
record the judge model/version, prompt, decoding settings, input fields,
random seed or repeat index, and calibration sample.

The judge must be calibrated against a held-out subset of adjudicated human
labels. Report agreement, disagreement cases, and failure categories. The
judge cannot replace human labels, define the primary evidence-recovery or
claim-support endpoint, set an exclusion rule after seeing results, or resolve
human disagreements.

## Failure, correction, and audit rules

- If a packet contains detector, attribution, condition, or source-identifier
  leakage, quarantine it, record the leakage type, repair or rebuild the packet,
  and repeat that unit's independent annotation.
- If a label is edited after submission, preserve the original record, editor,
  timestamp, reason, and guideline version. Do not overwrite an audit artifact.
- If an annotator withdraws or a record is incomplete, mark the unit pending and
  assign a replacement annotator without revealing the first label.
- If a source line is malformed, truncated, or unavailable, use the guideline's
  `uncertain`/exclusion rule and record the reason; do not reconstruct evidence
  from memory.
- Keep raw packets, blinded exports, adjudicated exports, sampling manifests,
  randomization records, and hashes as separate versioned artifacts.

The final analysis package must make it possible to reconstruct which units
were sampled, who independently labelled each unit using pseudonymous IDs,
which records were adjudicated or excluded, and which conditions were later
run on the locked core pool.

This document implements Step 53. The evidence-relevance definitions remain in
`evidence_relevance_annotation_guideline.md`; the claim-support definitions
remain in `claim_support_annotation_guideline.md`; interface construction and
blinding probes are implemented by Steps 54 and 55. Step 56 ethics
requirements, the approval/exemption decision record, and the pre-labeling
gate are documented in `annotation_ethics_documentation.md`.
