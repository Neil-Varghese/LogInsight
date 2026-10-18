# LogInsight-FV evidence-relevance annotation guideline

**Guideline version:** 1.0  
**Outcome:** human evidence relevance  
**Unit:** one complete HDFS `BlockId` block or one complete BGL evaluation window  
**Primary label:** binary relevance of an event occurrence to the unit-level anomaly

This guideline defines the human ground truth used to evaluate whether an
explanation identifies evidence for the detector's block- or window-level
anomaly. It is an evidence-selection task, not a root-cause investigation,
detector explanation, or claim-support task. Annotators must use the supplied
log evidence and the rules below rather than Integrated Gradients, occlusion
values, detector scores, LLM output, or assumptions about the system.

## What annotators receive

For each unit, the annotation view contains:

- an anonymized unit identifier;
- chronological event occurrences;
- a stable zero-based `event_position` for every occurrence;
- source `line_id` and `event_id` as display and audit identifiers;
- the event template, parsed parameters, and raw line when available; and
- an optional anonymized normal comparison view, clearly separated from the
  target unit.

Annotators do not receive detector probabilities, thresholds, predicted
classes, Integrated Gradients values or ranks, occlusion or leave-one-out
scores, LLM explanations, condition names, model or seed identifiers, or
experiment filenames. The later annotation-interface implementation must
enforce this boundary. Event position is the identity of an occurrence;
`event_id` alone is not sufficient because the same template can occur many
times in one unit.

## The annotation question

Read the complete chronological unit and answer:

> Which event occurrences are direct, observable evidence that this unit has
> the stated anomalous behavior, or that distinguish it from the supplied
> normal comparison?

Select an occurrence only when removing or changing it would make the evidence
for the anomaly materially weaker. Mark the smallest supported set. Do not
select every event that is temporally nearby or part of the same routine
workflow.

For a normal unit, select an event only when it is evidence of a concrete
deviation or failure in that unit; routine successful activity receives no
relevance label. An empty relevant set is valid. Annotators must not force a
positive event when the supplied evidence does not support one.

## Event-level labels

Every event occurrence receives exactly one provisional label:

| Label | Use when |
|---|---|
| `relevant` | The occurrence is direct evidence of the anomaly or a meaningful event-level contrast. |
| `not_relevant` | The occurrence is routine context, a consequence without diagnostic value, or unrelated to the anomaly. |
| `uncertain` | The evidence is insufficient or genuinely ambiguous under these rules. Do not guess. |

`uncertain` is a temporary quality-control label. It is resolved during
adjudication before the primary precision, recall, or F1 analysis. It is never
silently counted as either relevant or not relevant. If no adjudication stage
exists yet, the record remains incomplete and is excluded from the primary
metric with an explicit exclusion reason.

Annotators may add one descriptive role to a relevant or uncertain event:

- `failure_signal`: an error, exception, timeout, invalidation, mismatch,
  corruption, failed operation, or other explicit abnormal signal;
- `anomalous_transition`: a state change abnormal in this unit's chronology or
  relative to the normal comparison;
- `meaningful_contrast`: a difference from the normal comparison specific
  enough to support the anomaly interpretation;
- `recovery_or_consequence`: a recovery, invalidation, cleanup, or consequence
  that confirms the abnormal episode; or
- `context_only`: background needed to read the sequence but not evidence of
  the anomaly. This should normally receive `not_relevant`.

The role is descriptive and does not override the binary relevance label.

## Decision rules

1. **Use observable log evidence.** A relevant label must be supported by the
   event template, parameter value, raw line, chronology, or a meaningful
   contrast. Do not infer an unseen host failure, user intent, or physical
   root cause.

2. **Label occurrences, not templates.** If the same `event_id` or template
   appears several times, judge each `event_position` independently. Mark only
   the occurrences supported by the evidence.

3. **Prefer direct signals.** Explicit errors, exceptions, timeouts, failed
   operations, invalid-set additions, checksum or size mismatches, unexpected
   state transitions, and deletion following a failed or invalid block are
   relevant when they explain the unit's anomaly.

4. **Do not label routine context by proximity.** Allocation, receipt,
   successful storage, normal responder termination, timestamps, process IDs,
   block IDs, hostnames, ports, and ordinary sequence boundaries are not
   relevant merely because they occur near a failure.

5. **Parameters can make an event relevant.** Mark the whole occurrence when a
   parameter value is itself the supported abnormal signal, such as a
   documented mismatch, impossible value, or abnormal size. Do not mark an
   event solely because a hostname, address, block identifier, or numeric value
   differs from another unit without evidence that the difference is meaningful.

6. **Use chronology only when it changes interpretation.** A repeated event,
   missing expected transition, or out-of-order transition is relevant when
   the supplied sequence makes that deviation observable. Do not call an event
   anomalous simply because it appears early or late.

7. **Handle recovery and cleanup carefully.** A recovery, invalidation, or
   deletion event is relevant when it confirms the abnormal episode or its
   consequence. Routine successful completion or ordinary cleanup is not
   relevant without an anomaly-specific signal.

8. **Use contrastive differences conservatively.** A difference from the
   normal comparison is meaningful only when it is event-level, observable,
   and plausibly tied to the anomaly. Differences caused only by identifiers,
   timestamps, hosts, paths, or ordinary payload variation are not meaningful
   by themselves.

9. **Do not use detector or attribution information.** Detector confidence,
   predicted class, Integrated Gradients ranking, occlusion, leave-one-out,
   or a generated explanation cannot justify a relevance label.

10. **Abstain when the record is insufficient.** Use `uncertain` when the
    available lines do not establish whether an event is diagnostic, when the
    event is truncated or malformed, or when two rules conflict. Record a
    short reason for adjudication.

## Contrastive assessment

When a normal comparison is supplied, complete one unit-level assessment in
addition to event labels:

- `meaningful_difference`: `yes`, `no`, or `uncertain`;
- `difference_positions`: target event positions involved, if any; and
- `reason`: one short evidence-based sentence.

Mark `yes` only when the difference is more than an identifier, timestamp,
host, path, or ordinary payload variation and helps explain the anomaly. A
`no` assessment is valid even when the units have many textual differences.
Use `uncertain` when the comparison is incomplete or the difference cannot be
interpreted from the supplied evidence.

The contrastive assessment is secondary. The primary evidence set remains the
event-level `relevant` occurrences in the target unit.

## Recommended annotation record

The later export implementation should preserve these fields without adding
condition or attribution metadata to the annotator view:

```json
{
  "guideline_version": "1.0",
  "unit_id": "anonymized-unit-001",
  "dataset": "HDFS",
  "annotator_id_or_hash": "annotator-a",
  "events": [
    {
      "event_position": 7,
      "line_id": 9512780,
      "event_id": "e3df2680",
      "relevance": "relevant",
      "role": "failure_signal",
      "reason": "The event records the abnormal block outcome."
    }
  ],
  "contrastive_assessment": {
    "meaningful_difference": "uncertain",
    "difference_positions": [],
    "reason": "The comparison does not establish whether the size difference is diagnostic."
  },
  "unit_notes": "Optional short note for adjudication only"
}
```

The export must include all event occurrences, including `not_relevant` ones,
so omissions are distinguishable from an unlabelled record. It must preserve
the original event position and line identifiers for audit, while the
annotator-facing display may replace identifiers with study-local aliases.
Annotator identity must be pseudonymized in analysis artifacts.

## Quality and adjudication rules

Each core unit is independently labelled by at least two annotators. They must
not discuss individual units before submitting independent labels. A reviewer
checks that every occurrence has a provisional label, reasons are evidence
based, and the unit-level contrastive assessment is complete when applicable.

Disagreements are retained at event-position level. The adjudicator reviews
the complete evidence and this guideline, records the final binary label and a
short disagreement category, and does not use detector scores or attribution
values to break ties. The final label is the only label used for the primary
human-evidence-recovery metric; original labels, adjudication decisions, and
exclusion reasons remain available for audit.

## Worked examples

| Evidence pattern | Label | Reason |
|---|---|---|
| An explicit exception, timeout, checksum mismatch, or invalidation that explains the unit anomaly | `relevant` | Direct failure signal. |
| A routine block allocation or receipt event with no abnormal parameter or transition | `not_relevant` | Normal context alone does not establish an anomaly. |
| A later deletion that follows an explicitly invalid block state | `relevant` | It confirms the abnormal episode or consequence. |
| A normal responder termination with no failure signal | `not_relevant` | Routine completion is not diagnostic by itself. |
| A parameter differs from the comparison only by block ID, host, path, or timestamp | `not_relevant` | Identifier variation is not a meaningful contrast. |
| A truncated line or unexplained sequence difference | `uncertain` | The supplied evidence is insufficient; adjudicate later. |

This guideline is the evidence-relevance protocol for Step 51. Claim-support
labels, annotator blinding, interface implementation, consent, compensation,
and institutional review are specified in later steps and must not be added
to this outcome by implication.
