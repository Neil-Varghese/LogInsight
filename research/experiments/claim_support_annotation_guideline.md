# LogInsight-FV claim-support annotation guideline

**Guideline version:** 1.0  
**Outcome:** support of generated explanation claims  
**Unit:** one atomic claim from one structured explanation  
**Primary labels:** `supported`, `unsupported`, or `uncertain`

This guideline defines the secondary human outcome for whether an explanation's
claims are supported by the evidence supplied with that explanation. It is a
claim-entailment task. It does not ask whether the cited event is the true
cause of the incident, whether the event is independently relevant under the
Step 51 guideline, or whether the detector used that event internally.

## What annotators receive

For each explanation, the support packet contains:

- an anonymized explanation and its atomic claims;
- the target block/window evidence in chronological order;
- the cited event positions and line identifiers, with the corresponding raw
  lines/templates and parameters;
- the anonymized normal comparison when the claim is contrastive; and
- detector output or a recorded verification result only when a claim makes a
  detector-specific assertion.

Annotators do not receive condition names, attribution values or ranks,
occlusion values, leave-one-out values, model or seed identifiers, experiment
filenames, or the study hypothesis. They must not use Step 51 relevance labels
to decide support. A routine event can support a factual claim that it
occurred; that does not make the event relevant evidence for the anomaly.

## The annotation question

Read one claim at a time, inspect every citation it makes, and answer:

> Does the supplied evidence entail the claim as written, at the level of
> certainty and specificity used by the claim?

Judge the literal claim, including qualifiers such as “may,” “likely,”
“caused,” “all,” and “never.” Do not repair an overclaim by silently rewriting
it. If the claim contains multiple factual parts, all parts must be supported
for the claim to receive `supported`; otherwise split it into atomic claims
before annotation or record the unsupported component explicitly.

## Claim-support labels

| Label | Use when |
|---|---|
| `supported` | The cited and supplied evidence directly supports the claim at its stated certainty and scope. |
| `unsupported` | The evidence contradicts the claim, fails to establish it, contains an invented citation/event, or the claim makes an unjustified causal or detector assertion. |
| `uncertain` | The evidence is incomplete, ambiguous, contradictory, or too weak to decide support without guessing. |

`uncertain` is not a softer form of `supported`. It records a genuine evidence
limitation and is retained for adjudication. A claim is not uncertain merely
because the annotator would prefer more context when the supplied lines
clearly entail it.

## Support basis and issue categories

Annotators should record one primary support basis for audit:

- `direct_log_observation`: the cited line or event explicitly states the
  claimed occurrence, value, or state;
- `contrastive_observation`: the target and normal comparison establish the
  claimed event-level difference;
- `recorded_detector_output`: the supplied detector or verification record
  directly supports a detector-behavior claim;
- `qualified_interpretation`: the claim is explicitly framed as a possibility
  or uncertainty and the supplied evidence supports that limited framing; or
- `none` when the claim is unsupported or uncertain.

If a claim is not supported, record the most specific issue category:

- `missing_or_wrong_citation`;
- `invented_event_or_line`;
- `contradicted_by_evidence`;
- `causal_overclaim`;
- `detector_overclaim`;
- `contrast_not_established`;
- `scope_or_certainty_overclaim`; or
- `insufficient_or_ambiguous_context`.

The issue category describes the support decision and must not be used to
infer an event-relevance label.

## Decision rules

1. **Check citations first.** Every cited event position and line identifier
   must exist in the supplied packet and refer to the text used by the claim.
   Missing, mismatched, or invented evidence is unsupported unless the claim
   is explicitly an abstention statement.

2. **Require entailment, not topical similarity.** A line mentioning a block
   or component does not support every claim about that component. The text
   must establish the occurrence, value, relationship, or state asserted.

3. **Respect qualifiers and scope.** “A responder terminated” is supported by
   one matching line. “All responders terminated normally” requires evidence
   for all responders and for normality. “The block was deleted” is distinct
   from “the block was corrupt.”

4. **Do not infer root cause.** A raw sequence may support that an invalidation
   and deletion occurred. It does not by itself support “disk corruption caused
   the failure,” “operator X caused the incident,” or another physical cause
   without independent operational evidence.

5. **Qualified interpretations remain limited.** “The sequence may indicate a
   failed replication” can be supported or uncertain when the cited lines show
   the relevant failed transition. A definitive claim that the sequence proves
   that cause is unsupported unless the supplied evidence proves it.

6. **Handle contrastive claims explicitly.** A claim that the target differs
   from the normal comparison requires evidence from both sides or a supplied
   comparison record. Differences in block IDs, timestamps, hosts, paths, and
   ordinary payload values do not support an anomaly claim by themselves.

7. **Detector claims require detector evidence.** A claim about a probability,
   threshold, predicted class, score change, or detector rationale is supported
   only when the corresponding recorded detector output or verification result
   is in the support packet. Raw log text cannot establish an internal model
   decision.

8. **Treat event occurrence and event importance separately.** A routine
   receipt line can support “the block was received” even when Step 51 labels
   that event `not_relevant`. Conversely, an event labelled relevant under Step
   51 does not automatically support every claim that cites it.

9. **Abstention is not a factual claim.** An explicit abstention or uncertainty
   statement is recorded for abstention coverage and is not counted as a
   supported or unsupported factual claim. A claim that asserts a fact while
   merely mentioning uncertainty is still judged on that fact.

10. **Use `uncertain` rather than guessing.** Choose `uncertain` when lines
    are truncated, the comparison is incomplete, citations conflict, or the
    claim's interpretation cannot be resolved from the supplied packet.

## Claim-type guidance

The structured schema registers four claim types:

- **`anomalous_pattern`:** Supported when the cited log evidence directly
  establishes the described abnormal pattern. A routine sequence described as
  anomalous without an observable deviation is unsupported.
- **`contrastive_difference`:** Supported when the target and normal evidence
  establish a meaningful event-level difference. Identifier-only differences
  are insufficient.
- **`detector_behavior`:** Supported only by the supplied detector output or
  verification record, including the stated score and direction. Do not infer
  detector behavior from an LLM attribution or from raw logs.
- **`uncertainty`:** Supported when the claim accurately and specifically
  states what the evidence does not establish. Vague hedging does not rescue a
  separate unsupported factual assertion.

## Recommended annotation record

The later interface/export implementation should preserve one record per
atomic claim:

```json
{
  "guideline_version": "1.0",
  "explanation_id": "anonymized-explanation-001",
  "claim_id": "claim-2",
  "claim_type": "contrastive_difference",
  "claim_text": "The target block is invalidated after storage.",
  "cited_event_positions": [13, 14, 15],
  "cited_line_ids": [10986891, 10986892, 10986893],
  "support_label": "supported",
  "support_basis": "direct_log_observation",
  "issue_category": null,
  "rationale": "The cited lines record the block being added to the invalid set."
}
```

The export must preserve the original claim text and citations, including
claims marked `uncertain`. Annotator identity is pseudonymized. The support
record must retain the explanation condition only in a restricted provenance
ledger, not in the annotator-facing packet.

## Quality, denominator, and adjudication rules

Claim-support annotation is collected only on the pre-specified explanation
sample. At least two annotators independently label each included claim while
blind to condition and attribution access. They must not use the Step 51
relevance labels as a shortcut.

Disagreements are retained at claim level. An adjudicator reviews the claim,
citations, and supplied evidence using this guideline, records the final label
and issue category, and preserves both original decisions. Abstentions are
reported separately and are not forced into the support denominator.

For the annotated factual-claim set, report:

- `total_claims`: supported + unsupported + uncertain claims;
- `supported_claims`, `unsupported_claims`, and `uncertain_claims`;
- `support_rate = supported_claims / total_claims`; and
- `hallucination_rate = unsupported_claims / total_claims`.

These are claim-support outcomes. They must not be merged with event relevance,
human evidence-recovery F1, detector faithfulness, or perturbation consistency.

## Worked examples

| Claim and evidence | Label | Reason |
|---|---|---|
| “The block was added to the invalid set,” with matching invalid-set lines cited | `supported` | Direct log observation. |
| “The block was deleted,” citing only a receipt line | `unsupported` | The citation does not entail deletion; `missing_or_wrong_citation`. |
| “Disk corruption caused the incident,” with invalidation lines but no corruption evidence | `unsupported` | Causal overclaim. |
| “The target differs from the normal block because the block ID changed,” with only identifier differences | `unsupported` | Identifier variation is not a meaningful contrast. |
| “The sequence may indicate a failed replication,” with incomplete transition evidence | `uncertain` | The qualified interpretation cannot be resolved from the supplied packet. |
| “The detector probability fell after deletion,” with the recorded before/after scores supplied | `supported` | Recorded detector output supports the claim. |

This guideline is the claim-support protocol for Step 52. Evidence relevance,
annotation workflow, interface blinding, and ethics requirements are specified
in separate implementation steps and must not be inferred from this document.
