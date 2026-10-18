# Annotation ethics and participant-protection record

**Step:** 56  
**Document version:** 1.0  
**Protocol:** LogInsight-FV annotation study  
**Prepared:** 2026-10-18  
**Current status:** documentation prepared; institutional approval or exemption
is **pending**

This document is the pre-recruitment ethics record for the human annotation
work described in `annotation_workflow.md`. It records the safeguards that
must be in place before recruitment or core labeling starts. Preparing this
file does not constitute institutional approval, an exemption, or informed
consent. The study coordinator must complete the decision record below and
retain the supporting approval or exemption artifact before opening the
labeling gate.

## 1. Study scope and participant role

The study asks human annotators to identify evidence relevant to a log-anomaly
question. A separate, secondary sample asks annotators whether atomic claims
in generated explanations are supported by the displayed evidence. Annotators
will see bounded, anonymized log packets and written guidelines; they will not
see detector conditions, attribution values, model names, random seeds,
experiment filenames, or hypothesis text.

The planned core workflow uses two independent annotators per unit, followed
by blinded adjudication. An optional LLM judge is exploratory and does not
replace human labels. No participant is expected to diagnose a real incident,
operate production infrastructure, or disclose private operational
information.

Recruitment eligibility, language requirements, prior log-reading experience,
and any exclusion criteria must be written into the recruitment notice and
approved with the protocol. The planned population is competent adult
annotators; vulnerable populations and minors are not intentionally recruited.
If the institution requires a different population or additional safeguards,
the coordinator must update this record before recruitment.

## 2. Pre-labeling gate

The coordinator must verify every item below before recruiting an annotator or
collecting a core label:

- the evidence-relevance and claim-support guidelines are versioned and frozen;
- the consent text has been approved for the actual platform and task;
- the compensation amount, schedule, eligibility, and payment method are
  approved and recorded;
- the packet redaction and blinding checks pass on the sampled packets;
- only public benchmark logs or logs anonymized under the approved protocol
  are used;
- institutional ethics/IRB approval or a documented exemption has been
  received for this exact protocol version;
- the annotator training and practice-item procedure is ready; and
- a restricted data steward, contact address, retention period, and withdrawal
  procedure have been assigned.

Until all items are recorded in the decision record, the status is
`labeling_gate: blocked`.

## 3. Consent information and process

The participant-facing consent form must use plain language and include:

1. the study purpose and the fact that the task evaluates log-evidence and
   explanation claims;
2. what the participant will do, the approximate number of packets, expected
   duration, practice period, and any break policy;
3. the types of text that may appear (anonymized public benchmark log lines),
   including the possibility of technical or repetitive content;
4. foreseeable risks, safeguards, confidentiality limits, and the procedure
   for reporting an accidentally identifying or distressing line;
5. voluntary participation, the right to skip an item where the protocol
   permits, and the right to stop or withdraw without penalty;
6. compensation eligibility and timing, including what happens if the
   participant withdraws after completing part of the work;
7. who can answer questions, who receives complaints, and how to contact the
   study coordinator or institutional ethics office;
8. what records are collected (pseudonymous annotator ID, labels, rationales,
   timestamps, training status, and payment record where required);
9. how records will be stored, who may access the re-identification mapping,
   how long records will be retained, and what may be released in a paper or
   artifact; and
10. the consent statement, consent date, protocol version, and the participant
    copy or download mechanism.

Consent must be obtained before training labels are retained. The coordinator
must record a pseudonymous consent receipt, never place a participant's name
in the annotation export, and retain the signed or platform-generated consent
record in the restricted study location.

## 4. Compensation record

Compensation is a protocol field, not an after-the-fact discretionary reward.
The coordinator must record the approved currency or unit, rate or fixed
amount, expected task time, payment milestones, platform fees, eligibility for
partial completion, and payment date. The amount is intentionally left
`TO_BE_COMPLETED` here until the institution or recruitment platform approves
it; no amount may be inferred from this document.

Payment must not depend on agreeing with the study hypothesis or achieving a
particular label agreement. Withdrawal must not forfeit compensation already
earned under the approved schedule. Payment identifiers are stored separately
from annotation labels and are never released with the analysis artifact.

## 5. Data minimization and anonymization

The study uses public benchmark logs where possible. If a source is not
public, it must be anonymized and approved before it is shown to a participant.
Public availability alone does not remove the need for an ethics review or
privacy assessment.

Packets are built with `research/scripts/annotation_packet.py` and must pass
`validate_annotation_export` and the Step 55 blinding validator. The packet
layer replaces source unit and event identifiers with study-local aliases and
redacts or suppresses hostnames, IP addresses, filesystem paths, UUIDs,
emails, block identifiers, detector metadata, condition labels, attribution
values, model/seed names, and experiment filenames. Context is capped by the
registered token policy; omitted positions are recorded rather than silently
reconstructed.

The source-to-study alias mapping and any raw source material are restricted
to the data steward and are not joined to participant labels before the
blinded export is locked. The released artifact contains only the minimum
anonymized packets or derived summaries required for reproducibility. No
private operational logs, credentials, secrets, customer data, or directly
identifying information may be sent to annotators or an external LLM judge.

Storage controls must include access-limited directories, encrypted transfer
and storage where supported, versioned immutable exports, separate payment and
identity records, and a documented retention/deletion date. Any suspected
identifier leakage is quarantined, logged, removed, and re-annotated only
after the corrected packet passes the blinding check.

## 6. Risks and mitigations

The foreseeable risks are exposure to technical log text, accidental residual
identifiers, fatigue or repetitive work, and a participant's concern about
their labels being evaluated. Mitigations are bounded packets, public or
approved anonymized data, automated redaction plus human QA, practice items,
breaks, the ability to stop, pseudonymous IDs, blinded assignments, and
reporting of agreement as a study statistic rather than a participant
performance ranking.

Participants must be told how to report a problematic packet. The coordinator
will pause that unit, preserve an audit note without the participant's name in
the analysis export, notify the designated ethics/data contact, and determine
whether re-consent, deletion, or protocol reporting is required.

## 7. Institutional decision record

Complete and attach the institutional response before recruitment. Do not
replace `TO_BE_COMPLETED` or `null` with an inferred value.

```yaml
protocol_version: "annotation-ethics-v1.0"
study_name: "LogInsight-FV annotation study"
review_status: "pending"
review_type: "IRB approval | ethics approval | documented exemption"
institution: "TO_BE_COMPLETED"
review_board_or_office: "TO_BE_COMPLETED"
decision_id: null
decision_date: null
approved_protocol_version: null
conditions_or_required_changes: []
coordinator: "TO_BE_COMPLETED"
data_steward: "TO_BE_COMPLETED"
participant_contact: "TO_BE_COMPLETED"
ethics_contact: "TO_BE_COMPLETED"
consent_version: "TO_BE_COMPLETED"
compensation_version: "TO_BE_COMPLETED"
anonymization_qa_record: "TO_BE_COMPLETED"
labeling_gate: "blocked"
```

The supporting decision letter, exemption memo, or approved protocol must be
stored in the restricted ethics folder and referenced by hash in the final
artifact manifest. If the review office requests changes, update this record,
the consent form, and the workflow version before recruitment. A decision that
does not cover external annotators, the displayed data, or the current task
must be treated as incomplete.

## 8. Withdrawal, correction, and incident handling

The coordinator assigns a pseudonymous participant ID and records consent,
training completion, submissions, corrections, and withdrawal requests in a
restricted audit log. On withdrawal, stop new assignments, identify the
participant's records through the restricted mapping, apply the approved
deletion or retention rule, preserve only what the ethics decision requires,
and document the action. Do not edit a locked label in place; retain the
original, corrected value, reason, actor, timestamp, and guideline version.

Any accidental disclosure, packet leakage, unauthorized access, or payment
error is recorded as an incident with date, affected artifact, containment,
notification decision, and resolution. The coordinator escalates according to
the institutional decision and applicable platform requirements.

## 9. Required ethics artifact manifest

Before labeling, the study directory must contain or reference:

- the approved consent form and participant information sheet;
- the compensation schedule and payment log template;
- this ethics record and the completed institutional decision record;
- the anonymization and packet-validation report;
- the frozen annotation guidelines and workflow version;
- the recruitment notice and annotator contact/withdrawal procedure; and
- a retention, deletion, incident, and access-control record.

Only the anonymized, blinded label exports and the minimum necessary audit
metadata belong in the shareable research artifact. Identity, payment,
consent, institutional correspondence, and source-to-alias mappings remain
restricted.

## 10. Coordinator sign-off

```yaml
prepared_by: "TO_BE_COMPLETED"
prepared_date: "2026-10-18"
reviewed_by: "TO_BE_COMPLETED"
review_date: null
approval_or_exemption_verified: false
consent_verified: false
compensation_verified: false
anonymization_verified: false
labeling_gate: "blocked"
```

This record is complete as the Step 56 documentation artifact. The booleans
above must be updated only from actual supporting records; Step 57 and Step 58
must not begin while `labeling_gate` is `blocked`.
