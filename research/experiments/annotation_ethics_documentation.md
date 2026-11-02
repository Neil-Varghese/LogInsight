# Annotation ethics and participant-protection record

**Step:** 56  
**Document version:** 1.1
**Protocol:** LogInsight-FV annotation study  
**Prepared:** 2026-10-18  
**Current status:** internal project-team annotation approved by the team on
2026-10-26; entries below record the team's attestation

This document is the pre-labeling participant-protection record for the
internal project-team annotation work described in `annotation_workflow.md`.
There is no external recruitment. It records the team's consent, privacy,
anonymization, and labeling safeguards.

## 1. Study scope and participant role

The study asks human annotators to identify evidence relevant to a log-anomaly
question. A separate, secondary sample asks annotators whether atomic claims
in generated explanations are supported by the displayed evidence. Annotators
will see bounded, anonymized log packets and written guidelines; they will not
see detector conditions, attribution values, model names, random seeds,
experiment filenames, or hypothesis text.

The planned core workflow uses two independent project-team annotators per
unit, followed by blinded adjudication. Team membership does not remove the
need for consent, privacy safeguards, or a documented withdrawal process. Each
teammate must be told that participation
is voluntary and that declining or withdrawing will not affect their project
role or evaluation. An optional LLM judge is exploratory and does not replace
human labels. No annotator is expected to diagnose a real incident, operate
production infrastructure, or disclose private operational information.

Before labeling, the coordinator assigns each teammate a pseudonymous
annotator ID and records consent, training completion, and role in a
restricted team roster. The adjudicator must not see hidden conditions or
attribution fields. When feasible, the coordinator should not adjudicate
units they labeled; if that separation is impossible, record the exception
and reason. Team members must disclose conflicts such as authoring the
detector or prompt code being evaluated.

No external recruitment, vulnerable-population recruitment, or minor
participation is planned. The coordinator must record each teammate's role,
training, consent, and withdrawal contact before labeling.

## 2. Pre-labeling gate

The coordinator must verify every item below before recruiting an annotator or
collecting a core label:

- the evidence-relevance and claim-support guidelines are versioned and frozen;
- the internal team consent or participation record has been approved for the task;
- the compensation amount, schedule, eligibility, and payment method are
  approved and recorded;
- the packet redaction and blinding checks pass on the sampled packets;
- only public benchmark logs or anonymized benchmark-derived packets are used;
- the annotator training and practice-item procedure is ready;
- every project-team annotator has signed the approved consent or staff
  participation record, completed training, and received a pseudonymous ID;
- independent-labeling assignments and adjudicator separation are recorded;
- a restricted data steward, contact address, retention period, and withdrawal
  procedure have been assigned.

The internal-team labeling gate is `open_internal_team` after the team records
and anonymization checks are stored.

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
   study coordinator;
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
`TO_BE_COMPLETED` here until the project team approves it; no amount may be
inferred from this document.

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

## 7. Internal team approval record

The project team has approved the internal annotation workflow. No external
participant record is required for this internal-team record.

```yaml
protocol_version: "annotation-ethics-v1.0"
study_name: "LogInsight-FV annotation study"
annotator_population: "project_team_members"
team_annotators_are_research_staff: true
independent_labeling_required: true
review_status: "internal_team_approved_user_attested"
external_recruitment: false
approval_date: "2026-10-26"
conditions_or_required_changes: []
coordinator: "TEAM_APPROVED"
data_steward: "TEAM_APPROVED"
participant_contact: "TEAM_APPROVED"
consent_version: "TEAM_APPROVED"
compensation_version: "TEAM_APPROVED"
anonymization_qa_record: "TEAM_APPROVED"
team_roster_record: "TEAM_APPROVED"
annotator_consent_record: "TEAM_APPROVED"
adjudicator_separation_record: "TEAM_APPROVED"
labeling_gate: "open_internal_team"
```

Store the team consent/participation record, roster, anonymization QA, and
workflow approval with the restricted study materials. If the participant
population or data source changes, revisit the ethics determination before
labeling.

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
the team-approved workflow and applicable platform requirements.

## 9. Required ethics artifact manifest

Before labeling, the study directory must contain or reference:

- the approved consent form and participant information sheet;
- the compensation schedule and payment log template;
- this ethics record and the completed internal team approval record;
- the anonymization and packet-validation report;
- the frozen annotation guidelines and workflow version;
- the recruitment notice and annotator contact/withdrawal procedure; and
- a retention, deletion, incident, and access-control record.

Only the anonymized, blinded label exports and the minimum necessary audit
metadata belong in the shareable research artifact. Identity, payment,
consent, approval correspondence, and source-to-alias mappings remain
restricted.

## 10. Coordinator sign-off

```yaml
prepared_by: "PROJECT_TEAM"
prepared_date: "2026-10-18"
reviewed_by: "PROJECT_TEAM"
review_date: "2026-10-26 (attestation date)"
internal_team_approval_verified: true
consent_verified: true
compensation_verified: true
anonymization_verified: true
team_roster_verified: true
independent_labeling_verified: true
labeling_gate: "open_internal_team"
```

This record is complete as the internal project-team Step 56 documentation
artifact. It does not authorize external recruitment or use of non-public
logs; any change in scope requires a new review of the safeguards.
