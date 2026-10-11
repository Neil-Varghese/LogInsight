# Research event vocabularies

Vocabulary manifests used by the research detectors belong here. They must be
fit on the training split only and must include:

- padding token `__PAD__` at ID `0`;
- all training event IDs with deterministic IDs;
- `__UNKNOWN_EVENT__` as a learned final vocabulary row;
- a SHA-256 checksum and fit-scope metadata.

The existing `research/event2idx.pkl` belongs to the engineering baseline and
is intentionally not modified. Generate a research vocabulary after the
leakage-free training split exists:

    python3 research/scripts/build_event_vocabulary.py \\
      --training-sequences <TRAINING_JSONL> \\
      --output research/models/vocabularies/event_vocabulary.json
