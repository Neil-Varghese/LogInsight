"""Deterministic, train-only event vocabulary with an explicit unknown token."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PAD_TOKEN = "__PAD__"
UNKNOWN_EVENT_TOKEN = "__UNKNOWN_EVENT__"
VOCABULARY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class EventVocabulary:
    """Mapping used by both detector training and later perturbation tests."""

    token_to_id: dict[str, int]
    pad_token: str = PAD_TOKEN
    unknown_token: str = UNKNOWN_EVENT_TOKEN

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unknown_id(self) -> int:
        return self.token_to_id[self.unknown_token]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, sequence: Sequence[str]) -> list[int]:
        """Map known events and route every unseen event to the learned UNK row."""

        return [self.token_to_id.get(str(event), self.unknown_id) for event in sequence]

    def decode(self, sequence: Sequence[int]) -> list[str]:
        id_to_token = {index: token for token, index in self.token_to_id.items()}
        return [id_to_token.get(int(index), self.unknown_token) for index in sequence]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": VOCABULARY_SCHEMA_VERSION,
            "pad_token": self.pad_token,
            "pad_id": self.pad_id,
            "unknown_token": self.unknown_token,
            "unknown_id": self.unknown_id,
            "size": self.size,
            "token_to_id": dict(sorted(self.token_to_id.items(), key=lambda item: item[1])),
        }


def build_event_vocabulary(
    training_sequences: Iterable[Sequence[str]],
    pad_token: str = PAD_TOKEN,
    unknown_token: str = UNKNOWN_EVENT_TOKEN,
) -> EventVocabulary:
    """Fit a deterministic vocabulary from training sequences only.

    The explicit PAD row is ID 0.  Training events are sorted lexicographically
    so the mapping is reproducible independent of input iteration order.  The
    unknown-event row is always included, even if training has no OOV examples,
    which makes it available to test-time replacement and unseen events.
    """

    if pad_token == unknown_token:
        raise ValueError("pad_token and unknown_token must be different")
    events: set[str] = set()
    for sequence in training_sequences:
        events.update(str(event) for event in sequence)
    events.discard(pad_token)
    events.discard(unknown_token)
    token_to_id = {pad_token: 0}
    token_to_id.update({event: index for index, event in enumerate(sorted(events), start=1)})
    token_to_id[unknown_token] = len(token_to_id)
    return EventVocabulary(token_to_id, pad_token, unknown_token)


def vocabulary_checksum(vocabulary: EventVocabulary) -> str:
    """Hash the canonical token/id mapping for experiment-ledger recording."""

    payload = json.dumps(vocabulary.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_vocabulary(
    vocabulary: EventVocabulary,
    output_path: Path,
    fit_scope: str = "training_only",
    source_description: str = "training event sequences",
) -> dict[str, object]:
    """Write a self-describing vocabulary manifest."""

    if fit_scope != "training_only":
        raise ValueError("Research vocabulary must be fitted on training_only data")
    payload = vocabulary.to_dict()
    payload.update(
        {
            "fit_scope": fit_scope,
            "source_description": source_description,
            "vocabulary_sha256": vocabulary_checksum(vocabulary),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_vocabulary(path: Path) -> EventVocabulary:
    """Load and validate a saved vocabulary manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    token_to_id = {str(token): int(index) for token, index in payload["token_to_id"].items()}
    vocabulary = EventVocabulary(
        token_to_id,
        str(payload.get("pad_token", PAD_TOKEN)),
        str(payload.get("unknown_token", UNKNOWN_EVENT_TOKEN)),
    )
    if vocabulary.pad_id != int(payload["pad_id"]):
        raise ValueError("Saved vocabulary pad ID does not match its mapping")
    if vocabulary.unknown_id != int(payload["unknown_id"]):
        raise ValueError("Saved vocabulary unknown ID does not match its mapping")
    if vocabulary_checksum(vocabulary) != payload.get("vocabulary_sha256"):
        raise ValueError("Saved vocabulary checksum does not match its mapping")
    return vocabulary
