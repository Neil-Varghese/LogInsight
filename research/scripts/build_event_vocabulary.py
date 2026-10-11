"""Fit the research event vocabulary from a training-only JSONL sequence file.

The input must already be a training split.  The script deliberately refuses
to infer a split or fit from the full dataset, because that would leak test
events into the detector vocabulary.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

from event_vocabulary import UNKNOWN_EVENT_TOKEN, build_event_vocabulary, save_vocabulary


def iter_sequences(path: Path) -> Iterable[list[str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            sequence = record.get("Sequence")
            if not isinstance(sequence, list):
                raise ValueError(f"Record {line_number} has no list-valued Sequence field")
            yield [str(event) for event in sequence]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-sequences", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/models/vocabularies/event_vocabulary.json"),
    )
    parser.add_argument("--source-description", default="user-supplied training-only JSONL sequences")
    args = parser.parse_args()
    vocabulary = build_event_vocabulary(iter_sequences(args.training_sequences))
    payload = save_vocabulary(
        vocabulary,
        args.output,
        fit_scope="training_only",
        source_description=args.source_description,
    )
    print(json.dumps({
        "output": str(args.output),
        "size": payload["size"],
        "known_event_count": int(payload["size"]) - 2,
        "pad_token": payload["pad_token"],
        "pad_id": payload["pad_id"],
        "unknown_token": UNKNOWN_EVENT_TOKEN,
        "unknown_id": payload["unknown_id"],
        "fit_scope": payload["fit_scope"],
        "vocabulary_sha256": payload["vocabulary_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
