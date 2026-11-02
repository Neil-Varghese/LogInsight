"""Score completed Step 55 annotator responses against the hidden key."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from annotation_blinding import score_id_memorization_probe
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.annotation_blinding import score_id_memorization_probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True, help="JSON array of annotator responses")
    parser.add_argument("--answer-key", type=Path, default=Path("research/results/tables/step55_id_memorization_answer_key.json"))
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/step55_id_memorization_result.json"))
    args = parser.parse_args(argv)
    try:
        responses = json.loads(args.responses.read_text(encoding="utf-8"))
        key_record = json.loads(args.answer_key.read_text(encoding="utf-8"))
        result = score_id_memorization_probe(responses, key_record["answer_key"])
        result["answer_key_sha256"] = __import__("hashlib").sha256(args.answer_key.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as error:
        print(f"Step 55 probe scoring failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "output": str(args.output), "accuracy": result["accuracy"], "answered": result["answered"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
