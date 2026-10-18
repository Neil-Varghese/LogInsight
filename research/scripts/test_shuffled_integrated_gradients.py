import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    from shuffled_integrated_gradients import (
        build_shuffled_prompt_payload,
        build_shuffled_record,
        run_shuffled_integrated_gradients,
    )
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.shuffled_integrated_gradients import (
        build_shuffled_prompt_payload,
        build_shuffled_record,
        run_shuffled_integrated_gradients,
    )


def _record(unit_id="blk_1", values=(0.1, 0.8, -0.4)):
    events = []
    for position, value in enumerate(values):
        events.append(
            {
                "input_position": position,
                "original_sequence_index": position,
                "event_id": position + 4,
                "event_token": f"E{position + 4}",
                "is_padding": False,
                "attribution": value,
                "absolute_attribution": abs(value),
                "rank": position + 1,
            }
        )
    events.append(
        {
            "input_position": len(values),
            "original_sequence_index": None,
            "event_id": 0,
            "event_token": "__PAD__",
            "is_padding": True,
            "attribution": 0.0,
            "absolute_attribution": 0.0,
            "rank": None,
        }
    )
    return {
        "method": "integrated_gradients",
        "architecture": "loginsight_lstm",
        "seed": 13,
        "split": "test",
        "unit_id": unit_id,
        "label_id": 1,
        "probability": 0.9,
        "events": events,
    }


def _write(path: Path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class ShuffledIntegratedGradientsTests(unittest.TestCase):
    def test_within_block_permutation_preserves_positions_and_is_reproducible(self):
        first = build_shuffled_record(_record(), shuffle_type="within_block_permutation", shuffle_seed=2026)
        second = build_shuffled_record(_record(), shuffle_type="within_block_permutation", shuffle_seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(
            [event["input_position"] for event in first["events"]],
            [0, 1, 2, 3],
        )
        self.assertEqual(first["shuffle_type"], "within_block_permutation")
        self.assertFalse(first["verification_only"])

    def test_cross_block_donor_requires_matching_active_event_count(self):
        result = build_shuffled_record(
            _record("target"),
            shuffle_type="cross_block_donor",
            donor_record=_record("donor", values=(0.7, 0.2, -0.1)),
        )
        self.assertEqual(result["donor_unit_id"], "donor")
        self.assertEqual([event["attribution"] for event in result["events"][:3]], [0.7, 0.2, -0.1])
        with self.assertRaises(ValueError):
            build_shuffled_record(
                _record("target"),
                shuffle_type="cross_block_donor",
                donor_record=_record("donor", values=(0.7, 0.2)),
            )

    def test_prompt_payload_excludes_provenance_and_padding(self):
        record = build_shuffled_record(_record(), shuffle_type="within_block_permutation")
        payload = build_shuffled_prompt_payload(record)
        self.assertEqual(set(payload), {"events", "shuffle_type"})
        self.assertEqual(len(payload["events"]), 3)
        self.assertNotIn("source_unit_id", payload)
        self.assertNotIn("verification_only", payload)

    def test_rejects_non_integrated_gradients_source(self):
        record = _record()
        record["method"] = "event_occlusion"
        with self.assertRaises(ValueError):
            build_shuffled_record(record, shuffle_type="within_block_permutation")

    def test_run_writes_shuffled_artifact_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "seed_13_test_integrated_gradients.jsonl.gz"
            output = root / "shuffled.jsonl.gz"
            _write(source, [_record()])
            manifest = run_shuffled_integrated_gradients(
                source,
                shuffle_type="within_block_permutation",
                output=output,
                show_progress=False,
            )
            self.assertEqual(manifest["count"], 1)
            self.assertTrue(manifest["prompt_visible"])
            self.assertTrue((root / "shuffled.manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
