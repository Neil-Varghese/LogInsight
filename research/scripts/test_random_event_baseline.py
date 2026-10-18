import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    from random_event_baseline import (
        build_random_event_record,
        run_random_event,
        select_random_events,
    )
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.random_event_baseline import (
        build_random_event_record,
        run_random_event,
        select_random_events,
    )


def _record():
    return {
        "method": "integrated_gradients",
        "architecture": "loginsight_lstm",
        "seed": 13,
        "split": "test",
        "unit_id": "blk_1",
        "label_id": 1,
        "probability": 0.9,
        "events": [
            {"event_id": 4, "is_padding": False, "rank": 3, "input_position": 0, "original_sequence_index": 0, "event_token": "E4", "attribution": 0.1},
            {"event_id": 5, "is_padding": False, "rank": 1, "input_position": 1, "original_sequence_index": 1, "event_token": "E5", "attribution": 0.8},
            {"event_id": 6, "is_padding": False, "rank": 2, "input_position": 2, "original_sequence_index": 2, "event_token": "E6", "attribution": -0.4},
            {"event_id": 0, "is_padding": True, "rank": None, "input_position": 3, "original_sequence_index": None, "event_token": "__PAD__", "attribution": 0.0},
        ],
    }


class RandomEventBaselineTests(unittest.TestCase):
    def test_same_seed_reproduces_same_random_set_and_budget(self):
        first, first_seed = select_random_events(_record(), top_k=2, perturbation_seed=2026)
        second, second_seed = select_random_events(_record(), top_k=2, perturbation_seed=2026)
        self.assertEqual([event["event_id"] for event in first], [event["event_id"] for event in second])
        self.assertEqual(first_seed, second_seed)
        self.assertEqual(len(first), 2)

    def test_random_record_does_not_copy_integrated_gradients_scores(self):
        result = build_random_event_record(
            _record(), source_path=Path("source.jsonl.gz"), top_k=2, perturbation_seed=2026
        )
        self.assertEqual(result["condition"], "random_event")
        self.assertFalse(result["uses_attribution_scores"])
        self.assertFalse(result["llm_used"])
        self.assertFalse(result["prompt_rendered"])
        self.assertEqual(result["selected_event_count"], 2)
        self.assertNotIn("attribution", result["selected_events"][0])
        self.assertNotIn("rank", result["selected_events"][0])

    def test_rejects_non_integrated_gradients_source(self):
        record = _record()
        record["method"] = "event_occlusion"
        with self.assertRaises(ValueError):
            select_random_events(record, top_k=1)

    def test_run_writes_records_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = temporary_path / "seed_13_test_integrated_gradients.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(_record()) + "\n")
            output = temporary_path / "random.jsonl.gz"
            manifest = run_random_event(
                source, top_k=2, perturbation_seed=2026, output=output, show_progress=False
            )
            self.assertEqual(manifest["count"], 1)
            self.assertEqual(manifest["selected_event_count"], 2)
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                result = json.loads(handle.readline())
            self.assertEqual(result["selected_event_count"], 2)
            self.assertTrue((temporary_path / "random.manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
