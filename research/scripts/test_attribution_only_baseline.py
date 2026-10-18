import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    from attribution_only_baseline import (
        build_attribution_only_record,
        run_attribution_only,
        select_top_k_events,
    )
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.attribution_only_baseline import (
        build_attribution_only_record,
        run_attribution_only,
        select_top_k_events,
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
            {"event_id": 0, "is_padding": True, "rank": None, "attribution": 0.0},
            {"event_id": 4, "is_padding": False, "rank": 2, "attribution": -0.2, "absolute_attribution": 0.2, "input_position": 1},
            {"event_id": 5, "is_padding": False, "rank": 1, "attribution": 0.8, "absolute_attribution": 0.8, "input_position": 2},
            {"event_id": 6, "is_padding": False, "rank": 3, "attribution": 0.1, "absolute_attribution": 0.1, "input_position": 3},
        ],
    }


class AttributionOnlyBaselineTests(unittest.TestCase):
    def test_selects_ranked_non_padding_events(self):
        selected = select_top_k_events(_record(), top_k=2)
        self.assertEqual([event["event_id"] for event in selected], [5, 4])
        self.assertEqual([event["rank"] for event in selected], [1, 2])

    def test_baseline_record_has_no_llm_and_preserves_source(self):
        result = build_attribution_only_record(
            _record(), source_path=Path("source.jsonl.gz"), top_k=2
        )
        self.assertEqual(result["condition"], "attribution_only")
        self.assertFalse(result["llm_used"])
        self.assertFalse(result["prompt_rendered"])
        self.assertEqual(result["source_method"], "integrated_gradients")
        self.assertEqual(result["selected_event_ids"], [5, 4])
        self.assertNotIn("events", result)

    def test_rejects_non_integrated_gradients_source(self):
        record = _record()
        record["method"] = "event_occlusion"
        with self.assertRaises(ValueError):
            select_top_k_events(record, top_k=1)

    def test_run_writes_records_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = temporary_path / "seed_13_test_integrated_gradients.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(_record()) + "\n")
            output = temporary_path / "baseline.jsonl.gz"
            manifest = run_attribution_only(source, top_k=2, output=output, show_progress=False)
            self.assertEqual(manifest["count"], 1)
            self.assertEqual(manifest["selected_event_count"], 2)
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                result = json.loads(handle.readline())
            self.assertEqual(result["selected_event_ids"], [5, 4])
            self.assertTrue((temporary_path / "baseline.manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
