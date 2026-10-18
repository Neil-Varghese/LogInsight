import unittest

try:
    from no_detector_context_prompt import (
        PROMPT_TEMPLATE_ID,
        build_no_detector_context,
        render_no_detector_context_prompt,
    )
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.no_detector_context_prompt import (
        PROMPT_TEMPLATE_ID,
        build_no_detector_context,
        render_no_detector_context_prompt,
    )


def _record():
    return {
        "BlockId": "blk_123",
        "Label": "Anomaly",
        "Evidence": [
            {
                "Position": 0,
                "EventId": "E1",
                "LineId": 17,
                "EventTemplate": "Failure <*>",
                "ParameterList": ["host-a"],
                "RawLine": "failure on host-a",
            }
        ],
    }


class NoDetectorContextPromptTests(unittest.TestCase):
    def test_payload_contains_only_anomalous_evidence(self):
        payload = build_no_detector_context(_record())
        self.assertEqual(set(payload), {"anomalous_block_evidence"})
        self.assertEqual(payload["anomalous_block_evidence"]["events"][0]["event_id"], "E1")

    def test_storage_metadata_is_not_exposed(self):
        rendered = render_no_detector_context_prompt(_record())
        self.assertEqual(rendered[:2], ("no_detector_context", PROMPT_TEMPLATE_ID))
        self.assertNotIn("blk_123", rendered[2])
        self.assertNotIn('"Label"', rendered[2])
        self.assertNotIn("detector_output", rendered[2])

    def test_rendering_is_deterministic(self):
        first = render_no_detector_context_prompt(_record())
        second = render_no_detector_context_prompt(_record())
        self.assertEqual(first, second)

    def test_rejects_non_mapping_record(self):
        with self.assertRaises(ValueError):
            build_no_detector_context([])

    def test_rejects_empty_or_incomplete_evidence(self):
        with self.assertRaises(ValueError):
            build_no_detector_context({"Evidence": []})
        with self.assertRaises(ValueError):
            build_no_detector_context({"Evidence": [{"EventId": "E1"}]})


if __name__ == "__main__":
    unittest.main()
