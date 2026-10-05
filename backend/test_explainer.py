"""Offline tests for explainer.py (no network, no real key). Run: python -m unittest discover -s backend"""

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explainer

REPLY = {"summary": "s", "likely_cause": "c", "confidence": "low",
         "incomplete_sequence": True, "suggested_checks": ["x"]}


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *_):
        return json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps(REPLY)}]}}]}).encode()


class TestExplain(unittest.TestCase):
    def setUp(self):
        for patcher in (mock.patch.object(explainer, "_templates", return_value={"e1": "Receiving block <*>"}),
                        mock.patch.object(explainer, "_env", side_effect=lambda name, default="": "fake-key" if name == "GEMINI_API_KEY" else default)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def call(self, event_ids):
        sent = []
        def fake_urlopen(request, timeout=None):
            sent.append(request)
            return FakeResponse()
        with mock.patch.object(explainer.urllib.request, "urlopen", fake_urlopen):
            return explainer.explain("b", 0.9, event_ids), sent[0]

    def test_missing_key_raises(self):
        with mock.patch.object(explainer, "_env", return_value=""):
            with self.assertRaises(RuntimeError):
                explainer.explain("b", 0.9, ["e1"])

    def test_returns_parsed_reply_and_key_only_in_header(self):
        result, request = self.call(["e1"])
        body = request.data.decode()
        self.assertEqual(result, REPLY)
        self.assertEqual(request.get_header("X-goog-api-key"), "fake-key")
        self.assertNotIn("fake-key", body)
        self.assertIn("Receiving block <*>", body)

    def test_empty_reply_gives_clear_error(self):
        empty = FakeResponse()
        empty.read = lambda *_: json.dumps({"candidates": []}).encode()
        with mock.patch.object(explainer.urllib.request, "urlopen", lambda *a, **k: empty):
            with self.assertRaisesRegex(RuntimeError, "no usable answer"):
                explainer.explain("b", 0.9, ["e1"])

    def test_raw_lines_are_masked_in_request(self):
        sent = []
        def fake_urlopen(request, timeout=None):
            sent.append(request)
            return FakeResponse()
        line = "Receiving block blk_1 src: /10.250.19.102:54106 dest: /10.251.30.85:50010"
        with mock.patch.object(explainer.urllib.request, "urlopen", fake_urlopen):
            explainer.explain("b", 0.9, ["e1"], [line])
        body = sent[0].data.decode()
        self.assertNotIn("10.250.19.102", body)
        self.assertNotIn("10.251.30.85", body)
        self.assertIn("<IP>", body)
        self.assertIn("Receiving block blk_1", body)

    def test_long_sequences_are_truncated(self):
        _, request = self.call(["e1"] * (explainer.MAX_EVENTS + 20))
        body = request.data.decode()
        self.assertIn(f"{explainer.MAX_EVENTS}. Receiving block", body)
        self.assertNotIn(f"{explainer.MAX_EVENTS + 1}. Receiving block", body)


if __name__ == "__main__":
    unittest.main()
