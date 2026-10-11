import json
import tempfile
import unittest
from pathlib import Path

from event_vocabulary import (
    PAD_TOKEN,
    UNKNOWN_EVENT_TOKEN,
    build_event_vocabulary,
    load_vocabulary,
    save_vocabulary,
)


class EventVocabularyTest(unittest.TestCase):
    def test_unknown_row_is_reserved_and_unseen_events_are_mapped(self):
        vocabulary = build_event_vocabulary([["E2", "E1"], ["E3"]])
        self.assertEqual(vocabulary.pad_id, 0)
        self.assertEqual(vocabulary.token_to_id["E1"], 1)
        self.assertEqual(vocabulary.token_to_id["E3"], 3)
        self.assertEqual(vocabulary.unknown_id, 4)
        self.assertEqual(vocabulary.encode(["E1", "UNSEEN", "E3"]), [1, 4, 3])
        self.assertEqual(vocabulary.decode([1, 4, 999]), ["E1", UNKNOWN_EVENT_TOKEN, UNKNOWN_EVENT_TOKEN])

    def test_save_and_load_preserve_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocabulary.json"
            vocabulary = build_event_vocabulary([["E1"]])
            payload = save_vocabulary(vocabulary, path)
            loaded = load_vocabulary(path)
            self.assertEqual(payload["fit_scope"], "training_only")
            self.assertEqual(loaded.to_dict(), vocabulary.to_dict())
            self.assertEqual(loaded.pad_token, PAD_TOKEN)

    def test_training_scope_cannot_be_relaxed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_vocabulary(
                    build_event_vocabulary([["E1"]]),
                    Path(directory) / "vocabulary.json",
                    fit_scope="all_data",
                )


if __name__ == "__main__":
    unittest.main()
