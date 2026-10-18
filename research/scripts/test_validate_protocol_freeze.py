import copy
import json
import unittest
from pathlib import Path

from validate_protocol_freeze import validate_freeze


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "experiments" / "protocol_freeze_step62.json"


class ProtocolFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_registered_definitions_are_complete(self):
        result = validate_freeze(self.document)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["top_k_events"], 5)
        self.assertEqual(result["primary_shuffle"], "within_block_permutation")

    def test_missing_counterpart_rule_is_rejected(self):
        document = copy.deepcopy(self.document)
        document["normal_replacement"]["no_counterpart_rule"] = "use_nearest_event"
        with self.assertRaisesRegex(ValueError, "unchanged/unavailable"):
            validate_freeze(document)

    def test_primary_strata_are_explicit(self):
        strata = self.document["primary_strata"]
        self.assertEqual(strata["dataset"]["primary"], "HDFS")
        self.assertEqual(strata["detector"]["primary"], "loginsight_lstm")
        self.assertEqual(strata["llm"]["primary"], "llm_primary")


if __name__ == "__main__":
    unittest.main()
