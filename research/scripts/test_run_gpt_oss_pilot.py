import os
import tempfile
import unittest
from pathlib import Path

from run_gpt_oss_pilot import ProgressBar, load_repo_env, load_selected_by_unit, top_k_attribution_payload


class PilotRunnerTests(unittest.TestCase):
    def test_top_k_payload_excludes_padding_and_respects_rank(self):
        record = {
            "events": [
                {"is_padding": True, "input_position": 0, "rank": None, "event_id": 0},
                {"is_padding": False, "input_position": 2, "rank": 2, "event_id": 12, "attribution": 0.2, "absolute_attribution": 0.2},
                {"is_padding": False, "input_position": 1, "rank": 1, "event_id": 11, "attribution": 0.3, "absolute_attribution": 0.3},
            ]
        }
        payload = top_k_attribution_payload(record, 1)
        self.assertEqual(payload["top_k"], 1)
        self.assertEqual(payload["events"][0]["event_id"], "11")

    def test_progress_bar_can_be_disabled(self):
        progress = ProgressBar(2, enabled=False)
        progress.update("one")
        progress.close()
        self.assertEqual(progress.completed, 1)

    def test_load_selected_by_unit_reads_only_requested_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(
                '{"BlockId":"a","Sequence":["A"],"LabelId":0}\n'
                '{"BlockId":"b","Sequence":["B"],"LabelId":0}\n',
                encoding="utf-8",
            )
            records = load_selected_by_unit(path, {"b"}, unit_key="BlockId")
            self.assertEqual(set(records), {"b"})
            self.assertEqual(records["b"]["Sequence"], ["B"])

    def test_load_repo_env_does_not_override_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("LOGINSIGHT_TEST_ENV='from-file'\n", encoding="utf-8")
            previous = os.environ.get("LOGINSIGHT_TEST_ENV")
            try:
                os.environ.pop("LOGINSIGHT_TEST_ENV", None)
                load_repo_env(path)
                self.assertEqual(os.environ["LOGINSIGHT_TEST_ENV"], "from-file")
                os.environ["LOGINSIGHT_TEST_ENV"] = "existing"
                load_repo_env(path)
                self.assertEqual(os.environ["LOGINSIGHT_TEST_ENV"], "existing")
            finally:
                if previous is None:
                    os.environ.pop("LOGINSIGHT_TEST_ENV", None)
                else:
                    os.environ["LOGINSIGHT_TEST_ENV"] = previous


if __name__ == "__main__":
    unittest.main()
