import gzip
import json
import tempfile
import unittest
from pathlib import Path

from strip_bgl_label_prefix import split_label_prefix, strip_bgl_labels


class StripBGLLabelPrefixTest(unittest.TestCase):
    def test_split_normal_and_alert_lines(self):
        prefix, clean, alert = split_label_prefix("- 1117838570 2005.06.03 node payload")
        self.assertEqual((prefix, clean, alert), ("-", "1117838570 2005.06.03 node payload", 0))
        prefix, clean, alert = split_label_prefix("APPREAD 1117869872 2005.06.04 node payload")
        self.assertEqual((prefix, clean, alert), ("APPREAD", "1117869872 2005.06.04 node payload", 1))

    def test_writes_clean_input_and_separate_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "BGL.log"
            clean = root / "BGL_clean.log"
            audit = root / "audit.jsonl.gz"
            manifest = root / "manifest.json"
            source.write_text(
                "- 1117838570 2005.06.03 node normal\n"
                "APPREAD 1117869872 2005.06.04 node alert\n",
                encoding="utf-8",
            )
            summary = strip_bgl_labels(source, clean, audit, manifest, show_progress=False)
            self.assertEqual(summary["input_line_count"], 2)
            self.assertEqual(summary["alert_line_count"], 1)
            self.assertEqual(
                clean.read_text(encoding="utf-8"),
                "1117838570 2005.06.03 node normal\n1117869872 2005.06.04 node alert\n",
            )
            with gzip.open(audit, "rt", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(records[0]["RemovedLabelPrefix"], "-")
            self.assertEqual(records[1]["RemovedLabelPrefix"], "APPREAD")
            self.assertEqual(records[1]["IsAlert"], 1)


if __name__ == "__main__":
    unittest.main()
