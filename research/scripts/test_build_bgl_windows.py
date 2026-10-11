import gzip
import json
import tempfile
import unittest
from pathlib import Path

from build_bgl_windows import build_bgl_windows
from strip_bgl_label_prefix import strip_bgl_labels


class BuildBGLWindowsTest(unittest.TestCase):
    def test_builds_disjoint_epoch_windows_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "BGL.log"
            clean = root / "BGL_clean.log"
            audit = root / "audit.jsonl.gz"
            output = root / "windows.jsonl.gz"
            manifest = root / "manifest.json"
            raw.write_text(
                "- 1117838570 2005.06.03 node normal\n"
                "APPREAD 1117838571 2005.06.03 node alert\n"
                "- 1117838630 2005.06.03 node normal\n",
                encoding="utf-8",
            )
            strip_bgl_labels(raw, clean, audit, show_progress=False)
            summary = build_bgl_windows(clean, audit, output, manifest, show_progress=False)
            self.assertEqual(summary["window_count"], 2)
            self.assertEqual(summary["alert_window_count"], 1)
            self.assertEqual(summary["normal_window_count"], 1)
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(records[0]["LineStart"], 1)
            self.assertEqual(records[0]["LineEnd"], 2)
            self.assertEqual(records[0]["Label"], "Anomaly")
            self.assertEqual(records[1]["LineStart"], 3)
            self.assertEqual(records[1]["LineEnd"], 3)
            self.assertEqual(records[1]["Label"], "Normal")
            self.assertEqual(json.loads(manifest.read_text())["window_rule"]["overlap_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
