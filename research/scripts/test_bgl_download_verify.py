import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from download_verify_bgl import extract_bgl_log, file_hashes, raw_log_statistics


class BGLDownloadVerifyTest(unittest.TestCase):
    def test_hashes_and_extract_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "BGL.zip"
            raw = root / "BGL.log"
            content = b"- 2005-01-01 sample normal\n0x0001 2005-01-01 sample alert\n"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("BGL.log", content)
            stats = extract_bgl_log(archive, raw, show_progress=False)
            self.assertEqual(stats["line_count"], 2)
            self.assertEqual(stats["bytes"], len(content))
            self.assertEqual(stats["md5"], hashlib.md5(content).hexdigest())
            self.assertEqual(raw_log_statistics(raw)["first_line"], "- 2005-01-01 sample normal")
            self.assertEqual(file_hashes(archive)["bytes"], archive.stat().st_size)


if __name__ == "__main__":
    unittest.main()
