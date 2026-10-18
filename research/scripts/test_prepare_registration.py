import tempfile
import unittest
from pathlib import Path

from prepare_registration import build_manifest


class RegistrationPreparationTests(unittest.TestCase):
    def test_manifest_hashes_exact_files_and_keeps_generation_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for name, content in (("a.txt", "alpha"), ("b.txt", "beta")):
                path = root / name
                path.write_text(content, encoding="utf-8")
                files.append(name)
            manifest = build_manifest(root, files=files)
            self.assertEqual(manifest["status"], "prepared_pending_external_registration")
            self.assertFalse(manifest["llm_output_guard"]["llm_output_generation_allowed"])
            self.assertEqual(len(manifest["frozen_files"]), 2)
            self.assertTrue(all(len(record["sha256"]) == 64 for record in manifest["frozen_files"]))

    def test_registration_metadata_enables_generation_only_with_platform_and_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.txt").write_text("protocol", encoding="utf-8")
            manifest = build_manifest(root, files=["protocol.txt"], platform="OSF", registration_url="https://osf.io/example/")
            self.assertEqual(manifest["status"], "registered")
            self.assertTrue(manifest["llm_output_guard"]["llm_output_generation_allowed"])
            self.assertFalse(manifest["llm_output_guard"]["external_registration_required_for_current_run"])

    def test_exploratory_mode_allows_pilot_but_not_confirmatory_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.txt").write_text("protocol", encoding="utf-8")
            manifest = build_manifest(root, files=["protocol.txt"], exploratory=True)
            self.assertEqual(manifest["status"], "exploratory_unregistered")
            self.assertTrue(manifest["llm_output_guard"]["llm_output_generation_allowed"])
            self.assertFalse(manifest["llm_output_guard"]["external_registration_required_for_current_run"])

    def test_url_without_platform_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.txt").write_text("protocol", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "platform is required"):
                build_manifest(root, files=["protocol.txt"], registration_url="https://example.com")


if __name__ == "__main__":
    unittest.main()
