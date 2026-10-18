import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    from validate_attribution_storage import build_storage_manifest, validate_attribution_file
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.validate_attribution_storage import build_storage_manifest, validate_attribution_file


def _write(path: Path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _ig_record(unit_id="u1"):
    return {
        "method": "integrated_gradients",
        "unit_id": unit_id,
        "events": [{"event_id": 4, "attribution": 0.8, "absolute_attribution": 0.8}],
    }


def _occlusion_record(unit_id="u1"):
    return {
        "method": "event_occlusion",
        "unit_id": unit_id,
        "verification_only": True,
        "prompt_visible": False,
        "events": [{"event_id": 4, "occlusion_score": 0.3, "occluded_probability": 0.2}],
    }


def _loo_record(unit_id="u1"):
    return {
        "method": "leave_one_event_out",
        "unit_id": unit_id,
        "verification_only": True,
        "prompt_visible": False,
        "events": [{"event_id": 4, "leave_one_event_out_score": 0.3, "held_out_probability": 0.2}],
    }


class AttributionStorageTests(unittest.TestCase):
    def test_registers_separate_methods_without_merging_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ig = root / "ig.jsonl.gz"
            occlusion = root / "occlusion.jsonl.gz"
            loo = root / "loo.jsonl.gz"
            _write(ig, [_ig_record()])
            _write(occlusion, [_occlusion_record()])
            _write(loo, [_loo_record()])
            output = root / "registry.json"
            manifest = build_storage_manifest(
                integrated_gradients_files=[ig],
                occlusion_files=[occlusion],
                leave_one_event_out_files=[loo],
                output=output,
            )
            self.assertEqual([entry["method"] for entry in manifest["entries"]], [
                "integrated_gradients", "event_occlusion", "leave_one_event_out"
            ])
            self.assertTrue(output.exists())
            self.assertEqual(manifest["storage_policy"], "one_method_per_file_no_vector_merging")

    def test_rejects_mixed_method_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.jsonl.gz"
            _write(path, [_ig_record(), _occlusion_record("u2")])
            with self.assertRaises(ValueError):
                validate_attribution_file(path, "integrated_gradients")

    def test_rejects_verification_file_without_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = _occlusion_record()
            record.pop("prompt_visible")
            path = Path(temporary) / "bad.jsonl.gz"
            _write(path, [record])
            with self.assertRaises(ValueError):
                validate_attribution_file(path, "event_occlusion")

    def test_rejects_duplicate_path_registered_for_two_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ig.jsonl.gz"
            _write(path, [_ig_record()])
            with self.assertRaises(ValueError):
                build_storage_manifest(
                    integrated_gradients_files=[path],
                    occlusion_files=[path],
                    leave_one_event_out_files=[],
                    output=Path(temporary) / "registry.json",
                )

    def test_rejects_verification_fields_in_integrated_gradients(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = _ig_record()
            record["events"][0]["occlusion_score"] = 0.4
            path = Path(temporary) / "bad_ig.jsonl.gz"
            _write(path, [record])
            with self.assertRaises(ValueError):
                validate_attribution_file(path, "integrated_gradients")

    def test_rejects_integrated_gradients_without_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = _ig_record()
            del record["events"][0]["absolute_attribution"]
            path = Path(temporary) / "incomplete_ig.jsonl.gz"
            _write(path, [record])
            with self.assertRaises(ValueError):
                validate_attribution_file(path, "integrated_gradients")


if __name__ == "__main__":
    unittest.main()
