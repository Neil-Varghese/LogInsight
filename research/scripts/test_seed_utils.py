"""Small self-test for the LogInsight-FV seed utilities."""

import unittest

import numpy as np

from seed_utils import derive_seed, make_numpy_rng, set_global_seed


class SeedUtilsTest(unittest.TestCase):
    def test_derived_seeds_are_stable_and_scoped(self):
        self.assertEqual(derive_seed(2026, "perturbation", "block-1"), derive_seed(2026, "perturbation", "block-1"))
        self.assertNotEqual(derive_seed(2026, "block-1"), derive_seed(2026, "block-2"))

    def test_numpy_subrun_is_reproducible(self):
        first = make_numpy_rng(2026, "sample", 1).integers(0, 100, size=8)
        second = make_numpy_rng(2026, "sample", 1).integers(0, 100, size=8)
        np.testing.assert_array_equal(first, second)

    def test_global_seed_report(self):
        report = set_global_seed(2026, deterministic=False)
        self.assertEqual(report["seed"], 2026)
        self.assertFalse(report["deterministic_requested"])


if __name__ == "__main__":
    unittest.main()
