"""Reproducibility utilities for LogInsight-FV experiments.

Use ``set_global_seed`` once at the beginning of a process and use
``derive_seed`` for deterministic sub-runs such as detector seeds,
perturbations, annotation sampling, and repeated generations.

Important: PYTHONHASHSEED must be set before starting Python to control hash
randomization completely. This module sets it for child processes and records
that limitation in the returned report.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import numpy as np


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit seed for a named sub-run.

    The same master seed and parts always produce the same seed, independent
    of Python's process-specific hash randomization.
    """

    _validate_seed(master_seed)
    material = "|".join([str(master_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def make_numpy_rng(master_seed: int, *parts: object) -> np.random.Generator:
    """Return an isolated NumPy generator for a deterministic sub-run."""

    return np.random.default_rng(derive_seed(master_seed, *parts))


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and TensorFlow and return an environment report.

    TensorFlow is imported lazily so preprocessing-only scripts do not pay the
    model import cost. TensorFlow deterministic operations can reduce GPU
    throughput, so callers may explicitly disable them for exploratory runs;
    registered experiments should leave ``deterministic=True``.
    """

    _validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    tensorflow_version = None
    tensorflow_seeded = False
    tensorflow_determinism_enabled = False
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tensorflow_version = tf.__version__
        tensorflow_seeded = True
        if deterministic:
            tf.config.experimental.enable_op_determinism()
            tensorflow_determinism_enabled = True
    except ImportError:
        # Preprocessing and schema-validation environments do not need TF.
        pass

    return {
        "seed": seed,
        "python_hash_seed_requested": str(seed),
        "python_hash_seed_requires_process_restart": True,
        "deterministic_requested": deterministic,
        "tensorflow_seeded": tensorflow_seeded,
        "tensorflow_version": tensorflow_version,
        "tensorflow_determinism_enabled": tensorflow_determinism_enabled,
    }


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Set and report LogInsight experiment seeds.")
    parser.add_argument("seed", type=int)
    parser.add_argument("--non-deterministic", action="store_true")
    args = parser.parse_args()
    report = set_global_seed(args.seed, deterministic=not args.non_deterministic)
    print(json.dumps(report, indent=2, sort_keys=True))
