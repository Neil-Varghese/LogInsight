# Research scripts

Place deterministic preprocessing, training, attribution, perturbation,
evaluation, and validation scripts here. Scripts should accept explicit input
and output paths and record the configuration/run ID they use.

`seed_utils.py` is the shared seed implementation. Call
`set_global_seed(seed, deterministic=True)` at process startup, pass explicit
`random_state` values to scikit-learn, and use `derive_seed` or
`make_numpy_rng` for named sub-runs. Record the returned seed report in the
experiment ledger. LLM providers may not honor a seed, so repeated-generation
indices and the provider/model configuration must also be recorded.

`build_hdfs_blocks.py` reads the Drain structured HDFS CSV, extracts BlockIds
from `Content`, sorts events chronologically by `LineId`, joins the official labels, and
writes one compressed JSONL record per complete block. Use `--max-rows` only for
smoke tests; the full dataset must be built without that option. The normal
run displays a dependency-free progress bar for both reading and writing:

    python3 research/scripts/build_hdfs_blocks.py

`build_hdfs_block_evidence.py` builds the explanation-ready companion artifact.
It preserves each block's event-ID sequence and stores, for every event, the
exact raw line, EventTemplate, parsed ParameterList, source LineId, EventId,
and zero-based position within the block. It also writes a schema/provenance
manifest. The original raw log must be aligned one-to-one with the structured
CSV; the default paths are the repository's HDFS files:

    python3 research/scripts/build_hdfs_block_evidence.py

Use `--max-rows` only for smoke tests. Do not use it for the final artifact.

`download_verify_bgl.py` downloads the official LogHub BGL archive, verifies
the published archive MD5 and size, extracts `BGL.log`, verifies the official
line count, and records archive/raw MD5 and SHA-256 checksums in a manifest.
The download is intentionally run from the user's terminal because it needs
network access:

    python3 research/scripts/download_verify_bgl.py

`strip_bgl_label_prefix.py` removes only the first BGL label token from each
line before parsing. It writes parser-safe `BGL_clean.log` and a separate
compressed audit stream containing the original line, removed prefix, and
derived alert indicator. The audit stream must not be sent to the detector or
LLM. The full run displays a progress bar:

    python3 research/scripts/strip_bgl_label_prefix.py

`build_bgl_windows.py` freezes the BGL transfer unit as non-overlapping
60-second windows anchored to Unix epoch time. A window is anomalous when any
source line in it had an alert prefix. It stores inclusive clean-log line
ranges, so it does not duplicate the large cleaned log:

    python3 research/scripts/build_bgl_windows.py
