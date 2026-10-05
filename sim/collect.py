"""Start the fake cluster and copy its combined logs into sim/cluster.log (what the monitor watches).

Usage: python sim/collect.py        (Ctrl+C stops the copy and the containers)
Then start the server with LOGINSIGHT_WATCH=sim/cluster.log
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "sim" / "cluster.log"

subprocess.run(["docker", "compose", "up", "-d"], cwd=ROOT, check=True)
try:
    # Raw bytes straight to the file: a shell ">" on Windows PowerShell would re-encode it as UTF-16.
    with open(LOG, "wb") as out:
        subprocess.run(["docker", "compose", "logs", "-f", "--no-log-prefix", "--tail", "0"], cwd=ROOT, stdout=out)
except KeyboardInterrupt:
    pass
finally:
    subprocess.run(["docker", "compose", "down"], cwd=ROOT)
