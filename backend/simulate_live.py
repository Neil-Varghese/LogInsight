"""Demo helper: write a log file a few lines at a time so Live mode has something to watch.

Usage: python backend/simulate_live.py [source] [target] [lines_per_second]
Then open the page, refresh, pick the target file and click "Watch live".
"""

from pathlib import Path
import sys
import time

TEST_SETS = Path(__file__).resolve().parent.parent / "test sets"
source = Path(sys.argv[1]) if len(sys.argv) > 1 else TEST_SETS / "sample_complete_sessions.log"
target = Path(sys.argv[2]) if len(sys.argv) > 2 else TEST_SETS / "live_demo.log"
rate = float(sys.argv[3]) if len(sys.argv) > 3 else 20

target.write_text("", encoding="utf-8")
with source.open(encoding="utf-8", errors="replace") as src, target.open("a", encoding="utf-8") as out:
    for count, line in enumerate(src, 1):
        out.write(line)
        out.flush()  # whole lines only, so the watcher never sees half a line
        time.sleep(1 / rate)
        if count % 100 == 0:
            print(f"wrote {count} lines to {target.name}", flush=True)
print("done")
