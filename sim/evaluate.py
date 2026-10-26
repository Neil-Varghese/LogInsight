"""Score the model against the Docker cluster's answer key (sim/labels/*.csv).

Usage: python sim/evaluate.py [threshold]     (run while or after the server has been watching sim/cluster.log)
Only blocks in both the answer key and backend/monitor.sqlite3 are counted, and only once finished:
no new line for SETTLE seconds (a half-written block is scored on a partial sequence, which the LSTM was not trained on).
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTLE = 60
threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

truth = {}
for csv_file in (ROOT / "sim" / "labels").glob("*.csv"):
    for line in csv_file.read_text().splitlines():
        block, _, label = line.partition(",")
        truth[block] = label == "Anomaly"

with sqlite3.connect(ROOT / "backend" / "monitor.sqlite3") as db:
    rows = db.execute("SELECT block_id, anomaly_score, last_seen FROM blocks").fetchall()
newest = max((last for _, _, last in rows), default=0)
scores = {block: score for block, score, last in rows if newest - last >= SETTLE}

tp = fp = fn = tn = 0
for block, is_odd in truth.items():
    if block in scores:
        flagged = scores[block] > threshold
        tp += flagged and is_odd
        fp += flagged and not is_odd
        fn += not flagged and is_odd
        tn += not flagged and not is_odd

n = tp + fp + fn + tn
if not n:
    sys.exit("No scored blocks match the answer key yet. Is the server watching sim/cluster.log?")
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
print(f"finished blocks scored: {n} of {len(truth)} generated   threshold: {threshold}")
print(f"TP {tp}  FP {fp}  FN {fn}  TN {tn}")
print(f"accuracy {(tp + tn) / n:.4f}  precision {precision:.4f}  recall {recall:.4f}  F1 {f1:.4f}")
