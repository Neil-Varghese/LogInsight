"""Fake HDFS datanode: replays real HDFS block sessions as a live log stream on stdout.

Each session is copied from the sample files with a fresh block id and a current timestamp.
Several sessions run at once and their lines interleave, like a busy node. Docker captures stdout.
"""

import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

SAMPLES = Path(os.environ.get("SAMPLES", "/samples"))
RATE = float(os.environ.get("LINES_PER_SECOND", 5))   # per container
ANOMALY = float(os.environ.get("ANOMALY_RATE", 0.05))  # share of sessions taken from the anomalous sample
CONCURRENT = int(os.environ.get("CONCURRENT_BLOCKS", 8))
LINE = re.compile(r"^\d{6} \d{6} (.*)$")  # drop the old date/time, keep "pid LEVEL component: message"
BLOCK = re.compile(r"blk_-?\d+")


def sessions(path):
    by_block = {}
    for line in path.read_text().splitlines():
        m, b = LINE.match(line), BLOCK.search(line)
        if m and b:
            by_block.setdefault(b.group(), []).append(m.group(1))
    return list(by_block.items())


normal = sessions(SAMPLES / "sample_normal_sessions.log")
odd = sessions(SAMPLES / "sample_anomalous_sessions.log")
active = []  # each item is one block's remaining lines, last line first

while True:
    while len(active) < CONCURRENT:
        old, lines = random.choice(odd if random.random() < ANOMALY else normal)
        new = f"blk_{random.randint(-2**62, 2**62)}"
        active.append([l.replace(old, new) for l in lines][::-1])
    block = random.choice(active)
    print(f"{datetime.now(timezone.utc):%y%m%d %H%M%S} {block.pop()}", flush=True)
    if not block:
        active.remove(block)
    time.sleep(random.expovariate(RATE))
