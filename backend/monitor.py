"""Always-on watcher: tails one HDFS log file, scores every block as new lines arrive, keeps results in SQLite.

Unlike a batch run it never re-reads the file: it remembers a byte offset, turns each new line into an event
by matching it against the trained templates, and re-scores only the blocks that just got new events.
"""

from collections import Counter, deque
from functools import lru_cache
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re
import sqlite3
import threading
import time

import predictor
from explainer import _templates

HEADER = re.compile(r"^(\d{6}) (\d{6}) \d+ \w+ [^:]+: (.*)$")  # <Date> <Time> <Pid> <Level> <Component>: <Content>
BLOCK = re.compile(r"(blk_-?\d+)")
FEED = re.compile(r"^(\d{6}) (\d{6}) \d+ (\w+) ([^:]+): (.*)$")  # same header as HEADER, but keeps level and component for the live feed
# Threat ranking: HDFS logs with Log4j levels, which map onto the RFC 5424 syslog severity numbers (lower = worse):
# 2 Critical (FATAL), 3 Error, 4 Warning, 5 Notice (a level we do not recognise: needs a look, never ignored), 6 Informational, 7 Debug.
SEVERITY = {"FATAL": 2, "ERROR": 3, "WARN": 4, "WARNING": 4, "INFO": 6, "DEBUG": 7, "TRACE": 7}
UNKNOWN_SEVERITY = 5
FEED_SIZE = 5000
# One block copied over the network: "Received block blk_1 of size 67108864 from /10.1.1.1" (or "... src: /10.1.1.1:5000 dest: ... of size N").
TRANSFER = re.compile(r"^Received block \S+ .*?of size (\d+)")
IPADDR = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?!\d)")  # any IPv4 in a line, port excluded
SOURCE = re.compile(r"(?:from|src:) /([\d.]+)")
WINDOW = predictor.MAX_SEQUENCE_LENGTH  # the model sees the last 50 events of a block
POLL_SECONDS = 1
MAX_CHUNK = 8 * 1024 * 1024  # bytes read per tick, so a huge backlog is worked through in slices
BUCKETS = (10, 60, 300, 900, 3600, 21600, 86400)  # chart bucket sizes in seconds; the smallest giving <= 60 bars wins
TOP_EVENT_BLOCKS = 20000  # ponytail: "top events" looks at the 20k most recent blocks; aggregate incrementally if that is too slow

SCHEMA = """
CREATE TABLE IF NOT EXISTS blocks (
    block_id TEXT PRIMARY KEY, anomaly_score REAL NOT NULL, is_anomalous INTEGER NOT NULL,
    sequence_len INTEGER NOT NULL, event_ids TEXT NOT NULL, first_seen INTEGER, last_seen INTEGER);
CREATE TABLE IF NOT EXISTS explanations (block_id TEXT PRIMARY KEY, seq_len INTEGER NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS traffic (t INTEGER NOT NULL, src TEXT NOT NULL, bytes INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS ip_activity (t INTEGER NOT NULL, ip TEXT NOT NULL, n INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS blocks_last_seen ON blocks(last_seen);
CREATE INDEX IF NOT EXISTS traffic_t ON traffic(t);
"""


@lru_cache(maxsize=1)
def _event2idx():
    with predictor.EVENT_MAPPING_PATH.open("rb") as mapping_file:
        return pickle.load(mapping_file)


@lru_cache(maxsize=4096)
def _epoch(date, clock):
    """HDFS stamps lines yymmdd hhmmss with no timezone; treat them as UTC so charts are consistent."""
    return int(datetime.strptime(date + clock, "%y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp())


def model_scorer():
    """Load the Keras model once and return a function: list of encoded event lists -> list of probabilities."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, _ = predictor._load_model_and_event_mapping()

    def score(encoded):
        padded = pad_sequences(encoded, maxlen=WINDOW, padding="post")
        return model.predict(padded, batch_size=256, verbose=0).reshape(-1).tolist()

    return score


class Monitor:
    def __init__(self, log_path, db_path, scorer=None):
        self.log_path, self.db_path = Path(log_path), Path(db_path)
        self.scorer = scorer  # None = load the Keras model in the background thread (a fake can be injected in tests)
        self.lock = threading.Lock()  # one tick or reset at a time
        self.stop = threading.Event()
        # ponytail: every block ever seen stays in memory; evict idle blocks if a feed has millions of them.
        self.state = {}  # block_id -> {"events": last 50 event ids, "n": total events, "first": t, "last": t}
        self.offset = self.lines = self.unmatched = 0
        self.feed = deque(maxlen=FEED_SIZE)  # the newest processed lines, for the live log view
        self.feed_seq = 0  # never reset, so a reader's "give me lines after N" stays valid across a reset
        self.generation = 0  # bumped on reset so readers know to clear what they show
        self.severity_counts = Counter()
        self.timings = deque(maxlen=200)  # per busy tick: (unix time, lines, blocks scored, total seconds, scoring seconds)
        self.model_load_s = None
        self.rates = deque(maxlen=120)  # (unix time, lines/sec) per tick, for the throughput chart
        self.info = {"model_loaded": scorer is not None, "error": None, "last_ingest": None, "started": time.time()}
        self.patterns = []

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="monitor").start()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")  # the dashboard reads while this thread writes
        connection.executescript(SCHEMA)
        return connection

    def _setup(self):
        event2idx = _event2idx()
        # A few trained templates have a real block id baked in (a quirk of how they were learned); they would
        # swallow that one block's lines, so only templates that wildcard the block id are used.
        self.patterns = [(p, i) for p, i in predictor._load_master_template_patterns()
                         if i in event2idx and "blk_" not in p.pattern]
        if self.scorer is None:
            loading = time.perf_counter()
            self.scorer = model_scorer()
            self.model_load_s = time.perf_counter() - loading
        self.info["model_loaded"] = True
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.touch()
        db = self._connect()
        self._hydrate(db)
        return db

    def _run(self):
        try:
            db = self._setup()
        except Exception as error:
            self.info["error"] = f"Monitor could not start: {error}"
            print(self.info["error"], flush=True)
            return
        while not self.stop.is_set():
            began = time.time()
            try:
                with self.lock:
                    count = self._tick(db)
                self.info["error"] = None
            except Exception as error:
                # Never die: report it, restore memory from what was saved, try again next tick.
                count = 0
                self.info["error"] = str(error)
                print(f"Monitor tick failed: {error}", flush=True)
                with self.lock:
                    self._hydrate(db)
            if count:
                self.info["last_ingest"] = time.time()
            self.stop.wait(POLL_SECONDS)
            self.rates.append((time.time(), count / max(time.time() - began, 0.001)))

    def _hydrate(self, db):
        """Rebuild memory from disk, so a restart (or a failed tick) continues exactly where the saved offset says."""
        self.state = {}
        for block_id, n, events, first, last in db.execute(
            "SELECT block_id, sequence_len, event_ids, first_seen, last_seen FROM blocks"
        ):
            self.state[block_id] = {"events": deque(json.loads(events), maxlen=WINDOW), "n": n, "first": first, "last": last}
        meta = dict(db.execute("SELECT key, value FROM meta"))
        same_file = meta.get("file") == str(self.log_path)
        self.offset = int(meta.get("offset", 0)) if same_file else 0
        self.lines, self.unmatched = int(meta.get("lines", 0)), int(meta.get("unmatched", 0))

    def match_event(self, content):
        for pattern, event_id in self.patterns:
            if pattern.match(content):
                return event_id
        return None

    def _tick(self, db):
        began = time.perf_counter()
        size = self.log_path.stat().st_size
        if size < self.offset:
            self.offset = 0  # file was truncated or rotated: read the new one from the top
        if size == self.offset:
            return 0
        with self.log_path.open("rb") as log_file:
            log_file.seek(self.offset)
            chunk = log_file.read(MAX_CHUNK)
        end = chunk.rfind(b"\n") + 1  # only whole lines; a half-written last line waits for the next tick
        if not end:
            return 0
        lines = chunk[:end].decode("utf-8", "replace").splitlines()
        event2idx, dirty, unmatched, transfers, activity = _event2idx(), set(), 0, [], Counter()
        for line in lines:
            header = HEADER.match(line)
            if not header:
                unmatched += 1
                continue
            date, clock, content = header.groups()
            block = BLOCK.search(content)
            for ip in set(IPADDR.findall(content)):
                activity[(_epoch(date, clock), ip)] += 1  # once per line, so a src/dest pair on the same IP is not double counted
            sent = TRANSFER.match(content)
            if sent and (source := SOURCE.search(content)):
                transfers.append((_epoch(date, clock), source.group(1), int(sent.group(1))))
            if not block:
                continue  # not about a block (the training pipeline ignores these too)
            event = self.match_event(content.strip())
            if event is None:
                unmatched += 1
                continue
            when = _epoch(date, clock)
            entry = self.state.setdefault(block.group(1), {"events": deque(maxlen=WINDOW), "n": 0, "first": when, "last": when})
            entry["events"].append(event)
            entry["n"] += 1
            entry["first"], entry["last"] = min(entry["first"], when), max(entry["last"], when)
            dirty.add(block.group(1))
        ids = sorted(dirty)
        scoring = time.perf_counter()
        scores = self.scorer([[event2idx[e] for e in self.state[b]["events"]] for b in ids]) if ids else []
        score_s = time.perf_counter() - scoring
        db.executemany(
            """INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(block_id) DO UPDATE SET
               anomaly_score = excluded.anomaly_score, is_anomalous = excluded.is_anomalous,
               sequence_len = excluded.sequence_len, event_ids = excluded.event_ids, last_seen = excluded.last_seen""",
            [(b, s, int(s > predictor.PREDICTION_THRESHOLD), self.state[b]["n"], json.dumps(list(self.state[b]["events"])),
              self.state[b]["first"], self.state[b]["last"]) for b, s in zip(ids, scores)],
        )
        db.executemany("INSERT INTO traffic VALUES (?, ?, ?)", transfers)
        db.executemany("INSERT INTO ip_activity VALUES (?, ?, ?)", [(t, ip, n) for (t, ip), n in activity.items()])
        new = {"offset": self.offset + end, "lines": self.lines + len(lines), "unmatched": self.unmatched + unmatched,
               "file": str(self.log_path)}
        db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in new.items()])
        db.commit()  # blocks and offset land together, so a crash can never double-count lines
        self.offset, self.lines, self.unmatched = new["offset"], new["lines"], new["unmatched"]
        for line in lines:
            self._feed(line)  # only after the save above: a line shows up once it has been processed
        self.timings.append((time.time(), len(lines), len(ids), time.perf_counter() - began, score_s))
        return len(lines)

    def _feed(self, line):
        header = FEED.match(line)
        level = header.group(3).upper() if header else "UNKNOWN"
        severity = SEVERITY.get(level, UNKNOWN_SEVERITY)
        block = BLOCK.search(line)
        self.feed_seq += 1
        self.severity_counts[severity] += 1
        self.feed.append({
            "seq": self.feed_seq, "t": _epoch(header.group(1), header.group(2)) if header else None, "level": level, "sev": severity,
            "src": header.group(4).rsplit(".", 1)[-1] if header else "", "text": header.group(5) if header else line,
            "block": block.group(1) if block else None})

    def feed_since(self, after, limit=500):
        """Lines processed after sequence number ``after`` (oldest first). A negative ``after`` means "just show me the latest 100"."""
        lines = list(self.feed)
        fresh = lines[-100:] if after < 0 else [entry for entry in lines if entry["seq"] > after]
        # ponytail: a reader more than `limit` lines behind skips the middle; page by seq if gap-free history matters
        return {"generation": self.generation, "latest": self.feed_seq, "counts": dict(self.severity_counts), "lines": fresh[-limit:]}

    def watch(self, path):
        """Switch to another log file and start from zero (a no-op if it is already the watched file)."""
        path = Path(path)
        if path == self.log_path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        with self.lock:
            self.log_path = path
        self.reset()

    def reset(self):
        """Forget everything and re-read the file from the top."""
        with self.lock:
            db = self._connect()
            try:
                for table in ("blocks", "explanations", "traffic", "ip_activity", "meta"):
                    db.execute(f"DELETE FROM {table}")
                db.commit()
            finally:
                db.close()
            self.state, self.offset, self.lines, self.unmatched = {}, 0, 0, 0
            self.rates.clear()
            self.feed.clear()
            self.severity_counts.clear()
            self.generation += 1

    def perf(self):
        """Speed numbers over the last ~200 busy ticks (ticks that found new lines)."""
        ticks = list(self.timings)
        busy = sorted(t[3] for t in ticks)
        lines, blocks, seconds, scoring = (sum(t[i] for t in ticks) for i in (1, 2, 3, 4))
        return {"model_load_s": self.model_load_s, "ticks": len(ticks),
                "tick_avg_ms": 1000 * seconds / len(ticks) if ticks else 0,
                "tick_p95_ms": 1000 * busy[int(0.95 * (len(busy) - 1))] if ticks else 0,
                "capacity_lines_per_s": lines / seconds if seconds else 0,  # lines handled per busy second = the most it could keep up with
                "score_ms_per_block": 1000 * scoring / blocks if blocks else 0,
                "db_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0, "uptime_s": time.time() - self.info["started"]}

    def snapshot(self):
        size = self.log_path.stat().st_size if self.log_path.exists() else 0
        return {**self.info, "file": self.log_path.name, "lines": self.lines, "unmatched": self.unmatched,
                "blocks": len(self.state), "backlog_bytes": max(0, size - self.offset), "now": time.time(),
                "rates": [{"t": t, "rate": round(r, 1)} for t, r in self.rates], "perf": self.perf()}


def summary(connection, threshold):
    """Everything the charts need, computed at ``threshold`` so the slider re-colours history instantly."""
    total, anomalous = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(anomaly_score >= ?), 0) FROM blocks", (threshold,)).fetchone()
    first, last = connection.execute("SELECT MIN(first_seen), MAX(last_seen) FROM blocks").fetchone()
    span = (last - first) if total else 0
    bucket = next((b for b in BUCKETS if span / b <= 60), BUCKETS[-1])
    series = [{"t": t, "normal": normal, "anomalous": anomalous_count} for t, normal, anomalous_count in connection.execute(
        "SELECT (last_seen / ?) * ? AS t, SUM(anomaly_score < ?), SUM(anomaly_score >= ?) FROM blocks "
        "WHERE last_seen IS NOT NULL GROUP BY t ORDER BY t", (bucket, bucket, threshold, threshold))]
    bins = dict(connection.execute("SELECT MIN(CAST(anomaly_score * 10 AS INTEGER), 9) AS b, COUNT(*) FROM blocks GROUP BY b"))
    histogram = [{"bin": i, "count": bins.get(i, 0)} for i in range(10)]

    seen, flagged, normal_total, flagged_total = Counter(), Counter(), 0, 0
    for is_flagged, events in connection.execute(
        "SELECT anomaly_score >= ?, event_ids FROM blocks ORDER BY last_seen DESC LIMIT ?", (threshold, TOP_EVENT_BLOCKS)):
        distinct = set(json.loads(events))
        if is_flagged:
            flagged.update(distinct)
            flagged_total += 1
        else:
            seen.update(distinct)
            normal_total += 1
    templates = _templates()
    # Events that show up in a bigger share of flagged blocks than of normal ones are the likely culprits.
    top = sorted(
        ({"event_id": e, "template": templates.get(e, e), "anomalous_pct": 100 * c / flagged_total,
          "normal_pct": 100 * seen[e] / normal_total if normal_total else 0.0} for e, c in flagged.items() if flagged_total),
        key=lambda row: row["normal_pct"] - row["anomalous_pct"])[:8]
    return {"total": total, "anomalous": anomalous, "normal": total - anomalous, "rate": 100 * anomalous / total if total else 0.0,
            "bucket_seconds": bucket, "series": series, "histogram": histogram, "top_events": top}


def traffic_summary(connection):
    """Network traffic from block-copy lines: bytes over time and the busiest source machines."""
    total, count, first, last = connection.execute("SELECT COALESCE(SUM(bytes), 0), COUNT(*), MIN(t), MAX(t) FROM traffic").fetchone()
    bucket = next((b for b in BUCKETS if ((last - first) if count else 0) / b <= 60), BUCKETS[-1])
    series = [{"t": t, "bytes": size} for t, size in connection.execute(
        "SELECT (t / ?) * ? AS b, SUM(bytes) FROM traffic GROUP BY b ORDER BY b", (bucket, bucket))]
    sources = [{"ip": ip, "bytes": size, "transfers": n} for ip, size, n in connection.execute(
        "SELECT src, SUM(bytes) AS s, COUNT(*) FROM traffic GROUP BY src ORDER BY s DESC LIMIT 8")]
    return {"total_bytes": total, "transfers": count, "bucket_seconds": bucket, "series": series, "top_sources": sources}


def ip_summary(connection, window=60):
    """Every IP seen in the log. "Active" = it appeared within ``window`` seconds of the newest log line (log time, not wall clock)."""
    (newest,) = connection.execute("SELECT MAX(t) FROM ip_activity").fetchone()
    rows = connection.execute(
        "SELECT ip, SUM(n), MIN(t), MAX(t), (SELECT COALESCE(SUM(bytes), 0) FROM traffic WHERE src = ip_activity.ip) "
        "FROM ip_activity GROUP BY ip ORDER BY MAX(t) DESC, SUM(n) DESC LIMIT 200").fetchall()
    ips = [{"ip": ip, "lines": n, "first": first, "last": last, "bytes_sent": sent, "active": newest - last <= window}
           for ip, n, first, last, sent in rows]
    return {"newest": newest, "window": window, "active_count": sum(i["active"] for i in ips), "ips": ips}


IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def search_log(path, connection, ip="", start=None, end=None, limit=1000):
    """Every raw line mentioning ``ip`` (whole address, so 10.1.1.1 never matches 10.1.1.10) inside [start, end] (UTC epoch seconds).

    Also lists the blocks those lines belong to, with their saved anomaly scores.
    ponytail: re-reads the whole file on each search; index lines by IP if logs get huge.
    """
    pattern = re.compile(rf"(?<![\d.]){re.escape(ip)}(?!\d)") if ip else None
    lines, per_block, total = [], Counter(), 0
    with Path(path).open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if pattern and not pattern.search(line):
                continue
            if start is not None or end is not None:
                header = HEADER.match(line)
                if not header:
                    continue
                when = _epoch(header.group(1), header.group(2))
                if (start is not None and when < start) or (end is not None and when > end):
                    continue
            total += 1
            if len(lines) < limit:
                lines.append(line.rstrip("\n"))
            if block := BLOCK.search(line):
                per_block[block.group(1)] += 1
    scores, ids = {}, list(per_block)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        scores.update(connection.execute(
            f"SELECT block_id, anomaly_score FROM blocks WHERE block_id IN ({','.join('?' * len(chunk))})", chunk))
    blocks = sorted(({"block_id": b, "lines": n, "score": scores.get(b)} for b, n in per_block.items()),
                    key=lambda row: (row["score"] is None, -(row["score"] or 0)))
    return {"total_lines": total, "truncated": total > len(lines), "lines": lines, "total_blocks": len(blocks), "blocks": blocks[:200]}
