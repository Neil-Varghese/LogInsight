"""Local development server for the LogInsight frontend."""

from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import uuid
from urllib.parse import parse_qs, unquote, urlparse

from explainer import explain
from monitor import Monitor, summary
from predictor import predict_log_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_SETS_DIR = PROJECT_ROOT / "test sets"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard" / "out"  # built with: cd dashboard && npm run build
FRONTEND_DIR = PROJECT_ROOT / "frontend"  # the old plain-JS page, kept at /frontend/
JOBS = {}
JOBS_LOCK = threading.Lock()
TERMINAL_OUTPUT = sys.__stdout__
RUNS_DIR = Path(__file__).resolve().parent / "runs"
LIVE_STOPS = {}  # job_id -> threading.Event that ends that live watcher
LIVE_POLL_SECONDS = 3
RAW_LINE_LIMIT = 50  # same cap as the LSTM's 50 events per block
MONITOR = Monitor(PROJECT_ROOT / os.environ.get("LOGINSIGHT_WATCH", "test sets/live_demo.log"), Path(__file__).resolve().parent / "monitor.sqlite3")


def parse_threshold(query):
    """Anomaly cut-off from ?threshold=0..1; anything unusable falls back to the model's 0.5."""
    try:
        return min(1.0, max(0.0, float(query.get("threshold", ["0.5"])[0])))
    except ValueError:
        return 0.5


def create_run_database(job_id):
    RUNS_DIR.mkdir(exist_ok=True)
    database_path = RUNS_DIR / f"{job_id}.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("""CREATE TABLE blocks (
        block_id TEXT PRIMARY KEY, anomaly_score REAL NOT NULL, is_anomalous INTEGER NOT NULL,
        sequence_len INTEGER NOT NULL, event_ids TEXT NOT NULL)""")
    # seq_len lets a saved explanation be ignored once a live block has grown.
    connection.execute("CREATE TABLE explanations (block_id TEXT PRIMARY KEY, seq_len INTEGER NOT NULL, body TEXT NOT NULL)")
    return connection, database_path


def save_block_row(connection, block):
    """Insert a scored block, or overwrite it when a live re-score sees more events."""
    connection.execute(
        """INSERT INTO blocks VALUES (?, ?, ?, ?, ?) ON CONFLICT(block_id) DO UPDATE SET
           anomaly_score = excluded.anomaly_score, is_anomalous = excluded.is_anomalous,
           sequence_len = excluded.sequence_len, event_ids = excluded.event_ids""",
        (block["block_id"], block["anomaly_score"], int(block["is_anomalous"]),
         len(block["event_ids"]), json.dumps(block["event_ids"])),
    )


def read_block_lines(file_path, block_id, limit=RAW_LINE_LIMIT):
    """First ``limit`` raw log lines that mention this block."""
    # ponytail: scans the whole file per request; store line offsets per block if files get huge.
    pattern = re.compile(re.escape(block_id) + r"(?!\d)")
    lines = []
    with open(file_path, encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if pattern.search(line):
                lines.append(line.rstrip("\n"))
                if len(lines) == limit:
                    break
    return lines


def run_prediction(job_id, file_path):
    last_stage = None

    def update_progress(stage, progress):
        nonlocal last_stage
        progress = max(0, min(100, float(progress)))
        with JOBS_LOCK:
            JOBS[job_id].update({"stage": stage, "progress": progress})
        if stage != last_stage and last_stage is not None:
            TERMINAL_OUTPUT.write("\n")
        last_stage = stage
        filled = round(progress / 100 * 30)
        bar = "#" * filled + "-" * (30 - filled)
        TERMINAL_OUTPUT.write(f"\r{stage.title():<16} [{bar}] {progress:5.1f}%")
        TERMINAL_OUTPUT.flush()
        if progress == 100:
            TERMINAL_OUTPUT.write("\n")
            TERMINAL_OUTPUT.flush()

    connection, database_path = create_run_database(job_id)
    saved_count = 0

    def save_block(**block):
        nonlocal saved_count
        save_block_row(connection, block)
        saved_count += 1
        if saved_count % 1000 == 0:
            connection.commit()

    try:
        result = predict_log_file(file_path, update_progress, save_block)
        connection.commit()
        with JOBS_LOCK:
            JOBS[job_id].update({"stage": "complete", "progress": 100, "result": result, "database": str(database_path)})
        print(result, flush=True)
    except Exception as error:
        print(f"Prediction failed: {error}", flush=True)
        with JOBS_LOCK:
            JOBS[job_id].update({"stage": "error", "error": str(error)})
    finally:
        connection.close()


def run_live(job_id, file_path, stop):
    """Re-score the whole file whenever it grows, until ``stop`` is set."""
    # ponytail: re-parses the full file each time it changes; fine for demo-sized logs,
    # switch to incremental parsing before pointing it at a multi-GB live feed.
    connection, database_path = create_run_database(job_id)
    with JOBS_LOCK:
        JOBS[job_id]["database"] = str(database_path)
    last_size = -1
    try:
        while not stop.is_set():
            size = file_path.stat().st_size
            if size != last_size:
                last_size = size
                try:
                    result = predict_log_file(file_path, None, lambda **block: save_block_row(connection, block))
                    connection.commit()
                    with JOBS_LOCK:
                        JOBS[job_id].update({"result": result, "warning": None})
                except Exception as error:
                    # A half-written last line is normal while the file is growing; retry next change.
                    print(f"Live scoring skipped: {error}", flush=True)
                    last_size = -1
                    with JOBS_LOCK:
                        JOBS[job_id]["warning"] = str(error)
            stop.wait(LIVE_POLL_SECONDS)
    finally:
        connection.close()
        with JOBS_LOCK:
            JOBS[job_id]["stage"] = "stopped"
        LIVE_STOPS.pop(job_id, None)


class LogInsightHandler(SimpleHTTPRequestHandler):
    """Serve project files and expose the contents of ``test sets`` as JSON."""

    def do_GET(self):
        request_path = urlparse(self.path).path
        if request_path == "/api/test-sets":
            self.send_test_sets()
            return
        if request_path == "/api/monitor/status":
            self.send_json(HTTPStatus.OK, MONITOR.snapshot())
            return
        if request_path == "/api/monitor/summary":
            connection = sqlite3.connect(MONITOR.db_path)
            try:
                self.send_json(HTTPStatus.OK, summary(connection, parse_threshold(parse_qs(urlparse(self.path).query))))
            finally:
                connection.close()
            return
        path_parts = request_path.strip("/").split("/")
        if len(path_parts) >= 4 and path_parts[:2] == ["api", "jobs"] and path_parts[3] == "blocks":
            if len(path_parts) == 4:
                self.send_blocks(path_parts[2], parse_qs(urlparse(self.path).query))
            elif len(path_parts) == 5:
                self.send_block_detail(path_parts[2], unquote(path_parts[4]))
            elif len(path_parts) == 6 and path_parts[5] == "explain":
                self.send_explanation(path_parts[2], unquote(path_parts[4]))
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found.")
            return
        if request_path.startswith("/api/jobs/"):
            self.send_job(request_path.rsplit("/", 1)[-1])
            return
        super().do_GET()

    def translate_path(self, path):
        """Serve only the two page folders, never the project root (it holds .env and the source)."""
        request_path = urlparse(path).path
        if request_path == "/frontend" or request_path.startswith("/frontend/"):
            self.directory, request_path = str(FRONTEND_DIR), request_path[len("/frontend"):]
        else:
            self.directory = str(DASHBOARD_DIR)
        return super().translate_path(request_path)

    def log_message(self, format, *args):
        """Polling is expected, so do not flood the terminal with GET logs."""
        return

    def do_POST(self):
        request_path = urlparse(self.path).path
        parts = request_path.strip("/").split("/")
        if request_path == "/api/monitor/reset":
            MONITOR.reset()
            self.send_json(HTTPStatus.OK, {"reset": True})
            return
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
            stop = LIVE_STOPS.get(parts[2])
            if stop is None:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "No live watcher with that id."})
                return
            stop.set()
            self.send_json(HTTPStatus.OK, {"stopping": True})
            return
        if request_path not in ("/api/run", "/api/live"):
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found.")
            return
        live = request_path == "/api/live"
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            filename = payload["filename"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Invalid test-set filename.")
            file_path = TEST_SETS_DIR / filename
            if not file_path.is_file():
                raise FileNotFoundError("Selected test set no longer exists.")
            job_id = uuid.uuid4().hex
            with JOBS_LOCK:
                JOBS[job_id] = {"stage": "live" if live else "preprocessing", "progress": 0, "file": str(file_path)}
            if live:
                LIVE_STOPS[job_id] = threading.Event()
                threading.Thread(target=run_live, args=(job_id, file_path, LIVE_STOPS[job_id]), daemon=True).start()
            else:
                threading.Thread(target=run_prediction, args=(job_id, file_path), daemon=True).start()
            self.send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except FileNotFoundError as error:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
        except Exception as error:
            print(f"Prediction failed: {error}", flush=True)
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Prediction failed: {error}"})

    def send_test_sets(self):
        files = [
            {"name": entry.name, "path": entry.name}
            for entry in sorted(TEST_SETS_DIR.iterdir(), key=lambda item: item.name.lower())
            if entry.is_file() and not entry.name.startswith(".")
        ]
        self.send_json(HTTPStatus.OK, files)

    def send_job(self, job_id):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            payload = dict(job) if job else None
        if payload is None:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Run not found."})
            return
        self.send_json(HTTPStatus.OK, payload)

    def _open_run_database(self, job_id):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            database = job.get("database") if job else None
        if not database:
            return None
        return sqlite3.connect(database)

    def _raw_lines(self, job_id, block_id):
        with JOBS_LOCK:
            source = JOBS.get(job_id, {}).get("file")
        return read_block_lines(source, block_id) if source else []

    def send_blocks(self, job_id, query):
        connection = self._open_run_database(job_id)
        if connection is None:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Completed run not found."})
            return
        try:
            page = max(1, int(query.get("page", ["1"])[0]))
            limit = min(100, max(1, int(query.get("limit", ["25"])[0])))
            filter_value = query.get("filter", ["all"])[0]
            search = query.get("search", [""])[0]
            where, parameters = ["block_id LIKE ?"], [f"%{search}%"]
            threshold = parse_threshold(query)
            if filter_value == "anomalous": where.append("anomaly_score >= ?"); parameters.append(threshold)
            if filter_value == "normal": where.append("anomaly_score < ?"); parameters.append(threshold)
            clause = " WHERE " + " AND ".join(where)
            total = connection.execute("SELECT COUNT(*) FROM blocks" + clause, parameters).fetchone()[0]
            rows = connection.execute(
                "SELECT block_id, anomaly_score, is_anomalous, sequence_len, event_ids FROM blocks" + clause +
                " ORDER BY anomaly_score DESC LIMIT ? OFFSET ?", parameters + [limit, (page - 1) * limit]
            ).fetchall()
            blocks = [{"block_id": row[0], "anomaly_score": row[1], "is_anomalous": row[1] >= threshold,
                       "sequence_len": row[3], "event_ids": json.loads(row[4])} for row in rows]
            self.send_json(HTTPStatus.OK, {"blocks": blocks, "total": total, "page": page, "limit": limit})
        finally:
            connection.close()

    def send_block_detail(self, job_id, block_id):
        connection = self._open_run_database(job_id)
        if connection is None:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Completed run not found."})
            return
        try:
            row = connection.execute("SELECT block_id, anomaly_score, is_anomalous, sequence_len, event_ids FROM blocks WHERE block_id = ?", (block_id,)).fetchone()
            if row is None:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Block not found."})
                return
            self.send_json(HTTPStatus.OK, {"block_id": row[0], "anomaly_score": row[1], "is_anomalous": bool(row[2]), "sequence_len": row[3], "event_ids": json.loads(row[4]), "raw_logs": self._raw_lines(job_id, block_id)})
        finally:
            connection.close()

    def send_explanation(self, job_id, block_id):
        connection = self._open_run_database(job_id)
        if connection is None:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Completed run not found."})
            return
        try:
            row = connection.execute("SELECT anomaly_score, event_ids, sequence_len FROM blocks WHERE block_id = ?", (block_id,)).fetchone()
            if row is None:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Block not found."})
                return
            # Saved per run so a repeat click costs no API quota; ignored once a live block has grown.
            cached = connection.execute("SELECT body FROM explanations WHERE block_id = ? AND seq_len = ?", (block_id, row[2])).fetchone()
            if cached:
                self.send_json(HTTPStatus.OK, {"explanation": json.loads(cached[0])})
                return
            explanation = explain(block_id, row[0], json.loads(row[1]), self._raw_lines(job_id, block_id))
            connection.execute("INSERT OR REPLACE INTO explanations VALUES (?, ?, ?)", (block_id, row[2], json.dumps(explanation)))
            connection.commit()
            self.send_json(HTTPStatus.OK, {"explanation": explanation})
        except Exception as error:
            print(f"Explanation failed: {error}", flush=True)
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": f"Explanation failed: {error}"})
        finally:
            connection.close()

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    JOBS["monitor"] = {"stage": "monitoring", "progress": 100, "file": str(MONITOR.log_path), "database": str(MONITOR.db_path)}
    MONITOR.start()  # reuses the /api/jobs/monitor/blocks routes for block detail and explanations
    server = ThreadingHTTPServer(("127.0.0.1", 8000), LogInsightHandler)  # this machine only; "" would expose it to the whole network
    print(f"LogInsight is running at http://localhost:8000/  (watching {MONITOR.log_path})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
