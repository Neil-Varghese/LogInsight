"""Local development server for the LogInsight frontend."""

from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import sys
import threading
import uuid
from urllib.parse import parse_qs, unquote, urlparse

from predictor import predict_log_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_SETS_DIR = PROJECT_ROOT / "test sets"
JOBS = {}
JOBS_LOCK = threading.Lock()
TERMINAL_OUTPUT = sys.__stdout__
RUNS_DIR = Path(__file__).resolve().parent / "runs"


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

    database_path = RUNS_DIR / f"{job_id}.sqlite3"
    RUNS_DIR.mkdir(exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("""CREATE TABLE blocks (
        block_id TEXT PRIMARY KEY, anomaly_score REAL NOT NULL, is_anomalous INTEGER NOT NULL,
        sequence_len INTEGER NOT NULL, event_ids TEXT NOT NULL)""")
    saved_count = 0

    def save_block(**block):
        nonlocal saved_count
        connection.execute(
            "INSERT INTO blocks VALUES (?, ?, ?, ?, ?)",
            (block["block_id"], block["anomaly_score"], int(block["is_anomalous"]),
             len(block["event_ids"]), json.dumps(block["event_ids"])),
        )
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


class LogInsightHandler(SimpleHTTPRequestHandler):
    """Serve project files and expose the contents of ``test sets`` as JSON."""

    def do_GET(self):
        request_path = urlparse(self.path).path
        if request_path == "/api/test-sets":
            self.send_test_sets()
            return
        path_parts = request_path.strip("/").split("/")
        if len(path_parts) >= 4 and path_parts[:2] == ["api", "jobs"] and path_parts[3] == "blocks":
            if len(path_parts) == 4:
                self.send_blocks(path_parts[2], parse_qs(urlparse(self.path).query))
            elif len(path_parts) == 5:
                self.send_block_detail(path_parts[2], unquote(path_parts[4]))
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found.")
            return
        if request_path.startswith("/api/jobs/"):
            self.send_job(request_path.rsplit("/", 1)[-1])
            return
        if request_path == "/":
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/frontend/")
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format, *args):
        """Polling is expected, so do not flood the terminal with GET logs."""
        return

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found.")
            return
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
                JOBS[job_id] = {"stage": "preprocessing", "progress": 0}
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
            if filter_value == "anomalous": where.append("is_anomalous = 1")
            if filter_value == "normal": where.append("is_anomalous = 0")
            clause = " WHERE " + " AND ".join(where)
            total = connection.execute("SELECT COUNT(*) FROM blocks" + clause, parameters).fetchone()[0]
            rows = connection.execute(
                "SELECT block_id, anomaly_score, is_anomalous, sequence_len, event_ids FROM blocks" + clause +
                " ORDER BY anomaly_score DESC LIMIT ? OFFSET ?", parameters + [limit, (page - 1) * limit]
            ).fetchall()
            blocks = [{"block_id": row[0], "anomaly_score": row[1], "is_anomalous": bool(row[2]),
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
            self.send_json(HTTPStatus.OK, {"block_id": row[0], "anomaly_score": row[1], "is_anomalous": bool(row[2]), "sequence_len": row[3], "event_ids": json.loads(row[4]), "raw_logs": []})
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
    server = ThreadingHTTPServer(("", 8000), LogInsightHandler)
    print("LogInsight is running at http://localhost:8000/frontend/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
