"""Local development server for the LogInsight frontend."""

from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import parse_qs, unquote, urlparse

from auth import Auth, AuthError, SESSION_SECONDS
from explainer import explain
from monitor import IPV4, Monitor, ip_summary, search_log, summary, traffic_summary
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


SIM = {"proc": None, "mode": None, "started": None}  # the one simulator process this server started
SIM_LOCK = threading.Lock()
AUTH = Auth()
ALLOW_SIGNUP = os.environ.get("LOGINSIGHT_ALLOW_SIGNUP", "1") != "0"  # set to 0 once your accounts exist
MAX_BODY = 64 * 1024
COOKIE = "loginsight_session"


def sim_running():
    return SIM["proc"] is not None and SIM["proc"].poll() is None


def sim_info():
    """Simulator state, its settings from docker-compose.yml, and the newest lines in whatever file the monitor is watching."""
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    setting = lambda name: (re.search(rf'{name}:\s*"?([\d.]+)', compose) or [None, None])[1]
    log, tail = MONITOR.log_path, []
    if log.exists():
        with log.open("rb") as log_file:
            log_file.seek(max(0, log.stat().st_size - 8192))
            tail = log_file.read().decode("utf-8", "replace").splitlines()[-15:]
    proc = SIM["proc"]
    return {"containers": setting("scale"), "lines_per_second": setting("LINES_PER_SECOND"), "anomaly_rate": setting("ANOMALY_RATE"),
            "running": sim_running(), "mode": SIM["mode"], "elapsed_s": time.time() - SIM["started"] if SIM["started"] else None,
            "exit_code": proc.poll() if proc else None, "file": log.name, "tail": tail}


def sim_start(mode):
    """Start the replay (no Docker needed) or the Docker cluster, and point the monitor at its output. Returns (status, body)."""
    with SIM_LOCK:
        if sim_running():
            return HTTPStatus.CONFLICT, {"error": "A simulation is already running. Stop it first."}
        if mode == "docker":
            try:
                subprocess.run(["docker", "info"], capture_output=True, timeout=20, check=True)
            except (OSError, subprocess.SubprocessError):
                return HTTPStatus.BAD_REQUEST, {"error": "Docker Desktop is not running. Start it and try again, or use the Replay source."}
            target, command = PROJECT_ROOT / "sim" / "cluster.log", [sys.executable, str(PROJECT_ROOT / "sim" / "collect.py")]
        elif mode == "replay":
            target, command = TEST_SETS_DIR / "live_demo.log", [sys.executable, str(PROJECT_ROOT / "backend" / "simulate_live.py")]
        else:
            return HTTPStatus.BAD_REQUEST, {"error": "Source must be replay or docker."}
        MONITOR.watch(target)
        SIM.update(proc=subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), mode=mode, started=time.time())
    return HTTPStatus.OK, sim_info()


def sim_stop():
    with SIM_LOCK:
        if sim_running():
            SIM["proc"].terminate()
            if SIM["mode"] == "docker":  # collect.py cannot clean up when killed, so take the containers down here
                try:
                    subprocess.run(["docker", "compose", "down"], cwd=PROJECT_ROOT, capture_output=True, timeout=90)
                except (OSError, subprocess.SubprocessError):
                    pass
    return sim_info()


def parse_threshold(query):
    """Anomaly cut-off from ?threshold=0..1; anything unusable falls back to the model's 0.5."""
    try:
        return min(1.0, max(0.0, float(query.get("threshold", ["0.5"])[0])))
    except ValueError:
        return 0.5


def public_stats():
    """Headline numbers for the landing page: the biggest analysis so far (most blocks scored) across the live monitor and every saved run."""
    best = {"total": 0, "anomalous": 0}
    for path in [MONITOR.db_path, *RUNS_DIR.glob("*.sqlite3")]:
        try:  # ponytail: opens every run file per request; cache by file mtime if the history grows to hundreds
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                total, anomalous = connection.execute("SELECT COUNT(*), COALESCE(SUM(anomaly_score >= 0.5), 0) FROM blocks").fetchone()
                if total > best["total"]:
                    best = {"total": total, "anomalous": anomalous}
            finally:
                connection.close()
        except sqlite3.Error:
            continue  # a file that is mid-creation or has no blocks table
    best["rate"] = 100 * best["anomalous"] / best["total"] if best["total"] else 0.0
    best["model_loaded"] = bool(MONITOR.info.get("model_loaded"))
    return best


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

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def session_token(self):
        for part in self.headers.get("Cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE:
                return value
        return None

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= MAX_BODY:
            raise ValueError("Request body too large.")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def api_allowed(self, request_path):
        """Every /api/ route except the auth ones needs a signed-in session; answer 401 and return False otherwise."""
        if not request_path.startswith("/api/") or request_path.startswith(("/api/auth/", "/api/public/")) or AUTH.user(self.session_token()):
            return True
        self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "Sign in required."})
        return False

    def same_origin(self):
        """Browsers send Origin on POSTs; refuse ones from another site (CSRF). Scripts without an Origin header pass."""
        origin = self.headers.get("Origin")
        return origin is None or urlparse(origin).netloc == self.headers.get("Host")

    def handle_auth(self, request_path):
        """POST /api/auth/signup|login|logout. Returns True when it answered."""
        action = request_path.rsplit("/", 1)[-1]
        try:
            if action == "logout":
                AUTH.logout(self.session_token())
                self.send_json(HTTPStatus.OK, {"ok": True}, f"{COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0")
                return
            body = self.read_json()
            email, password = body.get("email"), body.get("password")
            if action == "signup":
                if not ALLOW_SIGNUP:
                    raise AuthError(403, "Sign-up is turned off on this server.")
                AUTH.signup(email, password)
            token = AUTH.login(email, password, self.client_address[0])
            # ponytail: no Secure flag because this serves plain http on localhost; add it behind https
            cookie = f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_SECONDS}"
            self.send_json(HTTPStatus.OK, {"email": email.strip().lower()}, cookie)
        except AuthError as error:
            self.send_json(error.status, {"error": str(error)})
        except (ValueError, AttributeError):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Bad request."})

    def do_GET(self):
        request_path = urlparse(self.path).path
        if request_path == "/api/auth/me":
            user = AUTH.user(self.session_token())
            self.send_json(HTTPStatus.OK if user else HTTPStatus.UNAUTHORIZED, user or {"error": "Not signed in."})
            return
        if not self.api_allowed(request_path):
            return
        if request_path == "/api/public/stats":  # the landing page's numbers; totals only, no log content
            self.send_json(HTTPStatus.OK, public_stats())
            return
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
        if request_path == "/api/monitor/scores":
            # Every block as [id, score, events, last_seen] so the dashboard can apply the threshold itself, with no round trip per slider move.
            connection = sqlite3.connect(MONITOR.db_path)
            try:  # ponytail: capped at 50k blocks (~3 MB of JSON); switch to a delta feed past that
                rows = connection.execute("SELECT block_id, anomaly_score, sequence_len, last_seen FROM blocks ORDER BY anomaly_score DESC LIMIT 50000").fetchall()
                self.send_json(HTTPStatus.OK, {"scores": rows})
            finally:
                connection.close()
            return
        if request_path == "/api/monitor/traffic":
            connection = sqlite3.connect(MONITOR.db_path)
            try:
                self.send_json(HTTPStatus.OK, traffic_summary(connection))
            finally:
                connection.close()
            return
        if request_path == "/api/monitor/feed":
            try:
                after = int(parse_qs(urlparse(self.path).query).get("after", ["-1"])[0])
            except ValueError:
                after = -1
            self.send_json(HTTPStatus.OK, MONITOR.feed_since(after))
            return
        if request_path == "/api/monitor/ips":
            connection = sqlite3.connect(MONITOR.db_path)
            try:
                self.send_json(HTTPStatus.OK, ip_summary(connection))
            finally:
                connection.close()
            return
        if request_path == "/api/monitor/search":
            self.send_search(parse_qs(urlparse(self.path).query))
            return
        if request_path == "/api/sim/info":
            self.send_json(HTTPStatus.OK, sim_info())
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
        if not self.same_origin():
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "Cross-site request refused."})
            return
        if request_path in ("/api/auth/signup", "/api/auth/login", "/api/auth/logout"):
            self.handle_auth(request_path)
            return
        if not self.api_allowed(request_path):
            return
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
        if request_path == "/api/sim/start":
            try:
                mode = self.read_json().get("mode")
            except (ValueError, AttributeError):
                mode = None
            status, body = sim_start(mode)
            self.send_json(status, body)
            return
        if request_path == "/api/sim/stop":
            self.send_json(HTTPStatus.OK, sim_stop())
            return
        if request_path not in ("/api/run", "/api/live"):
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found.")
            return
        live = request_path == "/api/live"
        try:
            payload = self.read_json()
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

    def send_search(self, query):
        ip = query.get("ip", [""])[0].strip()
        try:
            start, end = (int(query[k][0]) if query.get(k, [""])[0] else None for k in ("from", "to"))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Times must be whole epoch seconds."})
            return
        if ip and not IPV4.match(ip):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Enter a full IPv4 address, e.g. 10.250.19.102."})
            return
        if not ip and start is None and end is None:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Give an IP address, a time range, or both."})
            return
        connection = sqlite3.connect(MONITOR.db_path)
        try:
            self.send_json(HTTPStatus.OK, search_log(MONITOR.log_path, connection, ip, start, end))
        finally:
            connection.close()

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

    def send_json(self, status, payload, cookie=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if cookie:
            self.send_header("Set-Cookie", cookie)
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
