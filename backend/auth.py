"""Accounts and login sessions, kept in their own SQLite file.

Passwords are stored only as salted scrypt hashes. A session token is a random string given to the browser in a cookie;
the database keeps just its SHA-256 hash, so a leaked database cannot be used to sign in.
"""
import contextlib
import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "auth.sqlite3"
SESSION_SECONDS = 7 * 24 * 3600
MAX_FAILS, FAIL_WINDOW = 5, 600  # 5 wrong passwords per address+email in 10 minutes, then wait
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]+$")
SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)


class AuthError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _hash(password, salt):
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, **SCRYPT)


def _token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class Auth:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.fails = {}  # ponytail: in memory, so a restart forgets them; move into SQLite if the server runs on several processes
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, salt BLOB NOT NULL, hash BLOB NOT NULL, created REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL)")

    @contextlib.contextmanager
    def _connect(self):
        """Commit on success, roll back on error, and always close (sqlite3's own `with` does not close, which locks the file on Windows)."""
        db = sqlite3.connect(self.db_path)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _clean(email, password):
        email = (email or "").strip().lower()
        if not isinstance(password, str) or not EMAIL.match(email) or len(email) > 254:
            raise AuthError(400, "Enter a valid email address and a password.")
        return email

    def signup(self, email, password):
        email = self._clean(email, password)
        if not 8 <= len(password) <= 128:
            raise AuthError(400, "Password must be 8 to 128 characters.")
        salt = secrets.token_bytes(16)
        try:
            with self._connect() as db:
                db.execute("INSERT INTO users (email, salt, hash, created) VALUES (?, ?, ?, ?)", (email, salt, _hash(password, salt), time.time()))
        except sqlite3.IntegrityError:
            raise AuthError(409, "An account with that email already exists.") from None

    def login(self, email, password, address):
        """Return a fresh session token, or raise AuthError."""
        email = self._clean(email, password)
        key = f"{address}|{email}"
        now = time.time()
        with self.lock:
            recent = [t for t in self.fails.get(key, []) if now - t < FAIL_WINDOW]
            self.fails[key] = recent
            if len(recent) >= MAX_FAILS:
                raise AuthError(429, "Too many failed attempts. Try again in a few minutes.")
        with self._connect() as db:
            row = db.execute("SELECT id, salt, hash FROM users WHERE email = ?", (email,)).fetchone()
            # Hash even for an unknown email so the response time does not reveal which emails exist.
            candidate = _hash(password, row[1] if row else b"\0" * 16)
            if not row or not hmac.compare_digest(candidate, row[2]):
                with self.lock:
                    self.fails.setdefault(key, []).append(now)
                raise AuthError(401, "Wrong email or password.")
            token = secrets.token_urlsafe(32)
            db.execute("DELETE FROM sessions WHERE expires < ?", (now,))
            db.execute("INSERT INTO sessions VALUES (?, ?, ?)", (_token_hash(token), row[0], now + SESSION_SECONDS))
        with self.lock:
            self.fails.pop(key, None)
        return token

    def user(self, token):
        """The signed-in user for this cookie value, or None."""
        if not token:
            return None
        with self._connect() as db:
            row = db.execute(
                "SELECT users.email FROM sessions JOIN users ON users.id = sessions.user_id WHERE token_hash = ? AND expires > ?",
                (_token_hash(token), time.time()),
            ).fetchone()
        return {"email": row[0]} if row else None

    def logout(self, token):
        if token:
            with self._connect() as db:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
