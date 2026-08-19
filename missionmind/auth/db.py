"""MissionMind auth — SQLite persistence.

A single small SQLite database (stdlib `sqlite3`) holds users, sessions,
email-verification tokens and password-reset tokens. All token columns store
SHA-256 digests (see security.py), never raw tokens.

Connections are thread-local (FastAPI runs endpoints on a worker pool), WAL
mode allows concurrent readers, and the schema is created idempotently.

The database path is read lazily from MISSIONMIND_DB_PATH on first use so
tests can point it at a temp file before the app is exercised. In production
this must point at a persistent location; see missionmind/docs/SECURITY.md for
the serverless (Vercel) caveat.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
    "missionmind.db")

_local = threading.local()
_db_path: Optional[str] = None
_path_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    salt            TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'user',
    email_verified  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS email_verifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    used        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ev_user ON email_verifications(user_id);
CREATE TABLE IF NOT EXISTS password_resets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    used        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pr_user ON password_resets(user_id);
"""


def db_path() -> str:
    return os.getenv("MISSIONMIND_DB_PATH", _DEFAULT_DB)


def get_conn() -> sqlite3.Connection:
    """Lazy per-thread connection. Re-opens when MISSIONMIND_DB_PATH changes
    (tests swap databases). WAL mode + busy timeout make concurrent access
    safe with the stdlib driver."""
    global _db_path
    path = db_path()
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "path", None) == path:
        return conn
    with _path_lock:
        _db_path = path
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    _local.conn = conn
    _local.path = path
    return conn


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def close() -> None:
    """Close the calling thread's connection (used by tests)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    _local.conn = None
    _local.path = None
