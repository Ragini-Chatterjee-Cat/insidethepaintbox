"""Postgres persistence for user prefs.
Falls back to JSON files on disk when POSTGRES_URI is not set (local dev)."""

import json
import logging
import os
from pathlib import Path

_conn = None


def _get_conn():
    global _conn
    uri = os.environ.get("POSTGRES_URI")
    if not uri:
        return None

    if _conn is not None:
        try:
            _conn.execute("SELECT 1")
            return _conn
        except Exception:
            _conn = None

    try:
        import psycopg
        _conn = psycopg.connect(uri, autocommit=True)
        return _conn
    except Exception as e:
        logging.warning(f"DB connection failed: {e}")
        return None


def setup_tables():
    conn = _get_conn()
    if not conn:
        return
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_prefs (
            thread_id TEXT PRIMARY KEY,
            prefs JSONB NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)


def load_user_prefs(thread_id: str, prefs_dir: Path) -> dict:
    conn = _get_conn()
    if conn:
        try:
            row = conn.execute(
                "SELECT prefs FROM user_prefs WHERE thread_id = %s", (thread_id,)
            ).fetchone()
            return row[0] if row else {}
        except Exception as e:
            logging.warning(f"load_user_prefs DB error: {e}")
    f = prefs_dir / f"{thread_id}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_user_prefs(thread_id: str, prefs: dict, prefs_dir: Path):
    conn = _get_conn()
    if conn:
        try:
            conn.execute("""
                INSERT INTO user_prefs (thread_id, prefs, updated_at)
                VALUES (%s, %s::jsonb, NOW())
                ON CONFLICT (thread_id) DO UPDATE
                    SET prefs = EXCLUDED.prefs, updated_at = NOW()
            """, (thread_id, json.dumps(prefs)))
            return
        except Exception as e:
            logging.warning(f"save_user_prefs DB error: {e}")
    (prefs_dir / f"{thread_id}.json").write_text(json.dumps(prefs, indent=2))


