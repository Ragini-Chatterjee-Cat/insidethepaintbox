"""Postgres persistence for user prefs and commissions.
Falls back to JSON files on disk when POSTGRES_URI is not set (local dev)."""

import json
import logging
import os
from datetime import datetime
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS commissions (
            thread_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS commission_replies (
            thread_id TEXT PRIMARY KEY,
            message TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
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


def save_commission(thread_id: str, summary: str, timestamp: str, prefs_dir: Path):
    conn = _get_conn()
    if conn:
        try:
            conn.execute("""
                INSERT INTO commissions (thread_id, summary, created_at)
                VALUES (%s, %s, %s::timestamptz)
                ON CONFLICT (thread_id) DO UPDATE
                    SET summary = EXCLUDED.summary
            """, (thread_id, summary, timestamp))
            return
        except Exception as e:
            logging.warning(f"save_commission DB error: {e}")
    (prefs_dir / f"{thread_id}_commission.json").write_text(json.dumps({
        "thread_id": thread_id,
        "summary": summary,
        "timestamp": timestamp,
    }))


def get_pending_commissions(prefs_dir: Path) -> list:
    conn = _get_conn()
    if conn:
        try:
            rows = conn.execute("""
                SELECT c.thread_id, c.summary, c.created_at
                FROM commissions c
                LEFT JOIN commission_replies r ON c.thread_id = r.thread_id
                WHERE r.thread_id IS NULL
                ORDER BY c.created_at
            """).fetchall()
            return [
                {"thread_id": r[0], "summary": r[1], "timestamp": r[2].isoformat()}
                for r in rows
            ]
        except Exception as e:
            logging.warning(f"get_pending_commissions DB error: {e}")
    commissions = []
    for f in sorted(prefs_dir.glob("*_commission.json")):
        try:
            data = json.loads(f.read_text())
            if not (prefs_dir / f"{data['thread_id']}_reply.json").exists():
                commissions.append(data)
        except Exception:
            continue
    return commissions


def save_commission_reply(thread_id: str, message: str, prefs_dir: Path):
    conn = _get_conn()
    if conn:
        try:
            conn.execute("""
                INSERT INTO commission_replies (thread_id, message, created_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (thread_id) DO UPDATE
                    SET message = EXCLUDED.message, created_at = NOW()
            """, (thread_id, message))
            return
        except Exception as e:
            logging.warning(f"save_commission_reply DB error: {e}")
    (prefs_dir / f"{thread_id}_reply.json").write_text(json.dumps({
        "thread_id": thread_id,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }))


def get_commission_reply(thread_id: str, prefs_dir: Path) -> dict | None:
    conn = _get_conn()
    if conn:
        try:
            row = conn.execute(
                "SELECT message FROM commission_replies WHERE thread_id = %s", (thread_id,)
            ).fetchone()
            return {"message": row[0]} if row else None
        except Exception as e:
            logging.warning(f"get_commission_reply DB error: {e}")
    f = prefs_dir / f"{thread_id}_reply.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return None
