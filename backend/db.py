"""
Postgres persistence for visitor preferences (liked series, mentioned
artworks, tone), with a JSON-file fallback when Postgres isn't configured.

Purpose: this is the only place that reads or writes the `user_prefs`
table (or its per-thread JSON file equivalent) — LangGraph's own message
history is persisted separately, by its Postgres/Memory checkpointer in
agent/graph.py, not by this module.

Imported by: app.py (setup_tables(), called once at startup) and
agent/nodes.py (load_user_prefs()/save_user_prefs(), called on every
graph run by the load_preferences and save_preferences nodes).
"""

import json
import logging
import os
from pathlib import Path

_conn = None  # module-level cache: one connection reused across calls


# --- connection handling -----------------------------------------------

def _get_conn():
    """Return a live Postgres connection, or None if POSTGRES_URI isn't
    set or the connection can't be made. Reuses the cached connection
    when it's still alive; reconnects once if it's gone stale."""
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
    """Create the user_prefs table if it doesn't exist yet. No-op when
    Postgres isn't configured (there's no table to create for the JSON
    fallback — each thread just gets its own file on demand)."""
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


# --- reading and writing preferences ------------------------------------

def load_user_prefs(thread_id: str, prefs_dir: Path) -> dict:
    """Return this thread's saved preferences dict, or {} if none exist
    yet. Tries Postgres first, falls back to prefs_dir/<thread_id>.json."""
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
    """Persist this thread's preferences dict. Upserts into Postgres if
    configured, otherwise overwrites prefs_dir/<thread_id>.json."""
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


