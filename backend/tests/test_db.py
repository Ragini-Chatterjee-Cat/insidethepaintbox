"""Tests for db.py - Postgres persistence with a JSON-file fallback."""
import json

import db


def test_get_conn_returns_none_without_postgres_uri(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    db._conn = None
    assert db._get_conn() is None


def test_load_user_prefs_falls_back_to_json_when_no_postgres(tmp_path, monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    db._conn = None

    # No file yet - empty dict
    assert db.load_user_prefs("thread-1", tmp_path) == {}

    # Write directly, then read back through the fallback path
    (tmp_path / "thread-1.json").write_text(json.dumps({"tone": "curious"}))
    assert db.load_user_prefs("thread-1", tmp_path) == {"tone": "curious"}


def test_save_user_prefs_falls_back_to_json_when_no_postgres(tmp_path, monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    db._conn = None

    db.save_user_prefs("thread-2", {"liked_series": ["Mythical"]}, tmp_path)

    saved = json.loads((tmp_path / "thread-2.json").read_text())
    assert saved == {"liked_series": ["Mythical"]}


def test_setup_tables_is_a_noop_without_postgres(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    db._conn = None
    # Should not raise even though there's no real database to connect to.
    db.setup_tables()


def test_get_conn_reuses_live_connection(monkeypatch):
    """A cached connection that still answers SELECT 1 should be reused,
    not reconnected."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://fake/uri")

    class FakeConn:
        def execute(self, *_args, **_kwargs):
            return None

    fake_conn = FakeConn()
    db._conn = fake_conn

    assert db._get_conn() is fake_conn


def test_get_conn_reconnects_when_cached_connection_is_dead(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://fake/uri")

    class DeadConn:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("connection closed")

    db._conn = DeadConn()

    class FakePsycopg:
        @staticmethod
        def connect(*_args, **_kwargs):
            return "a-new-connection"

    import sys
    monkeypatch.setitem(sys.modules, "psycopg", FakePsycopg())

    assert db._get_conn() == "a-new-connection"
