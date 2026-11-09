"""Tests for app.py - rate limits on /chat/v2."""
import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture
def client(monkeypatch):
    """A client for the real app with the LLM side stubbed out. Not used
    as a context manager, so startup indexing (lifespan) never runs."""
    monkeypatch.setattr(app_module, "get_conversation_history", lambda _thread_id: [])
    monkeypatch.setattr(app_module, "chat_with_memory", lambda _message, _thread_id, _visitor_id: "hi!")
    app_module.limiter.reset()
    yield TestClient(app_module.app)
    app_module.limiter.reset()


def _chat(client, ip, thread_id="t1"):
    return client.post(
        "/chat/v2",
        json={"message": "hello", "thread_id": thread_id},
        headers={"X-Forwarded-For": ip},
    )


def test_client_ip_is_leftmost_forwarded_for_entry(client):
    """Railway's edge puts the real visitor IP first; anything after it is
    another proxy hop (e.g. Fastly), which must not be used as the key."""
    for _ in range(15):
        assert _chat(client, "1.1.1.1, 151.101.0.1").status_code == 200
    # Same visitor via a different edge hop is still the same visitor...
    assert _chat(client, "1.1.1.1, 151.101.9.9").status_code == 429
    # ...and a different visitor behind the same edge hop is not.
    assert _chat(client, "2.2.2.2, 151.101.0.1").status_code == 200


def test_per_ip_limit_cannot_be_dodged_with_new_thread_ids(client):
    """Regression test: the 20-turn cap is per thread_id, which the chat's
    "new conversation" button resets. The per-IP limit must not be."""
    for i in range(15):
        assert _chat(client, "1.1.1.1", thread_id=f"fresh-{i}").status_code == 200
    assert _chat(client, "1.1.1.1", thread_id="fresh-again").status_code == 429


def test_global_daily_cap_applies_across_all_visitors(client):
    """No single IP gets near its own limit, but the site-wide bucket
    still runs out - bounding spend even against many IPs."""
    for i in range(500):
        assert _chat(client, f"10.0.{i // 250}.{i % 250}").status_code == 200
    assert _chat(client, "9.9.9.9").status_code == 429


def test_chat_passes_visitor_id_through_for_preferences(client, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "chat_with_memory", lambda *args: calls.append(args) or "hi!")

    response = client.post(
        "/chat/v2", json={"message": "hello", "thread_id": "thread-2", "visitor_id": "visitor-1"}
    )

    assert response.status_code == 200
    assert calls == [("hello", "thread-2", "visitor-1")]


def test_chat_falls_back_to_thread_id_without_visitor_id(client, monkeypatch):
    """A visitor still running an older cached chat.js sends no visitor_id;
    their preferences stay keyed by thread_id, as before."""
    calls = []
    monkeypatch.setattr(app_module, "chat_with_memory", lambda *args: calls.append(args) or "hi!")

    client.post("/chat/v2", json={"message": "hello", "thread_id": "thread-2"})

    assert calls == [("hello", "thread-2", "thread-2")]


def test_chat_rejects_visitor_ids_that_could_escape_the_prefs_dir(client):
    """visitor_id becomes a filename in db.py's JSON fallback."""
    for bad in ("../../app", "a/b", "x" * 65):
        response = client.post(
            "/chat/v2", json={"message": "hello", "thread_id": "t1", "visitor_id": bad}
        )
        assert response.status_code == 422
