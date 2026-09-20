"""Tests for rag.py - embedding retry logic and content-change detection."""
from unittest.mock import MagicMock

import pytest
from voyageai.error import AuthenticationError, RateLimitError

import rag


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retry backoff would otherwise make these tests slow for no reason."""
    monkeypatch.setattr(rag.time, "sleep", lambda _seconds: None)


def test_voyage_embed_retries_transient_errors_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("rate limited")
        return MagicMock(embeddings=[[0.1, 0.2]])

    monkeypatch.setattr(rag._voyage_client, "multimodal_embed", flaky)

    result = rag._voyage_embed([["hello"]], input_type="query")

    assert calls["n"] == 3
    assert result.embeddings == [[0.1, 0.2]]


def test_voyage_embed_does_not_retry_auth_errors(monkeypatch):
    calls = {"n": 0}

    def bad_auth(*_args, **_kwargs):
        calls["n"] += 1
        raise AuthenticationError("bad key")

    monkeypatch.setattr(rag._voyage_client, "multimodal_embed", bad_auth)

    with pytest.raises(AuthenticationError):
        rag._voyage_embed([["hello"]], input_type="query")

    assert calls["n"] == 1  # failed immediately, no wasted retries


def test_voyage_embed_raises_after_exhausting_retries(monkeypatch):
    calls = {"n": 0}

    def always_limited(*_args, **_kwargs):
        calls["n"] += 1
        raise RateLimitError("still limited")

    monkeypatch.setattr(rag._voyage_client, "multimodal_embed", always_limited)

    with pytest.raises(RateLimitError):
        rag._voyage_embed([["hello"]], input_type="query", max_attempts=3)

    assert calls["n"] == 3


def test_embed_document_skips_images_that_fail_to_open(monkeypatch):
    """A missing/corrupt image shouldn't prevent the document from being
    embedded at all - it should just fall back to text-only."""
    captured = {}

    def fake_embed(inputs, input_type):
        captured["inputs"] = inputs
        return MagicMock(embeddings=[[0.5]])

    monkeypatch.setattr(rag, "_voyage_embed", fake_embed)

    rag.embed_document("some text", image_paths=["/does/not/exist.jpg"])

    # Only the text made it into the embed call - the bad image was skipped.
    assert captured["inputs"] == [["some text"]]


def test_embed_document_catches_corruption_past_the_header(monkeypatch, tmp_path):
    """Regression test: Image.open() only reads the header lazily, so a
    file corrupted past its header used to pass Image.open() without
    error and only fail later inside the Voyage SDK call, outside this
    function's try/except - causing index_documents' outer except to drop
    the ENTIRE document, not just the bad image. Forcing .load() here
    catches it in the right place."""
    bad_image = tmp_path / "corrupt.jpg"
    bad_image.write_bytes(b"not a real image")

    captured = {}

    def fake_embed(inputs, input_type):
        captured["inputs"] = inputs
        return MagicMock(embeddings=[[0.5]])

    monkeypatch.setattr(rag, "_voyage_embed", fake_embed)

    rag.embed_document("some text", image_paths=[str(bad_image)])

    assert captured["inputs"] == [["some text"]]  # corrupt image was skipped, not fatal


def test_content_signature_changes_with_content():
    docs_a = [{"source": "a.html", "content": "hello", "image_paths": []}]
    docs_b = [{"source": "a.html", "content": "goodbye", "image_paths": []}]

    assert rag._content_signature(docs_a) != rag._content_signature(docs_b)


def test_content_signature_is_stable_for_same_content():
    docs = [{"source": "a.html", "content": "hello", "image_paths": []}]
    assert rag._content_signature(docs) == rag._content_signature(docs)


def test_content_signature_changes_with_secret_flag():
    """Regression test: toggling an artwork's secret status alone (no
    other content change) used to produce an identical signature, so a
    reindex was skipped and the stale secret metadata stuck around."""
    public_doc = [{"source": "a.html", "content": "hello", "image_paths": [], "secret": False}]
    secret_doc = [{"source": "a.html", "content": "hello", "image_paths": [], "secret": True}]

    assert rag._content_signature(public_doc) != rag._content_signature(secret_doc)


def test_content_signature_changes_with_embedding_model(monkeypatch):
    docs = [{"source": "a.html", "content": "hello", "image_paths": []}]
    sig_before = rag._content_signature(docs)

    monkeypatch.setattr(rag, "EMBEDDING_MODEL_NAME", "a-different-model")
    sig_after = rag._content_signature(docs)

    assert sig_before != sig_after


def test_index_documents_skips_reindex_when_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_SIGNATURE_PATH", tmp_path / ".content_signature")
    monkeypatch.setattr(rag.collection, "count", lambda: 5)

    docs = [{"source": "a.html", "content": "hello", "image_paths": []}]
    signature = rag._content_signature(docs)
    (tmp_path / ".content_signature").write_text(signature)

    embed_calls = {"n": 0}
    monkeypatch.setattr(
        rag, "embed_document",
        lambda *a, **k: embed_calls.__setitem__("n", embed_calls["n"] + 1) or [0.1],
    )

    result = rag.index_documents(docs)

    assert result == 5  # returned the existing count, not a fresh index
    assert embed_calls["n"] == 0  # never called Voyage at all


def test_index_documents_does_not_wipe_collection_when_every_embed_fails(tmp_path, monkeypatch):
    """Regression test: index_documents used to delete the whole existing
    collection up front and rebuild it document-by-document. If Voyage
    was down for the whole run, the old (complete) index was already gone
    before the replacement failed to fill back in. Embedding now happens
    into a staging list BEFORE anything is deleted."""
    monkeypatch.setattr(rag, "_SIGNATURE_PATH", tmp_path / ".content_signature")
    monkeypatch.setattr(rag.collection, "count", lambda: 5)
    (tmp_path / ".content_signature").write_text("a-stale-signature")

    def always_fails(*_a, **_k):
        raise RuntimeError("Voyage is down")

    monkeypatch.setattr(rag, "embed_document", always_fails)
    delete_calls = []
    monkeypatch.setattr(rag.collection, "delete", lambda **k: delete_calls.append(k))

    docs = [{"source": "a.html", "content": "hello", "image_paths": [], "title": "A"}]
    result = rag.index_documents(docs)

    assert result == 5  # kept reporting the existing (untouched) count
    assert delete_calls == []  # never even attempted to wipe the collection


def test_index_documents_reindexes_when_content_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_SIGNATURE_PATH", tmp_path / ".content_signature")
    monkeypatch.setattr(rag.collection, "count", lambda: 5)
    monkeypatch.setattr(rag.collection, "get", lambda: {"ids": []})
    (tmp_path / ".content_signature").write_text("a-stale-signature")

    added = []
    monkeypatch.setattr(rag.collection, "add", lambda **kwargs: added.append(kwargs))
    monkeypatch.setattr(rag, "embed_document", lambda *a, **k: [0.1])

    docs = [{"source": "a.html", "content": "hello", "image_paths": [], "title": "A"}]
    result = rag.index_documents(docs)

    assert result == 1
    assert len(added) == 1
    # A fresh signature was recorded for next time.
    assert (tmp_path / ".content_signature").read_text() == rag._content_signature(docs)
