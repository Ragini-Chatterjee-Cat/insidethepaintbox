"""Tests for the artwork tool classes - patches embed_query/collection at
the point of use in each tool module (not at rag.py, where they're
defined), since `from .base import X` binds each module its own name."""
from langchain_core.tools import ToolException

import tools.filter_by_series as filter_by_series_mod
import tools.get_artwork_details as get_artwork_details_mod
import tools.recommend_similar as recommend_similar_mod
import tools.search_artworks as search_artworks_mod
from tools.filter_by_series import FilterBySeriesTool
from tools.get_artwork_details import GetArtworkDetailsTool
from tools.recommend_similar import RecommendSimilarTool
from tools.search_artworks import SearchArtworksTool


def _fake_query_result(titles_and_distances):
    return {
        "documents": [[f"content about {t}" for t, _ in titles_and_distances]],
        "metadatas": [[{"title": t, "series": "", "url": ""} for t, _ in titles_and_distances]],
        "distances": [[d for _, d in titles_and_distances]],
    }


def test_get_artwork_details_returns_title_match_even_at_high_distance(monkeypatch):
    """A confidently-named match should win regardless of embedding
    distance - the substring check is trusted over raw distance."""
    monkeypatch.setattr(get_artwork_details_mod, "embed_query", lambda _q: [0.0])
    monkeypatch.setattr(
        get_artwork_details_mod.collection, "query",
        lambda **_k: _fake_query_result([("Icarus", 1.9), ("Voices", 0.1)]),
    )

    result = GetArtworkDetailsTool()._run("Icarus")

    assert "Icarus" in result


def test_get_artwork_details_raises_for_unmatched_low_confidence_name(monkeypatch):
    """Regression test: previously this always returned the top embedding
    result no matter how irrelevant, contradicting the bot's rule to admit
    when nothing was found."""
    monkeypatch.setattr(get_artwork_details_mod, "embed_query", lambda _q: [0.0])
    monkeypatch.setattr(
        get_artwork_details_mod.collection, "query",
        lambda **_k: _fake_query_result([("Voices", 1.9), ("Icarus", 1.95)]),
    )

    try:
        GetArtworkDetailsTool()._run("Something Totally Unrelated")
        assert False, "expected a ToolException"
    except ToolException:
        pass


def test_get_artwork_details_accepts_confident_untitled_match(monkeypatch):
    """A close embedding match should still be trusted even without an
    exact title substring match."""
    monkeypatch.setattr(get_artwork_details_mod, "embed_query", lambda _q: [0.0])
    monkeypatch.setattr(
        get_artwork_details_mod.collection, "query",
        lambda **_k: _fake_query_result([("Voices", 0.4), ("Icarus", 1.9)]),
    )

    result = GetArtworkDetailsTool()._run("Vices")  # close typo, no substring match

    assert "Voices" in result


def test_get_artwork_details_excludes_secret_pieces(monkeypatch):
    """Regression test: this query had no secret filter at all, so a
    visitor who already knew a secret artwork's exact name could get its
    full details directly - contradicting reveal_secret.py's own
    docstring claim that it's "the only path back to them"."""
    seen_kwargs = {}

    def fake_query(**kwargs):
        seen_kwargs.update(kwargs)
        return _fake_query_result([("Icarus", 0.2)])

    monkeypatch.setattr(get_artwork_details_mod, "embed_query", lambda _q: [0.0])
    monkeypatch.setattr(get_artwork_details_mod.collection, "query", fake_query)

    GetArtworkDetailsTool()._run("Icarus")

    assert seen_kwargs["where"] == {"secret": False}


def test_recommend_similar_source_lookup_excludes_secret_pieces(monkeypatch):
    """Regression test: only the *second* query (the actual
    recommendations) filtered secrets; the first query, used to look up
    the source artwork, didn't - so a secret piece's real title could
    leak into the response header even though it never appeared in the
    recommended list itself."""
    seen_kwargs = {}

    def fake_source_query(**kwargs):
        seen_kwargs.update(kwargs)
        return _fake_query_result([("Icarus", 0.1)])

    monkeypatch.setattr(recommend_similar_mod, "embed_query", lambda _q: [0.0])
    monkeypatch.setattr(recommend_similar_mod.collection, "query", fake_source_query)

    try:
        RecommendSimilarTool()._run("Icarus")
    except ToolException:
        pass  # only the first (source) query's kwargs matter for this test

    assert seen_kwargs["where"] == {"secret": False}


def test_filter_by_series_excludes_secret_pieces(monkeypatch):
    seen_kwargs = {}

    def fake_get(**kwargs):
        seen_kwargs.update(kwargs)
        return {"metadatas": [{"title": "Icarus", "url": ""}]}

    monkeypatch.setattr(filter_by_series_mod.collection, "get", fake_get)

    FilterBySeriesTool()._run("Mythical")

    assert seen_kwargs["where"] == {"$and": [{"series": "Mythical"}, {"secret": False}]}


def test_search_artworks_excludes_secret_pieces(monkeypatch):
    seen_kwargs = {}

    def fake_query(**kwargs):
        seen_kwargs.update(kwargs)
        return _fake_query_result([("Icarus", 0.2)])

    monkeypatch.setattr(search_artworks_mod, "embed_query", lambda _q: [0.0])
    monkeypatch.setattr(search_artworks_mod.collection, "query", fake_query)

    SearchArtworksTool()._run("something mythical")

    assert seen_kwargs["where"] == {"secret": False}


def test_search_artworks_drops_low_confidence_results(monkeypatch):
    monkeypatch.setattr(search_artworks_mod, "embed_query", lambda _q: [0.0])
    monkeypatch.setattr(
        search_artworks_mod.collection, "query",
        lambda **_k: _fake_query_result([("Barely related", 5.0)]),
    )

    result = SearchArtworksTool()._run("something mythical")

    assert "No artworks closely matched" in result


def test_filter_by_series_matches_case_insensitively(monkeypatch):
    monkeypatch.setattr(
        filter_by_series_mod.collection, "get",
        lambda **_k: {"metadatas": [{"title": "Icarus", "url": "https://example.com/icarus"}]},
    )

    result = FilterBySeriesTool()._run("mythical")

    assert "Icarus" in result


def test_filter_by_series_raises_for_unknown_series():
    try:
        FilterBySeriesTool()._run("Not A Real Series")
        assert False, "expected a ToolException"
    except ToolException as e:
        assert "not found" in str(e)
