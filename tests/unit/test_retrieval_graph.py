"""Retrieval and the fact graph. BM25, FAISS, networkx and python-louvain come with the dev install; torch does not."""

from types import SimpleNamespace

import pytest

from hallumat.detector import graph, retrieval
from hallumat.detector.retrieval import Document, SerpApiSource, WikipediaSource
from tests.fakes import KeywordEmbedder


def test_query_chunks_are_every_contiguous_span_once():
    assert retrieval.query_chunks("a b a") == ["a", "b", "a b", "b a", "a b a"]
    assert retrieval.query_chunks("") == []


def test_rank_chunks_puts_the_best_bm25_match_first():
    pytest.importorskip("rank_bm25")
    ranked = retrieval.rank_chunks("steel alloy", ["copper", "steel alloy", "steel"])
    assert ranked[0][0] == "steel alloy"
    assert [chunk for chunk, _ in ranked][-1] == "copper"
    assert retrieval.rank_chunks("x", []) == []


class FakeWiki:
    def __init__(self, pages: dict[str, str], failing: str | None = None) -> None:
        self.pages = pages
        self.failing = failing
        self.looked_up: list[str] = []

    def page(self, title: str) -> SimpleNamespace:
        self.looked_up.append(title)
        if title == self.failing:
            raise ConnectionError("timeout")
        return SimpleNamespace(
            exists=lambda: title in self.pages,
            summary=self.pages.get(title, ""),
            fullurl=f"https://en.wikipedia.org/wiki/{title}",
        )


def test_wikipedia_source_keeps_existing_pages_and_logs_failures(caplog: pytest.LogCaptureFixture):
    wiki = FakeWiki({"Steel": "Steel is an alloy."}, failing="rust")
    docs = WikipediaSource(client=wiki).fetch("q", ["Steel", "rust", "nothing"])
    assert docs == [Document("Steel is an alloy.", "Wikipedia", "https://en.wikipedia.org/wiki/Steel")]
    assert "rust" in caplog.text


def test_wikipedia_source_honours_the_lookup_cap():
    wiki = FakeWiki({})
    WikipediaSource(max_lookups=2, client=wiki).fetch("q", ["a", "b", "c"])
    assert wiki.looked_up == ["a", "b"]


class FakeSession:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def get(self, url: str, params: dict[str, object], timeout: float) -> SimpleNamespace:
        self.requests.append(params)
        reply = self.responses[str(params["engine"])]
        if isinstance(reply, Exception):
            raise reply
        status, body = reply  # type: ignore[misc]
        return SimpleNamespace(status_code=status, json=lambda: body)


def test_serpapi_source_reads_google_snippets_and_scholar_titles():
    session = FakeSession(
        {
            "google": (200, {"organic_results": [{"snippet": "Steel snippet", "link": "g1"}, {"title": "no snippet"}]}),
            "google_scholar": (200, {"organic_results": [{"title": "A paper", "link": "s1"}]}),
        }
    )
    docs = SerpApiSource("key", session=session).fetch("steel", [])
    assert docs == [Document("Steel snippet", "Google Search", "g1"), Document("A paper", "Google Scholar", "s1")]
    assert all(r["api_key"] == "key" and r["q"] == "steel" and r["num"] == 5 for r in session.requests)


def test_serpapi_source_survives_errors_and_bad_status(caplog: pytest.LogCaptureFixture):
    session = FakeSession({"google": ConnectionError("down"), "google_scholar": (429, {})})
    assert SerpApiSource("key", session=session).fetch("q", []) == []
    assert "Google Search retrieval failed" in caplog.text
    assert "HTTP 429" in caplog.text


def test_serpapi_source_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(retrieval.SERPAPI_KEY_ENV, raising=False)
    assert SerpApiSource.from_environment() is None
    monkeypatch.setenv(retrieval.SERPAPI_KEY_ENV, "secret")
    source = SerpApiSource.from_environment(results=3)
    assert source is not None
    assert (source.api_key, source.results) == ("secret", 3)


def test_dense_top_k_returns_the_nearest_documents():
    pytest.importorskip("faiss")
    embedder = KeywordEmbedder(["steel", "copper", "glass"])
    docs = [Document("glass glass", "w"), Document("steel steel", "w"), Document("copper", "w")]
    assert retrieval.dense_top_k("steel", docs, embedder, k=1) == [docs[1]]
    assert len(retrieval.dense_top_k("steel", docs, embedder, k=10)) == 3
    assert retrieval.dense_top_k("steel", [], embedder, k=3) == []


def test_extract_facts_keeps_sentences_of_at_least_the_minimum_length():
    text = "Short one. This sentence is long enough to count.  Another sufficiently long sentence here."
    assert graph.extract_facts(text, 21) == [
        "This sentence is long enough to count",
        "Another sufficiently long sentence here",
    ]


def test_fact_graph_links_similar_facts_and_groups_communities():
    pytest.importorskip("networkx")
    pytest.importorskip("community")
    embedder = KeywordEmbedder(["steel", "glass"])
    facts = ["steel is strong", "steel rusts", "glass is brittle", "glass is clear"]
    fact_graph = graph.build_fact_graph(facts, embedder, threshold=0.5)
    assert sorted(tuple(sorted(edge)) for edge in fact_graph.edges()) == [(0, 1), (2, 3)]
    graph.assign_communities(fact_graph, seed=0)
    groups = graph.facts_by_community(fact_graph)
    assert sorted(map(sorted, groups.values())) == sorted([sorted(facts[:2]), sorted(facts[2:])])
    table = graph.format_communities(groups)
    assert "Community 0" in table
    assert "1. " in table


def test_fact_graph_of_nothing_is_empty():
    pytest.importorskip("networkx")
    assert graph.build_fact_graph([], KeywordEmbedder(["a"]), 0.5).number_of_nodes() == 0
