"""Evidence retrieval for extrinsic verification.

The query is split into every contiguous word span, the spans are ranked by BM25 against the query, each span is
looked up as a Wikipedia page title, and the summaries of the pages that exist become documents. When a SerpAPI
key is set ($SERPAPI_API_KEY), Google Search snippets and Google Scholar titles are added. A dense retriever then
keeps the top-k documents closest to the query.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np

from hallumat._optional import require
from hallumat.detector.models import Embedder

log = logging.getLogger(__name__)

SERPAPI_URL: Final = "https://serpapi.com/search"
SERPAPI_KEY_ENV: Final = "SERPAPI_API_KEY"
USER_AGENT: Final = "HalluMat/2.0 (https://github.com/bhanuprakashvangala/HalluMat)"


@dataclass(frozen=True)
class Document:
    """A piece of evidence and where it came from."""

    text: str
    source: str
    link: str | None = None


class EvidenceSource(Protocol):
    def fetch(self, query: str, ranked_chunks: Sequence[str]) -> list[Document]: ...


def query_chunks(query: str) -> list[str]:
    """Every contiguous span of the query's words, shortest first, without duplicates."""
    words = query.split()
    chunks: dict[str, None] = {}
    for n in range(1, len(words) + 1):
        for i in range(len(words) - n + 1):
            chunks.setdefault(" ".join(words[i : i + n]), None)
    return list(chunks)


def rank_chunks(query: str, chunks: Sequence[str]) -> list[tuple[str, float]]:
    """Chunks with their BM25 score against the query, best first (stable for ties)."""
    if not chunks:
        return []
    bm25 = require("rank_bm25", "rank-bm25", "detector").BM25Okapi([chunk.split() for chunk in chunks])
    scores = bm25.get_scores(query.split())
    return sorted(zip(chunks, (float(s) for s in scores), strict=True), key=lambda pair: pair[1], reverse=True)


class WikipediaSource:
    """English Wikipedia page summaries, looking each chunk up as a page title."""

    def __init__(self, max_lookups: int | None = None, client: Any = None) -> None:
        self.max_lookups = max_lookups
        self._client = client

    def _wiki(self) -> Any:
        if self._client is None:
            wikipediaapi = require("wikipediaapi", "wikipedia-api", "detector")
            self._client = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")
        return self._client

    def fetch(self, query: str, ranked_chunks: Sequence[str]) -> list[Document]:
        wiki = self._wiki()
        chunks = ranked_chunks if self.max_lookups is None else ranked_chunks[: self.max_lookups]
        documents = []
        for chunk in chunks:
            try:
                page = wiki.page(chunk)
                if page.exists():
                    documents.append(Document(page.summary, "Wikipedia", page.fullurl))
            except Exception as exc:
                log.warning("Wikipedia lookup failed for %r: %s", chunk, exc)
        return documents


class SerpApiSource:
    """Google Search snippets and Google Scholar titles through SerpAPI."""

    def __init__(self, api_key: str, results: int = 5, timeout: float = 10.0, session: Any = None) -> None:
        self.api_key = api_key
        self.results = results
        self.timeout = timeout
        self._session = session

    @classmethod
    def from_environment(cls, results: int = 5, timeout: float = 10.0) -> SerpApiSource | None:
        """A source using $SERPAPI_API_KEY, or None if it is not set."""
        key = os.environ.get(SERPAPI_KEY_ENV)
        return cls(key, results, timeout) if key else None

    def _get(self, params: dict[str, Any]) -> dict[str, Any] | None:
        session = self._session or require("requests", "requests", "detector")
        response = session.get(SERPAPI_URL, params={**params, "api_key": self.api_key}, timeout=self.timeout)
        if response.status_code != 200:
            log.warning("SerpAPI %s returned HTTP %s", params.get("engine"), response.status_code)
            return None
        result: dict[str, Any] = response.json()
        return result

    def fetch(self, query: str, ranked_chunks: Sequence[str]) -> list[Document]:
        documents = []
        for engine, source, field in (
            ("google", "Google Search", "snippet"),
            ("google_scholar", "Google Scholar", "title"),
        ):
            try:
                result = self._get({"q": query, "num": self.results, "engine": engine})
            except Exception as exc:
                log.warning("%s retrieval failed: %s", source, exc)
                continue
            for item in (result or {}).get("organic_results", []):
                if field in item:
                    documents.append(Document(item[field], source, item.get("link")))
        return documents


def dense_top_k(query: str, documents: Sequence[Document], embedder: Embedder, k: int) -> list[Document]:
    """The k documents nearest the query by exact L2 distance between embeddings (FAISS IndexFlatL2)."""
    if not documents:
        return []
    faiss = require("faiss", "faiss-cpu", "detector")
    embeddings = np.asarray(embedder.embed_documents([doc.text for doc in documents]), dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    query_embedding = np.asarray([embedder.embed_query(query)], dtype=np.float32)
    _, indices = index.search(query_embedding, min(k, len(documents)))
    return [documents[i] for i in indices[0]]


def retrieve(
    query: str, sources: Sequence[EvidenceSource], embedder: Embedder, k: int
) -> tuple[list[Document], list[Document]]:
    """(every retrieved document, the top-k by dense retrieval)."""
    ranked = [chunk for chunk, _ in rank_chunks(query, query_chunks(query))]
    documents = [doc for source in sources for doc in source.fetch(query, ranked)]
    return documents, dense_top_k(query, documents, embedder, k)
