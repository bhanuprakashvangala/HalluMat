"""Pick the final questions: no exact or near duplicates, balanced between the two sources."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence

import numpy as np

from hallumat._optional import require
from hallumat.detector.models import Embedder
from hallumat.extend.records import Record, normalise

log = logging.getLogger(__name__)

NEAR_DUPLICATE = 0.92
"""Cosine similarity of question embeddings above which two questions count as the same question."""


def drop_exact_duplicates(candidates: Sequence[Record]) -> list[Record]:
    seen: set[str] = set()
    unique = []
    for record in candidates:
        key = normalise(record["question"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def drop_near_duplicates(
    candidates: Sequence[Record], embedder: Embedder, threshold: float = NEAR_DUPLICATE
) -> list[Record]:
    """Keep a question unless an earlier kept question is more similar than threshold (inner product of
    normalised embeddings, searched with FAISS)."""
    if not candidates:
        return []
    faiss = require("faiss", "faiss-cpu", "detector")
    vectors = np.asarray(embedder.embed_documents([r["question"] for r in candidates]), dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    kept = []
    for i, record in enumerate(candidates):
        vector = vectors[i : i + 1]
        if index.ntotal:
            scores, _ = index.search(vector, 1)
            if scores[0][0] > threshold:
                continue
        index.add(vector)
        kept.append(record)
    log.info("near-duplicate filter kept %d of %d", len(kept), len(candidates))
    return kept


def choose(mp: Sequence[Record], wiki: Sequence[Record], total: int, seed: int = 0) -> list[Record]:
    """total questions, half from each source when both have enough; the shortfall of one comes from the other.
    Materials Project questions are taken evenly across properties and, within a property, simpler compounds
    (fewer elements) first, in random order among equals."""
    rng = random.Random(seed)
    wiki_pool = list(wiki)
    rng.shuffle(wiki_pool)
    by_topic: dict[str, list[Record]] = {}
    for record in mp:
        by_topic.setdefault(record["topic"], []).append(record)
    for pool in by_topic.values():
        rng.shuffle(pool)
        # pop() takes from the end: most elements first in the list, so the simplest compounds are taken first.
        pool.sort(key=lambda r: r.get("n_elements") or 9, reverse=True)
    mp_pool = []
    while any(by_topic.values()):
        for topic in sorted(by_topic):
            if by_topic[topic]:
                mp_pool.append(by_topic[topic].pop())

    n_wiki = min(len(wiki_pool), max(total // 2, total - len(mp_pool)))
    n_mp = min(len(mp_pool), total - n_wiki)
    chosen = wiki_pool[:n_wiki] + mp_pool[:n_mp]
    rng.shuffle(chosen)
    if len(chosen) < total:
        log.warning("only %d questions available for a target of %d", len(chosen), total)
    return chosen
