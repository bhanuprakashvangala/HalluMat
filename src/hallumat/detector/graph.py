"""The fact graph: sentences of the refined answer, linked by embedding similarity and grouped by Louvain
community detection, so related claims (and contradictions between them) can be read side by side."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

from hallumat._optional import require
from hallumat.detector.models import Embedder


def extract_facts(text: str, min_length: int) -> list[str]:
    """The text's sentences (split on '.') that are at least min_length characters after stripping."""
    return [fact.strip() for fact in text.split(".") if len(fact.strip()) >= min_length]


def build_fact_graph(facts: Sequence[str], embedder: Embedder, threshold: float) -> Any:
    """A networkx graph with one node per fact (attributes text, embedding, type='fact') and an edge, weighted by
    cosine similarity, between every pair of facts more similar than threshold."""
    nx = require("networkx", "networkx", "detector")
    graph = nx.Graph()
    if not facts:
        return graph
    embeddings = embedder.embed_documents(list(facts))
    for i, fact in enumerate(facts):
        graph.add_node(i, text=fact, embedding=embeddings[i], type="fact")
        for j in range(i):
            similarity = float(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
            if similarity > threshold:
                graph.add_edge(i, j, weight=similarity)
    return graph


def assign_communities(graph: Any, seed: int | None = None) -> dict[int, int]:
    """Louvain communities by edge weight; stored as the 'community' node attribute and returned."""
    nx = require("networkx", "networkx", "detector")
    community_louvain = require("community", "python-louvain", "detector")
    partition: dict[int, int] = community_louvain.best_partition(graph, weight="weight", random_state=seed)
    nx.set_node_attributes(graph, partition, "community")
    return partition


def facts_by_community(graph: Any) -> dict[int, list[str]]:
    """Fact texts grouped by community, communities in ascending order."""
    groups: dict[int, list[str]] = {}
    for node in graph.nodes():
        groups.setdefault(graph.nodes[node].get("community", 0), []).append(graph.nodes[node].get("text", ""))
    return dict(sorted(groups.items()))


def format_communities(groups: dict[int, list[str]]) -> str:
    rows = [
        {"Community": f"Community {community}", "Fragments": "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))}
        for community, facts in groups.items()
    ]
    return str(tabulate(rows, headers="keys", tablefmt="fancy_grid"))
