"""Figures of the fact graph: 3D and 2D Kamada-Kawai layouts coloured by community, and a TF-IDF similarity
heatmap of the facts. Each function saves a PNG and returns its path, or None when there is nothing to draw."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hallumat._optional import require

DPI = 150


def _pyplot(show: bool) -> Any:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(plt: Any, path: Path, show: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


def _style(graph: Any) -> tuple[list[int], list[int]]:
    degrees = dict(graph.degree())
    sizes = [300 + 150 * degrees[node] for node in graph.nodes()]
    colors = [graph.nodes[node].get("community", 0) for node in graph.nodes()]
    return sizes, colors


def graph_3d(graph: Any, path: Path, show: bool = False) -> Path | None:
    if not graph.nodes():
        return None
    nx = require("networkx", "networkx", "detector")
    plt = _pyplot(show)
    pos = nx.kamada_kawai_layout(graph, dim=3)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    for u, v in graph.edges():
        (x0, y0, z0), (x1, y1, z1) = pos[u], pos[v]
        ax.plot([x0, x1], [y0, y1], [z0, z1], color="gray", alpha=0.5, lw=1)
    sizes, colors = _style(graph)
    nodes = list(graph.nodes())
    scatter = ax.scatter(
        [pos[n][0] for n in nodes], [pos[n][1] for n in nodes], [pos[n][2] for n in nodes],
        s=sizes, c=colors, cmap="viridis", alpha=0.9,
    )  # fmt: skip
    for node in nodes:
        x, y, z = pos[node]
        ax.text(x, y, z, f"{node}", fontsize=10, color="black")
    ax.set_title("3D Knowledge Graph Visualization", fontsize=16)
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)
    ax.set_zlabel("Z-axis", fontsize=12)
    fig.colorbar(scatter, shrink=0.5, aspect=10, label="Community")
    return _finish(plt, path, show)


def graph_2d(graph: Any, path: Path, show: bool = False) -> Path | None:
    if not graph.nodes():
        return None
    nx = require("networkx", "networkx", "detector")
    plt = _pyplot(show)
    pos = nx.kamada_kawai_layout(graph, dim=2)
    plt.figure(figsize=(14, 10))
    sizes, colors = _style(graph)
    nodes = nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color=colors, cmap=plt.cm.viridis, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="black")
    plt.title("2D Knowledge Graph Visualization", fontsize=16)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.colorbar(nodes, label="Community")
    return _finish(plt, path, show)


def similarity_heatmap(facts: Sequence[str], path: Path, show: bool = False) -> Path | None:
    """TF-IDF cosine similarity between every pair of facts; needs at least two facts."""
    if len(facts) < 2:
        return None
    import pandas as pd
    import seaborn as sns
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    plt = _pyplot(show)
    similarities = cosine_similarity(TfidfVectorizer().fit_transform(facts))
    names = [f"Fact {i + 1}" for i in range(len(facts))]
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        pd.DataFrame(similarities, columns=names, index=names),
        annot=True, cmap="YlGnBu", linewidths=0.5, annot_kws={"size": 10}, fmt=".2f",
    )  # fmt: skip
    plt.title("Heatmap of Fact Similarity (Consistency)", fontsize=14)
    plt.xlabel("Fact Fragments", fontsize=12)
    plt.ylabel("Fact Fragments", fontsize=12)
    return _finish(plt, path, show)
