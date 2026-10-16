"""The dataset figures: level distribution, spread and trend by Group_ID range, and the agreement confusion matrix.

Every function draws one figure, saves it as PNG (200 dpi, tight bounding box) and closes it. With the plotting
stack pinned in constraints/reproduce.txt the files are byte-identical to the committed results/figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from hallumat.dataset import LEVELS

DPI = 200

FIGURE_NAMES = (
    "level_distribution.png",
    "level_spread_by_group.png",
    "level_trend_by_group.png",
    "confusion_matrix.png",
)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(plt: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def level_distribution(levels: pd.DataFrame, path: Path) -> Path:
    """Histogram with KDE of the hallucination levels (1 to 3)."""
    import seaborn as sns

    plt = _pyplot()
    plt.figure(figsize=(10, 5))
    sns.histplot(levels["Hallucination_Level"], bins=20, kde=True, color="blue")
    plt.title("Distribution of Hallucination Levels")
    plt.xlabel("Hallucination Level")
    plt.ylabel("Frequency")
    return _save(plt, path)


def level_spread_by_group(levels: pd.DataFrame, path: Path) -> Path:
    """Box plot of the levels in each Group_ID range."""
    import seaborn as sns

    plt = _pyplot()
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Group_Range", y="Hallucination_Level", data=levels)
    plt.xticks(rotation=45, ha="right")
    plt.title("Hallucination Level Spread Across Groups (Binned)")
    plt.xlabel("Group ID Range")
    plt.ylabel("Hallucination Level (1=Low, 3=High)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    return _save(plt, path)


def level_trend_by_group(levels: pd.DataFrame, path: Path) -> Path:
    """Mean level in each Group_ID range."""
    import seaborn as sns

    plt = _pyplot()
    plt.figure(figsize=(12, 6))
    sns.pointplot(x="Group_Range", y="Hallucination_Level", data=levels, estimator="mean", errorbar=None)
    plt.xticks(rotation=45, ha="right")
    plt.title("Average Hallucination Trend Across Group Ranges")
    plt.xlabel("Group ID Range")
    plt.ylabel("Average Hallucination Level")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    return _save(plt, path)


def confusion_heatmap(matrix: npt.NDArray[np.int64], path: Path) -> Path:
    """Annotated heatmap of the assigned (rows) against the recomputed (columns) levels."""
    import seaborn as sns

    plt = _pyplot()
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(LEVELS), yticklabels=list(LEVELS))
    plt.xlabel("Recomputed Labels")
    plt.ylabel("Computed Labels")
    plt.title("Confusion Matrix: Computed vs Recomputed Hallucination Levels")
    return _save(plt, path)
