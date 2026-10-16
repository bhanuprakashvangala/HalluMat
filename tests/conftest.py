"""Shared fixtures: the released dataset (skipped when absent, as in an sdist) and a tiny hand-made one."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    if not (REPO / "data" / "HalluMatData.csv").is_file():
        pytest.skip("the released data/HalluMatData.csv is not present (sdist checkout)")
    return REPO


@pytest.fixture(scope="session")
def released(repo_root: Path) -> pd.DataFrame:
    from hallumat.dataset import load_dataset

    return load_dataset(repo_root / "data" / "HalluMatData.csv")


def tiny_frame() -> pd.DataFrame:
    """Three questions with known levels.

    Group 1: Low, Low, Low (PHCS 0). Group 2: Low, High (PHCS sqrt(2)). Group 3: Medium, High, High (PHCS ~0.577).
    """
    assigned = ["Low", "Low", "Low", "Low", "High", "Medium", "High", "High"]
    recomputed = ["Low", "High", "High", "Low", "High", "High", "High", "Medium"]
    groups = [1, 1, 1, 2, 2, 3, 3, 3]
    n = len(groups)
    return pd.DataFrame(
        {
            "Question": [f"q{i}" for i in range(n)],
            "Generated_Response": [f"r{i}" for i in range(n)],
            "Original_Answer": [f"a{i}" for i in range(n)],
            "Hallucination_Score": [0.1 * i for i in range(n)],
            "Hallucination_Level": assigned,
            "Similarity_Score": [0.05 * i for i in range(n)],
            "Computed_Hallucination_Level": recomputed,
            "Group_ID": groups,
        }
    )


@pytest.fixture
def tiny() -> pd.DataFrame:
    return tiny_frame()


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    path = tmp_path / "HalluMatData.csv"
    tiny_frame().to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def restore_hallumat_logger() -> Iterator[None]:
    """The CLI replaces the 'hallumat' logger's handlers; put them back after each test."""
    logger = logging.getLogger("hallumat")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    yield
    logger.handlers[:], logger.level, logger.propagate = saved
