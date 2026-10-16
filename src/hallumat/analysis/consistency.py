"""Paraphrase consistency: the PHCS score (Table 2), the ANOVA across groups and the Welch t-test.

The Paraphrased Hallucination Consistency Score (PHCS) of a question is the sample standard deviation of the
hallucination levels (1 = Low, 2 = Medium, 3 = High) of its paraphrases: 0 means every paraphrase got the same
level, and a higher score means the model's answers swing between levels when the question is reworded.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import f_oneway, ttest_ind

from hallumat.dataset import LEVEL_COLUMNS, level_codes

GROUP_BIN_SIZE = 100
"""Width of the Group_ID ranges in the spread and trend figures."""

TOP_N = 10
"""Rows of Table 2, and the size of each side of the t-test."""


@dataclass(frozen=True)
class TestResult:
    """A test statistic and its p-value."""

    statistic: float
    pvalue: float


def numeric_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy of frame with both level columns as 1 to 3 and a categorical Group_Range column.

    Group_Range bins Group_ID into [0, 100), [100, 200), ... labelled '0-99', '100-199', ...
    """
    data = frame.copy()
    for column in LEVEL_COLUMNS:
        data[column] = level_codes(data[column], start=1)
    data["Group_ID"] = pd.to_numeric(data["Group_ID"], errors="coerce")
    data = data.dropna(subset=["Group_ID"])
    data["Group_ID"] = data["Group_ID"].astype(int)

    bins = list(range(0, data["Group_ID"].max() + GROUP_BIN_SIZE, GROUP_BIN_SIZE))
    labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
    data["Group_Range"] = pd.cut(data["Group_ID"], bins=bins, labels=labels, right=False)
    return data


def anova_across_groups(levels: pd.DataFrame) -> TestResult:
    """One-way ANOVA of the hallucination level across all question groups (frame from numeric_levels)."""
    result = f_oneway(*[group["Hallucination_Level"].dropna() for _, group in levels.groupby("Group_ID")])
    return TestResult(float(result.statistic), float(result.pvalue))


def phcs_scores(levels: pd.DataFrame) -> pd.DataFrame:
    """PHCS per question: columns Group_ID and PHCS_Score, one row per group in Group_ID order."""
    scores = levels.groupby("Group_ID")["Hallucination_Level"].std().reset_index()
    scores.columns = pd.Index(["Group_ID", "PHCS_Score"])
    return scores


def most_inconsistent(scores: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """The n groups with the highest PHCS (Table 2). Ties keep pandas' default sort order."""
    return scores.sort_values(by="PHCS_Score", ascending=False).head(n)


def least_inconsistent(scores: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """The n groups with the lowest PHCS."""
    return scores.sort_values(by="PHCS_Score", ascending=True).head(n)


def variability_t_test(levels: pd.DataFrame, scores: pd.DataFrame, n: int = TOP_N) -> TestResult:
    """Welch t-test of the levels in the n most against the n least inconsistent groups."""
    high = most_inconsistent(scores, n)["Group_ID"]
    low = least_inconsistent(scores, n)["Group_ID"]
    result = ttest_ind(
        levels[levels["Group_ID"].isin(high)]["Hallucination_Level"].dropna(),
        levels[levels["Group_ID"].isin(low)]["Hallucination_Level"].dropna(),
        equal_var=False,
    )
    return TestResult(float(result.statistic), float(result.pvalue))


def overview(levels: pd.DataFrame) -> pd.DataFrame:
    """count, mean, std, min and max of every numeric column."""
    return levels.describe().loc[["count", "mean", "std", "min", "max"]]


def level_distribution(levels: pd.DataFrame) -> pd.DataFrame:
    """Rows per hallucination level (1 to 3): columns Hallucination_Level and Count."""
    counts = levels["Hallucination_Level"].value_counts().sort_index()
    return pd.DataFrame({"Hallucination_Level": counts.index, "Count": counts.values})
