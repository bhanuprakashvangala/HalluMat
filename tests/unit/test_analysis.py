"""The analysis functions on a tiny dataset whose answers are worked out by hand."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f_oneway, ttest_ind

from hallumat.analysis import agreement, composition, consistency
from hallumat.analysis.report import analyse, format_report, write_metrics


def test_composition_counts_groups_paraphrases_and_levels(tiny: pd.DataFrame):
    table = composition.dataset_composition(tiny)
    assert table["Count"].tolist() == [3, 8, 8, 4, 1, 3]
    assert table["Metric"].iloc[0] == "Total Unique Queries"


def test_composition_does_not_count_empty_cells(tiny: pd.DataFrame):
    tiny.loc[0, "Generated_Response"] = None
    assert composition.dataset_composition(tiny)["Count"].tolist()[:3] == [3, 8, 7]


def test_numeric_levels_maps_one_to_three_and_bins_groups(tiny: pd.DataFrame):
    levels = consistency.numeric_levels(tiny)
    assert levels["Hallucination_Level"].tolist() == [1, 1, 1, 1, 3, 2, 3, 3]
    assert levels["Computed_Hallucination_Level"].tolist() == [1, 3, 3, 1, 3, 3, 3, 2]
    assert list(levels["Group_Range"].cat.categories) == ["0-99"]
    assert tiny["Hallucination_Level"].iloc[0] == "Low", "the input frame must not change"


def test_group_ranges_are_hundreds(tiny: pd.DataFrame):
    tiny["Group_ID"] = [1, 1, 1, 150, 150, 250, 250, 250]
    levels = consistency.numeric_levels(tiny)
    assert list(levels["Group_Range"].cat.categories) == ["0-99", "100-199", "200-299"]
    assert levels["Group_Range"].astype(str).tolist() == ["0-99"] * 3 + ["100-199"] * 2 + ["200-299"] * 3


def test_phcs_is_the_sample_std_of_each_groups_levels(tiny: pd.DataFrame):
    scores = consistency.phcs_scores(consistency.numeric_levels(tiny))
    assert scores["Group_ID"].tolist() == [1, 2, 3]
    assert scores["PHCS_Score"].tolist() == pytest.approx([0.0, math.sqrt(2), np.std([2, 3, 3], ddof=1)])


def test_most_and_least_inconsistent_order(tiny: pd.DataFrame):
    scores = consistency.phcs_scores(consistency.numeric_levels(tiny))
    assert consistency.most_inconsistent(scores, 2)["Group_ID"].tolist() == [2, 3]
    assert consistency.least_inconsistent(scores, 1)["Group_ID"].tolist() == [1]


def test_tests_match_scipy_called_directly(tiny: pd.DataFrame):
    levels = consistency.numeric_levels(tiny)
    groups = [g["Hallucination_Level"] for _, g in levels.groupby("Group_ID")]
    expected = f_oneway(*groups)
    result = consistency.anova_across_groups(levels)
    assert (result.statistic, result.pvalue) == pytest.approx((expected.statistic, expected.pvalue))

    scores = consistency.phcs_scores(levels)
    t = consistency.variability_t_test(levels, scores, n=1)
    expected_t = ttest_ind(groups[1], groups[0], equal_var=False)
    assert t.statistic == pytest.approx(expected_t.statistic)


def test_level_distribution_and_overview(tiny: pd.DataFrame):
    levels = consistency.numeric_levels(tiny)
    assert consistency.level_distribution(levels).values.tolist() == [[1, 4], [2, 1], [3, 3]]
    assert list(consistency.overview(levels).index) == ["count", "mean", "std", "min", "max"]


def test_agreement_metrics_by_hand(tiny: pd.DataFrame):
    metrics = agreement.agreement_metrics(tiny)
    assert metrics["Metric"].tolist() == ["Accuracy", "Precision", "Recall", "F1-Score"]
    # 4 of the 8 recomputed levels match: rows 0, 3, 4 and 6.
    assert metrics["Score"].iloc[0] == pytest.approx(0.5)
    assert agreement.display_table(metrics)["Metric"].tolist()[1] == "Precision (macro)"


def test_confusion_rows_are_assigned_and_columns_recomputed(tiny: pd.DataFrame):
    expected = np.array([[2, 0, 2], [0, 0, 1], [0, 1, 2]])
    np.testing.assert_array_equal(agreement.level_confusion(tiny), expected)


def test_confusion_keeps_all_three_levels_when_one_is_absent(tiny: pd.DataFrame):
    tiny["Hallucination_Level"] = "High"
    tiny["Computed_Hallucination_Level"] = "High"
    assert agreement.level_confusion(tiny).shape == (3, 3)


def test_metrics_csv_has_lf_line_ends(tiny: pd.DataFrame, tmp_path: Path):
    path = write_metrics(analyse(tiny), tmp_path)
    data = path.read_bytes()
    assert data.startswith(b"Metric,Score\n")
    assert b"\r" not in data


def test_report_names_every_section(tiny: pd.DataFrame):
    text = format_report(analyse(tiny))
    for heading in [
        "Table 1",
        "ANOVA",
        "Welch t-test",
        "Dataset overview",
        "Table 2",
        "level distribution",
        "Agreement",
    ]:
        assert heading in text
