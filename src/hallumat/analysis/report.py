"""Run the whole dataset analysis once, write its outputs and format it as text."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from tabulate import tabulate

from hallumat.analysis import agreement, composition, consistency, figures

log = logging.getLogger(__name__)

METRICS_FILE = "agreement_metrics.csv"


@dataclass(frozen=True)
class Analysis:
    """Every table and statistic the dataset analysis reports."""

    composition: pd.DataFrame
    levels: pd.DataFrame
    anova: consistency.TestResult
    t_test: consistency.TestResult
    overview: pd.DataFrame
    most_inconsistent: pd.DataFrame
    level_distribution: pd.DataFrame
    metrics: pd.DataFrame
    confusion: npt.NDArray[np.int64]


def analyse(frame: pd.DataFrame) -> Analysis:
    """Compute the analysis of a loaded HalluMat dataset."""
    levels = consistency.numeric_levels(frame)
    scores = consistency.phcs_scores(levels)
    return Analysis(
        composition=composition.dataset_composition(frame),
        levels=levels,
        anova=consistency.anova_across_groups(levels),
        t_test=consistency.variability_t_test(levels, scores),
        overview=consistency.overview(levels),
        most_inconsistent=consistency.most_inconsistent(scores),
        level_distribution=consistency.level_distribution(levels),
        metrics=agreement.agreement_metrics(frame),
        confusion=agreement.level_confusion(frame),
    )


def _grid(table: pd.DataFrame) -> str:
    return str(tabulate(cast(Any, table), headers="keys", tablefmt="grid"))


def format_composition(result: Analysis) -> str:
    return "Dataset composition (Table 1)\n" + result.composition.to_string()


def format_consistency(result: Analysis) -> str:
    """ANOVA, t-test, overview, Table 2 and the level distribution."""
    sections = [
        "ANOVA of the hallucination level across question groups\n"
        f"F-statistic: {result.anova.statistic:.4f}, p-value: {result.anova.pvalue:.6f}",
        f"Welch t-test, {consistency.TOP_N} most vs {consistency.TOP_N} least inconsistent groups\n"
        f"T-statistic: {result.t_test.statistic:.4f}, p-value: {result.t_test.pvalue:.6f}",
        "Dataset overview\n" + _grid(result.overview),
        f"Top {consistency.TOP_N} most inconsistent groups by PHCS (Table 2)\n" + _grid(result.most_inconsistent),
        "Hallucination level distribution (1 = Low, 3 = High)\n" + _grid(result.level_distribution),
    ]
    return "\n\n".join(sections)


def format_agreement(result: Analysis) -> str:
    return "Agreement of the recomputed with the assigned levels\n" + _grid(agreement.display_table(result.metrics))


def format_report(result: Analysis) -> str:
    return "\n\n".join([format_composition(result), format_consistency(result), format_agreement(result)])


def write_metrics(result: Analysis, out: Path) -> Path:
    """Write agreement_metrics.csv (LF line ends on every platform)."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / METRICS_FILE
    result.metrics.to_csv(path, index=False, lineterminator="\n")
    return path


def write_figures(result: Analysis, out: Path) -> list[Path]:
    """Write the four figures to out/figures."""
    directory = out / "figures"
    names = dict(zip(("distribution", "spread", "trend", "confusion"), figures.FIGURE_NAMES, strict=True))
    return [
        figures.level_distribution(result.levels, directory / names["distribution"]),
        figures.level_spread_by_group(result.levels, directory / names["spread"]),
        figures.level_trend_by_group(result.levels, directory / names["trend"]),
        figures.confusion_heatmap(result.confusion, directory / names["confusion"]),
    ]


def write_outputs(result: Analysis, out: Path) -> list[Path]:
    """Write the metrics CSV and the figures under out and return their paths."""
    written = [write_metrics(result, out), *write_figures(result, out)]
    for path in written:
        log.debug("wrote %s", path)
    return written
