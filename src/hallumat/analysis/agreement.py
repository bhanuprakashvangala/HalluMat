"""Agreement between the assigned hallucination levels and the levels recomputed by the second labelling pass."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from hallumat.dataset import LEVELS, level_codes

METRIC_NAMES = ("Accuracy", "Precision", "Recall", "F1-Score")
DISPLAY_NAMES = ("Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)")


def _codes(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return (
        level_codes(frame["Hallucination_Level"], start=0),
        level_codes(frame["Computed_Hallucination_Level"], start=0),
    )


def agreement_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and macro precision, recall and F1 of the recomputed against the assigned levels.

    Columns Metric and Score, full precision; this is results/agreement_metrics.csv.
    """
    assigned, recomputed = _codes(frame)
    precision, recall, f1, _ = precision_recall_fscore_support(assigned, recomputed, average="macro", zero_division=0)
    return pd.DataFrame(
        {"Metric": list(METRIC_NAMES), "Score": [accuracy_score(assigned, recomputed), precision, recall, f1]}
    )


def display_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """The metrics rounded to three decimals, with the macro averages named as such."""
    return pd.DataFrame({"Metric": list(DISPLAY_NAMES), "Value": metrics["Score"].to_numpy()}).round(3)


def level_confusion(frame: pd.DataFrame) -> npt.NDArray[np.int64]:
    """3 x 3 counts: rows are the assigned level, columns the recomputed level, both Low/Medium/High."""
    assigned, recomputed = _codes(frame)
    return np.asarray(confusion_matrix(assigned, recomputed, labels=list(range(len(LEVELS)))))
