"""The HalluMat dataset: loading, validation and the hallucination-level encodings.

HalluMatData.csv has one row per paraphrased question. Group_ID ties the paraphrases of one original question
together (1 to 655). Hallucination_Level is the label assigned from the BERT similarity between the generated
response and the reference answer (`hallumat.labeling`); Computed_Hallucination_Level is the level recomputed
by the second labelling pass, and the two are compared in `hallumat.analysis.agreement`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from hallumat.errors import DatasetError
from hallumat.paths import require_file

LEVELS: Final = ("Low", "Medium", "High")
"""The three hallucination levels, from least to most hallucinated."""

COLUMNS: Final = (
    "Question",
    "Generated_Response",
    "Original_Answer",
    "Hallucination_Score",
    "Hallucination_Level",
    "Similarity_Score",
    "Computed_Hallucination_Level",
    "Group_ID",
)

LEVEL_COLUMNS: Final = ("Hallucination_Level", "Computed_Hallucination_Level")


def level_codes(labels: pd.Series, start: int) -> pd.Series:
    """Map Low/Medium/High to start, start + 1 and start + 2; any other value becomes NaN.

    The paper's statistics use 1 to 3 (1 = Low); the agreement metrics and the confusion matrix use 0 to 2.
    """
    return labels.map({level: start + i for i, level in enumerate(LEVELS)})


def load_dataset(path: Path) -> pd.DataFrame:
    """Read HalluMatData.csv and check its columns, levels and group ids.

    Raises DataNotFoundError if the file is missing and DatasetError if a column is missing, a level is not one
    of Low/Medium/High, or a Group_ID is not an integer.
    """
    frame = pd.read_csv(require_file(path, "HalluMat dataset"))
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(f"{path}: missing column(s) {', '.join(missing)}")
    for column in LEVEL_COLUMNS:
        unknown = sorted(set(frame[column].dropna()) - set(LEVELS))
        if unknown:
            raise DatasetError(f"{path}: {column} has values outside {'/'.join(LEVELS)}: {unknown}")
    if not pd.api.types.is_integer_dtype(frame["Group_ID"]):
        raise DatasetError(f"{path}: Group_ID must hold integers")
    return frame
