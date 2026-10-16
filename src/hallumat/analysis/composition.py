"""Table 1 of the paper: how many questions, paraphrases and responses the dataset holds, by hallucination level."""

from __future__ import annotations

import pandas as pd


def dataset_composition(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the Metric/Count table of Table 1 for a loaded HalluMat dataset.

    A Group_ID is one original question; its rows are the paraphrases. Paraphrases and responses are counted as
    non-empty cells, so an empty question or response would not count.
    """
    levels = frame["Hallucination_Level"].value_counts()
    return pd.DataFrame(
        {
            "Metric": [
                "Total Unique Queries",
                "Total Paraphrased Queries",
                "Total Generated Responses",
                "Low Hallucination Responses",
                "Medium Hallucination Responses",
                "High Hallucination Responses",
            ],
            "Count": [
                frame["Group_ID"].nunique(),
                frame["Question"].count(),
                frame["Generated_Response"].count(),
                levels.get("Low", 0),
                levels.get("Medium", 0),
                levels.get("High", 0),
            ],
        }
    )
