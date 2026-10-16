from pathlib import Path

import pandas as pd
import pytest

from hallumat.dataset import COLUMNS, LEVELS, level_codes, load_dataset
from hallumat.errors import DataNotFoundError, DatasetError


def test_level_codes_start_at_the_given_value():
    labels = pd.Series(["Low", "Medium", "High", "Other"])
    assert level_codes(labels, start=1).tolist()[:3] == [1, 2, 3]
    assert level_codes(labels, start=0).tolist()[:3] == [0, 1, 2]
    assert pd.isna(level_codes(labels, start=0).iloc[3])


def test_load_dataset_reads_a_valid_file(tiny_csv: Path):
    frame = load_dataset(tiny_csv)
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == 8


def test_load_dataset_names_a_missing_file(tmp_path: Path):
    with pytest.raises(DataNotFoundError, match="HalluMat dataset not found"):
        load_dataset(tmp_path / "missing.csv")


def test_load_dataset_names_missing_columns(tiny_csv: Path):
    frame = pd.read_csv(tiny_csv).drop(columns=["Group_ID", "Similarity_Score"])
    frame.to_csv(tiny_csv, index=False)
    with pytest.raises(DatasetError, match="Similarity_Score, Group_ID"):
        load_dataset(tiny_csv)


def test_load_dataset_rejects_unknown_levels(tiny_csv: Path):
    frame = pd.read_csv(tiny_csv)
    frame.loc[0, "Computed_Hallucination_Level"] = "Extreme"
    frame.to_csv(tiny_csv, index=False)
    with pytest.raises(
        DatasetError, match=r"Computed_Hallucination_Level has values outside Low/Medium/High: \['Extreme'\]"
    ):
        load_dataset(tiny_csv)


def test_load_dataset_rejects_non_integer_groups(tiny_csv: Path):
    frame = pd.read_csv(tiny_csv)
    frame["Group_ID"] = frame["Group_ID"].astype(str) + "a"
    frame.to_csv(tiny_csv, index=False)
    with pytest.raises(DatasetError, match="Group_ID must hold integers"):
        load_dataset(tiny_csv)


def test_released_dataset_has_the_expected_shape(released: pd.DataFrame):
    assert released.shape == (3269, len(COLUMNS))
    assert released["Group_ID"].nunique() == 655
    assert set(released["Hallucination_Level"]) == set(LEVELS)
