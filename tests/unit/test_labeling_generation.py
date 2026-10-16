"""Labelling and response generation with fake models (no torch needed)."""

import numpy as np
import pandas as pd
import pytest

from hallumat.errors import DatasetError
from hallumat.generation import generate_responses
from hallumat.labeling import classify, label_responses, level_for_similarity


class FakeEmbedder:
    """Fixed vectors per text; unknown texts raise so a test notices unexpected calls."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []

    def embed(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.asarray(self.vectors[text], dtype=np.float32)


@pytest.mark.parametrize(
    ("similarity", "level"),
    [(1.0, "Low"), (0.8001, "Low"), (0.8, "Medium"), (0.65, "Medium"), (0.5, "Medium"), (0.4999, "High"), (-1, "High")],
)
def test_level_thresholds(similarity: float, level: str):
    assert level_for_similarity(similarity) == level


def test_classify_uses_cosine_similarity():
    embedder = FakeEmbedder({"same": [1, 0], "also": [1, 0], "orth": [0, 1]})
    assert classify("same", "also", embedder) == (pytest.approx(1.0), "Low")
    assert classify("same", "orth", embedder) == (pytest.approx(0.0), "High")


@pytest.mark.parametrize(("response", "reference"), [("", "x"), ("x", "   "), ("", "")])
def test_classify_empty_text_is_high_without_embedding(response: str, reference: str):
    embedder = FakeEmbedder({})
    assert classify(response, reference, embedder) == (0.0, "High")
    assert embedder.calls == []


def test_label_responses_fills_missing_text_and_keeps_other_columns():
    frame = pd.DataFrame({"Generated_Response": ["a", None], "Original_Answer": ["b", "b"], "Group_ID": [7, 8]})
    embedder = FakeEmbedder({"a": [1, 1], "b": [1, 0]})
    labelled = label_responses(frame, embedder)
    assert labelled["Hallucination_Level"].tolist() == ["Medium", "High"]
    assert labelled["Hallucination_Score"].iloc[0] == pytest.approx(1 / np.sqrt(2))
    assert labelled["Group_ID"].tolist() == [7, 8]
    assert frame["Generated_Response"].isna().iloc[1], "the input frame must not change"


def test_label_responses_names_missing_columns():
    with pytest.raises(DatasetError, match="Original_Answer"):
        label_responses(pd.DataFrame({"Generated_Response": ["a"]}), FakeEmbedder({}))


class FakeGenerator:
    def generate(self, question: str) -> str:
        if question == "boom":
            raise RuntimeError("CUDA out of memory")
        return f"answer to {question}"


def test_generate_responses_maps_columns_and_skips_failures(caplog: pytest.LogCaptureFixture):
    questions = pd.DataFrame(
        {"Question": ["q1", "boom", "q3"], "Answer": ["a1", "a2", "a3"], "Explanation": ["e1", "e2", "e3"], "Extra": 1}
    )
    responses = generate_responses(questions, FakeGenerator())
    assert responses.to_dict("records") == [
        {"Question": "q1", "Generated_Response": "answer to q1", "Original_Answer": "a1", "Explanation": "e1"},
        {"Question": "q3", "Generated_Response": "answer to q3", "Original_Answer": "a3", "Explanation": "e3"},
    ]
    assert "index 1" in caplog.text


def test_generate_responses_with_no_rows_keeps_the_columns():
    empty = pd.DataFrame({"Question": [], "Answer": [], "Explanation": []})
    assert list(generate_responses(empty, FakeGenerator()).columns) == [
        "Question",
        "Generated_Response",
        "Original_Answer",
        "Explanation",
    ]


def test_generate_responses_names_missing_columns():
    with pytest.raises(DatasetError, match="Explanation"):
        generate_responses(pd.DataFrame({"Question": ["q"], "Answer": ["a"]}), FakeGenerator())
