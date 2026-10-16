"""Label generated responses by their BERT similarity to the reference answer.

Both texts are embedded with bert-base-uncased (mean of the last hidden states) and compared by cosine
similarity: above 0.8 is Low hallucination, 0.5 to 0.8 is Medium, and below 0.5 (or an empty text) is High.
"""

from __future__ import annotations

import logging
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from hallumat._optional import require
from hallumat.errors import DatasetError

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final = "bert-base-uncased"
LOW_ABOVE: Final = 0.8
"""Similarity above this is Low hallucination."""
MEDIUM_FROM: Final = 0.5
"""Similarity from this up to LOW_ABOVE (inclusive) is Medium; below it is High."""

INPUT_COLUMNS: Final = ("Generated_Response", "Original_Answer")


class TextEmbedder(Protocol):
    def embed(self, text: str) -> npt.NDArray[np.float32]: ...


class BertEmbedder:
    """Mean-pooled last hidden states of a Hugging Face encoder (default bert-base-uncased)."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        transformers = require("transformers", "transformers", "models")
        self._torch: Any = require("torch", "torch", "models")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._model = transformers.AutoModel.from_pretrained(model_name)

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        return np.asarray(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())


def level_for_similarity(similarity: float) -> str:
    """The hallucination level for a cosine similarity between a response and its reference answer."""
    if similarity > LOW_ABOVE:
        return "Low"
    if MEDIUM_FROM <= similarity <= LOW_ABOVE:
        return "Medium"
    return "High"


def classify(response: str, reference: str, embedder: TextEmbedder) -> tuple[float, str]:
    """(similarity, level) of response against reference; an empty text on either side is (0.0, 'High')."""
    if not response.strip() or not reference.strip():
        return 0.0, "High"
    similarity = float(cosine_similarity([embedder.embed(response)], [embedder.embed(reference)])[0][0])
    return similarity, level_for_similarity(similarity)


def label_responses(frame: pd.DataFrame, embedder: TextEmbedder) -> pd.DataFrame:
    """A copy of frame with Hallucination_Score and Hallucination_Level set for every row.

    Missing responses or answers count as empty text.
    """
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(f"missing column(s) {', '.join(missing)}")
    labelled = frame.copy()
    labelled["Generated_Response"] = labelled["Generated_Response"].fillna("")
    labelled["Original_Answer"] = labelled["Original_Answer"].fillna("")
    results = [
        classify(response, reference, embedder)
        for response, reference in zip(labelled["Generated_Response"], labelled["Original_Answer"], strict=True)
    ]
    labelled["Hallucination_Score"] = [score for score, _ in results]
    labelled["Hallucination_Level"] = [level for _, level in results]
    log.info("labelled %d responses", len(labelled))
    return labelled
