"""Batched BERT labelling: the same embedding and thresholds as `hallumat.labeling`, many texts at a time on a GPU.

The single-text labeller averages the last hidden states over every token of one unpadded text. Here texts are
padded into batches, so the average is taken over the attention mask, which gives the same vector.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from hallumat._optional import require
from hallumat.labeling import DEFAULT_MODEL, level_for_similarity


class BatchBertEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 64) -> None:
        transformers = require("transformers", "transformers", "models")
        self._torch: Any = require("torch", "torch", "models")
        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._model = transformers.AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed_many(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """One mean-pooled vector per text (rows in input order)."""
        torch = self._torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors = np.zeros((len(texts), self._model.config.hidden_size), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            inputs = self._tokenizer([texts[i] for i in idx], return_tensors="pt", truncation=True, padding=True).to(
                self.device
            )
            with torch.no_grad():
                hidden = self._model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            vectors[idx] = mean.float().cpu().numpy()
        return vectors


def cosine_rows(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Cosine similarity of each row of a with the same row of b."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    norms = np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a64 * b64).sum(axis=1) / norms
    return np.asarray(np.nan_to_num(result), dtype=np.float64)


def label_pairs(responses: Sequence[str], references: Sequence[str], embed: Any) -> list[tuple[float, str]]:
    """(similarity, level) per pair; an empty text on either side is (0.0, 'High'), as in hallumat.labeling."""
    usable = [i for i, (r, a) in enumerate(zip(responses, references, strict=True)) if r.strip() and a.strip()]
    results: list[tuple[float, str]] = [(0.0, "High")] * len(responses)
    if usable:
        sims = cosine_rows(embed([responses[i] for i in usable]), embed([references[i] for i in usable]))
        for i, sim in zip(usable, sims, strict=True):
            results[i] = (float(sim), level_for_similarity(float(sim)))
    return results
