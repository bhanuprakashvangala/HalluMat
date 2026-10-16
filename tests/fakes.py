"""Scriptable stand-ins for the detector's models."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


class ScriptedGenerator:
    """Answers each prompt with reply(prompt) and records (prompt, max_new_tokens)."""

    def __init__(self, reply: Callable[[str], str] | None = None) -> None:
        self.reply = reply or (lambda prompt: prompt)
        self.calls: list[tuple[str, int]] = []

    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        self.calls.append((prompt, max_new_tokens))
        return self.reply(prompt)


class FixedNLI:
    """Returns single for one text and pair_probs(text, pair) for a pair."""

    def __init__(
        self,
        single: Sequence[float] = (0.1, 0.2, 0.7),
        pair: Callable[[str, str], Sequence[float]] | None = None,
    ) -> None:
        self.labels: tuple[str, ...] = ("contradiction", "neutral", "entailment")
        self.single = np.asarray(single, dtype=np.float32)
        self.pair = pair or (lambda text, evidence: (0.0, 0.0, 1.0))
        self.calls: list[tuple[str, str | None]] = []

    def probabilities(self, text: str, pair: str | None = None) -> np.ndarray:
        self.calls.append((text, pair))
        if pair is None:
            return self.single
        return np.asarray(self.pair(text, pair), dtype=np.float32)


class KeywordEmbedder:
    """Bag-of-keywords vectors over a fixed vocabulary, so similarity is predictable."""

    def __init__(self, vocabulary: Sequence[str]) -> None:
        self.vocabulary = list(vocabulary)

    def _vector(self, text: str) -> list[float]:
        words = text.lower().split()
        return [float(words.count(term)) + 1e-6 for term in self.vocabulary]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
