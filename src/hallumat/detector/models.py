"""The three models behind the detector, behind small interfaces so they can be swapped or faked in tests.

- TextGenerator: the LLM that answers, verifies and refines (Llama-2-7B-chat).
- NLIModel: class probabilities for one text or a (premise, hypothesis) pair (DeBERTa-v3).
- Embedder: sentence embeddings for retrieval and the fact graph (all-MiniLM-L6-v2).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from hallumat._optional import require


class TextGenerator(Protocol):
    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        """The generated text, which starts with the prompt itself."""
        ...


POSITIONAL_NLI_LABELS = ("contradiction", "neutral", "entailment")
"""Labels by class index for a model whose config does not name them (the original detector's assumption)."""


class NLIModel(Protocol):
    labels: tuple[str, ...]
    """Lower-case label name of each class index."""

    def probabilities(self, text: str, pair: str | None = None) -> npt.NDArray[np.float32]:
        """Softmax over the classifier's labels for text, or for the pair (text, pair)."""
        ...


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HFTextGenerator:
    """A transformers text-generation pipeline over a causal LM, placed with device_map='auto'."""

    def __init__(self, model_name: str, max_new_tokens: int = 200) -> None:
        transformers = require("transformers", "transformers", "models")
        require("torch", "torch", "models")
        require("accelerate", "accelerate", "models")
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._pipeline: Any = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens
        )

    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        return str(self._pipeline(prompt, max_new_tokens=max_new_tokens, truncation=True)[0]["generated_text"])


class HFNLIModel:
    """A sequence classifier. Class names come from the model config when it names an 'entailment' class, so NLI
    models with any label order work; otherwise classes 0, 1 and 2 are read as contradiction, neutral and entailment.

    microsoft/deberta-v3-large, the original detector's choice, is a base model without a classification head:
    transformers initialises a random two-class head for it, which logs a warning, never predicts entailment, and
    changes on every load. An NLI-trained checkpoint such as
    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli avoids both.
    """

    def __init__(self, model_name: str) -> None:
        transformers = require("transformers", "transformers", "models")
        self._torch: Any = require("torch", "torch", "models")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        named = [str(self._model.config.id2label[i]).lower() for i in range(self._model.config.num_labels)]
        self.labels: tuple[str, ...] = tuple(named) if "entailment" in named else POSITIONAL_NLI_LABELS

    def probabilities(self, text: str, pair: str | None = None) -> npt.NDArray[np.float32]:
        if pair is None:
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
        else:
            inputs = self._tokenizer(text, pair, return_tensors="pt", truncation=True)
        logits = self._model(**inputs).logits
        return np.asarray(self._torch.nn.functional.softmax(logits, dim=-1).detach().numpy()[0])


class SentenceEmbedder:
    """sentence-transformers embeddings. Newlines become spaces before encoding, as LangChain's
    HuggingFaceEmbeddings did in the original detector."""

    def __init__(self, model_name: str) -> None:
        sentence_transformers = require("sentence_transformers", "sentence-transformers", "detector")
        self._model: Any = sentence_transformers.SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [text.replace("\n", " ") for text in texts]
        return [list(map(float, row)) for row in self._model.encode(cleaned)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
