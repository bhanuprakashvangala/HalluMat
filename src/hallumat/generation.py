"""Generate model responses to the paraphrased questions (Llama-2-7B-chat by default).

The input table has columns Question, Answer and Explanation; the output has Question, Generated_Response,
Original_Answer and Explanation, one row per question that generated without an error.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Final, Protocol

import pandas as pd

from hallumat._optional import require
from hallumat.errors import DatasetError

log = logging.getLogger(__name__)

DEFAULT_MODEL: Final = "meta-llama/Llama-2-7b-chat-hf"
INPUT_COLUMNS: Final = ("Question", "Answer", "Explanation")
TOKEN_ENV: Final = "HF_TOKEN"


@dataclass(frozen=True)
class GenerationSettings:
    """Sampling settings passed to model.generate. max_length counts the prompt tokens too."""

    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


class ResponseGenerator(Protocol):
    def generate(self, question: str) -> str: ...


class HFResponseGenerator:
    """A Hugging Face causal LM; the access token is read from $HF_TOKEN (Llama 2 is gated)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, settings: GenerationSettings | None = None) -> None:
        transformers = require("transformers", "transformers", "models")
        require("torch", "torch", "models")
        token = os.environ.get(TOKEN_ENV)
        self.settings = settings or GenerationSettings()
        self._tokenizer: Any = transformers.AutoTokenizer.from_pretrained(model_name, token=token)
        self._model: Any = transformers.AutoModelForCausalLM.from_pretrained(model_name, token=token)
        # Llama has no pad token; padding with end-of-sequence avoids a generate() error.
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def generate(self, question: str) -> str:
        inputs = self._tokenizer(question, return_tensors="pt", padding=True)
        outputs = self._model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=self.settings.max_length,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            num_return_sequences=1,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        return str(self._tokenizer.decode(outputs[0], skip_special_tokens=True))


def generate_responses(questions: pd.DataFrame, generator: ResponseGenerator) -> pd.DataFrame:
    """Answer every question; a question whose generation raises is logged and skipped."""
    missing = [column for column in INPUT_COLUMNS if column not in questions.columns]
    if missing:
        raise DatasetError(f"missing column(s) {', '.join(missing)}")
    rows = []
    for index, row in questions[list(INPUT_COLUMNS)].iterrows():
        try:
            response = generator.generate(row["Question"])
        except Exception:
            log.exception("generation failed for the question at index %s; skipping it", index)
            continue
        rows.append(
            {
                "Question": row["Question"],
                "Generated_Response": response,
                "Original_Answer": row["Answer"],
                "Explanation": row["Explanation"],
            }
        )
    log.info("generated %d of %d responses", len(rows), len(questions))
    return pd.DataFrame(rows, columns=["Question", "Generated_Response", "Original_Answer", "Explanation"])
