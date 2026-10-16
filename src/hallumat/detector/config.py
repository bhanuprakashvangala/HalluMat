"""HalluMatDetector settings: the models, the score weights and every threshold, with the paper's defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

DEFAULT_WEIGHTS: Final = MappingProxyType(
    {"self_consistency": 0.2, "confidence": 0.2, "entropy": 0.2, "nli_verification": 0.4}
)
"""Weights of the four signals in the reliability score; they sum to 1."""

PERTURBATION_TEMPLATES: Final = (
    "Is it accurate to say that ’[statement]’?",
    "Would you consider the statement ’[statement]’ to be correct?",
    "Can we confirm that ’[statement]’ is true?",
    "Does the statement ’[statement]’ hold true?",
    "Is ’[statement]’ a valid statement?",
    "Is there accuracy in the claim ’[statement]’?",
    "Could ’[statement]’ be considered a factual statement?",
    "Is it correct to assume that ’[statement]’ is true?",
    "Would it be right to say ’[statement]’ is accurate?",
    "Does the statement ’[statement]’ accurately reflect the truth?",
)
"""Rewordings used by the paraphrase self-consistency check; [statement] is replaced by the query."""


@dataclass(frozen=True)
class DetectorConfig:
    """Models, generation lengths, rounds, weights and thresholds of the detector."""

    llm_model: str = "meta-llama/Llama-2-7b-chat-hf"
    nli_model: str = "microsoft/deberta-v3-large"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    max_new_tokens: int = 200
    """Tokens for an answer, a verification, a refinement and a perturbed answer."""
    short_new_tokens: int = 50
    """Tokens for the yes/no internal-contradiction check."""
    context_new_tokens: int = 100
    """Tokens for the in-context self-verification."""

    verification_rounds: int = 2
    refinement_rounds: int = 3
    entropy_samples: int = 5

    weights: MappingProxyType[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS)
    intrinsic_threshold: float = 0.5
    """Below this intrinsic score the detector retrieves evidence; at or above it, NLI verification counts as 1."""
    low_confidence_below: float = 0.5
    high_confidence_from: float = 0.7

    top_k: int = 5
    """Documents kept by the dense retriever."""
    similarity_threshold: float = 0.5
    """Cosine similarity above which two facts are linked in the fact graph."""
    min_fact_length: int = 21
    """Shortest sentence (in characters, after stripping) that counts as a fact."""

    max_wikipedia_lookups: int | None = None
    """Cap on Wikipedia page lookups per query; None looks up every query chunk, as in the paper."""
    search_timeout: float = 10.0
    """Seconds before a Google or Google Scholar request (SerpAPI) is abandoned."""
    search_results: int = 5
