"""Intrinsic checks: signals computed from the model and its answer alone, without external evidence."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import scipy.stats

from hallumat.detector.config import PERTURBATION_TEMPLATES, DetectorConfig
from hallumat.detector.models import NLIModel, TextGenerator


@dataclass(frozen=True)
class IntrinsicScores:
    """The intrinsic signals of one answer.

    self_consistency, internal_consistency and in_context_verification are 0 or 1. Only self_consistency and the
    two standard deviations enter the reliability score; the others are reported.
    """

    self_consistency: int
    max_confidence: float
    confidence_std: float
    avg_entropy: float
    entropy_std: float
    internal_consistency: int
    in_context_verification: int


def answer(generator: TextGenerator, query: str, config: DetectorConfig) -> str:
    """The model's answer to the query with no retrieved context."""
    return generator(f"Query: {query}", config.max_new_tokens)


def chain_of_verification(generator: TextGenerator, response: str, config: DetectorConfig) -> int:
    """Ask the model to verify the response repeatedly; 1 if every round returns the text unchanged."""
    consistent = True
    current = response
    for _ in range(config.verification_rounds):
        prompt = f"Is the following statement logically sound and factually consistent? '{current}'"
        revised = generator(prompt, config.max_new_tokens)
        if revised.strip() != current.strip():
            consistent = False
        current = revised
    return int(consistent)


def confidence(nli: NLIModel, response: str) -> tuple[float, float]:
    """(max, standard deviation) of the NLI class probabilities of the response."""
    probs = nli.probabilities(response)
    return float(np.max(probs)), float(np.std(probs))


def entropy(nli: NLIModel, response: str, samples: int) -> tuple[float, float]:
    """(mean, standard deviation) of the entropy of the NLI class probabilities over repeated passes."""
    scores = [scipy.stats.entropy(nli.probabilities(response).astype(np.float64)) for _ in range(samples)]
    return float(np.mean(scores)), float(np.std(scores))


def self_refine(generator: TextGenerator, response: str, config: DetectorConfig) -> str:
    """Ask the model to correct the response, feeding each correction back in."""
    refined = response
    for _ in range(config.refinement_rounds):
        prompt = f"Review the following statement and correct any inconsistencies or errors: '{refined}'"
        refined = generator(prompt, config.max_new_tokens)
    return refined


def internal_consistency(generator: TextGenerator, response: str, config: DetectorConfig) -> int:
    """1 if the model's yes/no answer to 'does this contradict itself?' contains 'no'."""
    prompt = f"Does the following statement contain any internal contradictions? '{response}' Answer 'Yes' or 'No'."
    return int("no" in generator(prompt, config.short_new_tokens).lower())


def in_context_verification(generator: TextGenerator, response: str, config: DetectorConfig) -> int:
    """1 unless the model says it is 'not sure' whether the response is true."""
    reply = generator(f"Do you know if this statement is true? {response}", config.context_new_tokens)
    return int("not sure" not in reply.lower())


def perturb_query(query: str, rng: random.Random | None = None) -> str:
    """The query wrapped in a randomly chosen rewording template."""
    template = (rng or random).choice(PERTURBATION_TEMPLATES)
    return template.replace("[statement]", query)


def paraphrase_consistency(
    generator: TextGenerator, query: str, response: str, config: DetectorConfig, rng: random.Random | None = None
) -> int:
    """1 if the answer to a reworded query is identical to the response. Not part of the reliability score."""
    reply = generator(perturb_query(query, rng), config.max_new_tokens)
    return int(reply.strip() == response.strip())


def evaluate(
    generator: TextGenerator, nli: NLIModel, response: str, config: DetectorConfig
) -> tuple[IntrinsicScores, str]:
    """Run the intrinsic checks in the detector's order; return the scores and the self-refined answer."""
    self_consistency = chain_of_verification(generator, response, config)
    max_confidence, confidence_std = confidence(nli, response)
    avg_entropy, entropy_std = entropy(nli, response, config.entropy_samples)
    refined = self_refine(generator, response, config)
    scores = IntrinsicScores(
        self_consistency=self_consistency,
        max_confidence=max_confidence,
        confidence_std=confidence_std,
        avg_entropy=avg_entropy,
        entropy_std=entropy_std,
        internal_consistency=internal_consistency(generator, refined, config),
        in_context_verification=in_context_verification(generator, response, config),
    )
    return scores, refined
