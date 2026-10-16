"""The reliability score and how it is read."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from hallumat.detector.config import DetectorConfig
from hallumat.detector.intrinsic import IntrinsicScores


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def reliability_score(
    self_consistency: float,
    confidence_std: float,
    entropy_std: float,
    nli_verification: float,
    weights: Mapping[str, float],
) -> float:
    """Weighted sum of self-consistency, 1 - confidence std, 1 - entropy std and NLI verification."""
    return (
        self_consistency * weights["self_consistency"]
        + (1 - confidence_std) * weights["confidence"]
        + (1 - entropy_std) * weights["entropy"]
        + nli_verification * weights["nli_verification"]
    )


def score(intrinsic: IntrinsicScores, nli_verification: float, config: DetectorConfig) -> float:
    return reliability_score(
        intrinsic.self_consistency, intrinsic.confidence_std, intrinsic.entropy_std, nli_verification, config.weights
    )


def confidence_level(reliability: float, config: DetectorConfig) -> Confidence:
    if reliability < config.low_confidence_below:
        return Confidence.LOW
    if reliability < config.high_confidence_from:
        return Confidence.MEDIUM
    return Confidence.HIGH


def low_confidence_reasons(intrinsic: IntrinsicScores, nli_verification: float) -> list[str]:
    """Why an answer scored low, one sentence per failing signal."""
    reasons = []
    if intrinsic.self_consistency == 0:
        reasons.append("The model's answer changed when it was asked to verify it.")
    if intrinsic.max_confidence < 0.5:
        reasons.append("The model's confidence in its answer was low.")
    if intrinsic.avg_entropy > 0.7:
        reasons.append("The answer had high entropy, which indicates uncertainty.")
    if nli_verification < 0.5:
        reasons.append("Little retrieved evidence entails the answer (low NLI verification).")
    return reasons
