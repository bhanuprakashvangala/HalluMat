"""Extrinsic verification: does retrieved evidence entail the answer?"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hallumat.detector.models import NLIModel


def entailment_label(nli: NLIModel, claim: str, evidence: str) -> str:
    """The most probable NLI label of the pair (claim, evidence), by the model's own label names."""
    index = int(np.argmax(nli.probabilities(claim, evidence)))
    return nli.labels[index] if index < len(nli.labels) else f"label_{index}"


def nli_verification(nli: NLIModel, claim: str, evidence: Sequence[str]) -> float:
    """Fraction of the evidence texts whose label is entailment; 0 when there is no evidence."""
    if not evidence:
        return 0.0
    return sum(entailment_label(nli, claim, text) == "entailment" for text in evidence) / len(evidence)
