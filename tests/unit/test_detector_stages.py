"""Each detector stage on its own, with fake models."""

import random

import numpy as np
import pytest
import scipy.stats

from hallumat.detector import extrinsic, intrinsic, scoring
from hallumat.detector.config import DEFAULT_WEIGHTS, PERTURBATION_TEMPLATES, DetectorConfig
from hallumat.detector.intrinsic import IntrinsicScores
from tests.fakes import FixedNLI, ScriptedGenerator

CONFIG = DetectorConfig()


def test_answer_prompt_and_length():
    generator = ScriptedGenerator(lambda prompt: "answer")
    assert intrinsic.answer(generator, "What is steel?", CONFIG) == "answer"
    assert generator.calls == [("Query: What is steel?", 200)]


def test_chain_of_verification_is_one_only_if_every_round_repeats_the_text():
    echo = ScriptedGenerator(lambda prompt: "  same  ")
    assert intrinsic.chain_of_verification(echo, "same", CONFIG) == 1
    assert len(echo.calls) == CONFIG.verification_rounds
    assert echo.calls[0][0] == "Is the following statement logically sound and factually consistent? 'same'"

    changes = ScriptedGenerator(lambda prompt: "different")
    assert intrinsic.chain_of_verification(changes, "same", CONFIG) == 0
    # The second round verifies the first round's output.
    assert "'different'" in changes.calls[1][0]


def test_confidence_and_entropy_follow_the_nli_probabilities():
    probs = [0.1, 0.2, 0.7]
    nli = FixedNLI(single=probs)
    max_conf, std = intrinsic.confidence(nli, "text")
    assert max_conf == pytest.approx(0.7)
    assert std == pytest.approx(np.std(np.float32(probs)), rel=1e-6)

    mean, spread = intrinsic.entropy(nli, "text", samples=5)
    assert mean == pytest.approx(scipy.stats.entropy(probs), rel=1e-6)
    assert spread == 0.0
    assert len(nli.calls) == 6


def test_self_refine_feeds_each_round_back():
    generator = ScriptedGenerator(lambda prompt: f"<{len(prompt)}>")
    refined = intrinsic.self_refine(generator, "start", CONFIG)
    assert len(generator.calls) == CONFIG.refinement_rounds
    assert "'start'" in generator.calls[0][0]
    assert refined == f"<{len(generator.calls[-1][0])}>"


@pytest.mark.parametrize(("reply", "expected"), [("No.", 1), ("Yes, it does.", 0), ("I know nothing", 1)])
def test_internal_consistency_looks_for_no(reply: str, expected: int):
    generator = ScriptedGenerator(lambda prompt: reply)
    assert intrinsic.internal_consistency(generator, "x", CONFIG) == expected
    assert generator.calls[0][1] == CONFIG.short_new_tokens


@pytest.mark.parametrize(("reply", "expected"), [("It is true.", 1), ("I am NOT SURE.", 0)])
def test_in_context_verification(reply: str, expected: int):
    generator = ScriptedGenerator(lambda prompt: reply)
    assert intrinsic.in_context_verification(generator, "x", CONFIG) == expected
    assert generator.calls == [("Do you know if this statement is true? x", CONFIG.context_new_tokens)]


def test_perturb_query_uses_a_template_and_a_seedable_rng():
    first = intrinsic.perturb_query("steel rusts", random.Random(3))
    assert first == intrinsic.perturb_query("steel rusts", random.Random(3))
    assert any(first == t.replace("[statement]", "steel rusts") for t in PERTURBATION_TEMPLATES)


def test_evaluate_runs_the_checks_in_the_detectors_order():
    generator = ScriptedGenerator(lambda prompt: "No")
    scores, refined = intrinsic.evaluate(generator, FixedNLI(), "answer", CONFIG)
    prompts = [prompt for prompt, _ in generator.calls]
    assert [p.split()[0] for p in prompts] == ["Is", "Is", "Review", "Review", "Review", "Does", "Do"]
    assert refined == "No"
    assert scores.internal_consistency == 1
    assert scores.self_consistency == 0


def test_entailment_label_and_verification_fraction():
    nli = FixedNLI(pair=lambda claim, evidence: (0.9, 0.05, 0.05) if "wrong" in evidence else (0.0, 0.1, 0.9))
    assert extrinsic.entailment_label(nli, "c", "wrong") == "contradiction"
    assert extrinsic.nli_verification(nli, "c", ["ok", "wrong", "ok", "ok"]) == 0.75
    assert extrinsic.nli_verification(nli, "c", []) == 0.0


def scores(**overrides: float) -> IntrinsicScores:
    values = dict(
        self_consistency=1,
        max_confidence=0.9,
        confidence_std=0.1,
        avg_entropy=0.2,
        entropy_std=0.0,
        internal_consistency=1,
        in_context_verification=1,
    )
    values.update(overrides)
    return IntrinsicScores(**values)  # type: ignore[arg-type]


def test_reliability_score_is_the_weighted_sum():
    assert scoring.reliability_score(1, 0, 0, 1, DEFAULT_WEIGHTS) == pytest.approx(1.0)
    assert scoring.reliability_score(0, 1, 1, 0, DEFAULT_WEIGHTS) == pytest.approx(0.0)
    assert scoring.reliability_score(1, 0.5, 0.25, 0.5, DEFAULT_WEIGHTS) == pytest.approx(0.2 + 0.1 + 0.15 + 0.2)
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(("value", "level"), [(0.49, "low"), (0.5, "medium"), (0.69, "medium"), (0.7, "high")])
def test_confidence_levels(value: float, level: str):
    assert scoring.confidence_level(value, CONFIG) == level


def test_low_confidence_reasons():
    assert scoring.low_confidence_reasons(scores(), 1.0) == []
    reasons = scoring.low_confidence_reasons(scores(self_consistency=0, max_confidence=0.3, avg_entropy=0.9), 0.2)
    assert len(reasons) == 4


def test_entailment_uses_the_models_own_label_order():
    nli = FixedNLI(pair=lambda claim, evidence: (0.8, 0.1, 0.1))
    nli.labels = ("entailment", "neutral", "contradiction")
    assert extrinsic.entailment_label(nli, "c", "e") == "entailment"
    assert extrinsic.nli_verification(nli, "c", ["e", "e"]) == 1.0


def test_a_two_class_head_never_counts_as_entailment():
    nli = FixedNLI(pair=lambda claim, evidence: (0.2, 0.8))
    assert extrinsic.entailment_label(nli, "c", "e") == "neutral"
    assert extrinsic.nli_verification(nli, "c", ["e"]) == 0.0
