"""HalluMatDetector end to end with fake models."""

from pathlib import Path

import pytest

from hallumat.detector import DetectorConfig, HalluMatDetector, format_report, save_plots
from hallumat.detector import pipeline as pipeline_module
from hallumat.detector.retrieval import Document
from tests.fakes import FixedNLI, KeywordEmbedder, ScriptedGenerator

REFINED = "Steel is an alloy of iron and carbon. Steel rusts when iron oxidises in water. Ok."


# With the default weights the intrinsic score (NLI verification taken as 1) cannot fall below about 0.7, so
# retrieval needs a raised threshold to run; the tests of that path set one.
RETRIEVE = DetectorConfig(intrinsic_threshold=0.99)


def echo_then_refine(prompt: str) -> str:
    """Answers with the prompt, repeats it unchanged when asked to verify (self-consistent), refines to REFINED."""
    if prompt.startswith("Query:"):
        return prompt
    if prompt.startswith("Is the following"):
        return "Query: steel"
    if prompt.startswith("Review"):
        return REFINED
    return "No"


def detector(nli: FixedNLI, sources=(), generator=None, config: DetectorConfig | None = None) -> HalluMatDetector:
    return HalluMatDetector(
        generator or ScriptedGenerator(echo_then_refine),
        nli,
        KeywordEmbedder(["steel", "iron", "alloy", "rusts", "water"]),
        sources,
        config or DetectorConfig(),
    )


def test_default_intrinsic_score_never_drops_below_the_retrieval_threshold():
    # Worst case: inconsistent, the flattest possible NLI spread is still >= 0, entropy std 0.
    worst = DetectorConfig().weights
    floor = 0 * worst["self_consistency"] + (1 - 0.4714) * worst["confidence"] + 1 * worst["entropy"] + 1 * 0.4
    assert floor > DetectorConfig().intrinsic_threshold


def test_a_confident_answer_skips_retrieval():
    pytest.importorskip("networkx")
    pytest.importorskip("community")
    report = detector(FixedNLI(single=(0.0, 0.0, 1.0))).detect("steel", seed=0)
    assert report.intrinsic.self_consistency == 1
    assert report.intrinsic_score >= 0.5
    assert not report.retrieved
    assert report.nli_verification == 1.0
    assert report.reliability == pytest.approx(report.intrinsic_score)
    assert report.confidence == "high"
    assert report.reasons == []
    assert report.facts == ["Steel is an alloy of iron and carbon", "Steel rusts when iron oxidises in water"]
    assert sum(len(facts) for facts in report.communities.values()) == 2


class StaticSource:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.queries: list[str] = []

    def fetch(self, query, ranked_chunks):
        self.queries.append(query)
        return self.docs


def test_a_doubtful_answer_is_checked_against_retrieved_evidence():
    pytest.importorskip("rank_bm25")
    pytest.importorskip("faiss")
    pytest.importorskip("community")
    source = StaticSource([Document("steel alloy iron", "Wikipedia", "w1"), Document("water", "Wikipedia", "w2")])
    nli = FixedNLI(single=(0.4, 0.3, 0.3), pair=lambda claim, ev: (0, 0, 1) if "steel" in ev else (1, 0, 0))
    generator = ScriptedGenerator(lambda prompt: "changes every time" if prompt.startswith("Is") else REFINED)
    report = detector(nli, [source], generator, RETRIEVE).detect("steel alloy", seed=0)
    assert report.retrieved
    assert source.queries == ["steel alloy"]
    assert len(report.evidence) == 2
    assert report.nli_verification == 0.5
    assert report.reliability < report.intrinsic_score
    text = format_report(report)
    assert "Retrieved 2 document(s)" in text
    assert "NLI verification: 0.50" in text


def test_no_evidence_means_zero_verification_and_reasons(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pipeline_module.retrieval, "retrieve", lambda query, sources, embedder, k: ([], []))
    generator = ScriptedGenerator(lambda prompt: {"Does": "Yes", "Is": "changed"}.get(prompt.split()[0], "x"))
    report = detector(FixedNLI(single=(0.4, 0.3, 0.3)), generator=generator, config=RETRIEVE).detect("q")
    assert report.retrieved
    assert report.nli_verification == 0.0
    assert report.confidence == "low"
    assert any("NLI verification" in reason for reason in report.reasons)
    assert report.facts == []
    assert report.fact_graph is None
    assert "No sentence of the refined answer" in format_report(report)


def test_save_plots_writes_the_three_figures(tmp_path: Path):
    pytest.importorskip("networkx")
    pytest.importorskip("community")
    pytest.importorskip("scipy")
    report = detector(FixedNLI(single=(0.0, 0.0, 1.0))).detect("steel", seed=0)
    written = save_plots(report, tmp_path)
    assert sorted(path.name for path in written) == ["fact_graph_2d.png", "fact_graph_3d.png", "fact_similarity.png"]
    assert all(path.stat().st_size > 1000 for path in written)


def test_save_plots_without_a_graph_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pipeline_module.retrieval, "retrieve", lambda *args: ([], []))
    report = detector(
        FixedNLI(single=(0.4, 0.3, 0.3)), generator=ScriptedGenerator(lambda p: "x"), config=RETRIEVE
    ).detect("q")
    assert save_plots(report, tmp_path) == []
