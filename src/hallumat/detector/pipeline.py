"""HalluMatDetector: answer a query, check the answer, and score how reliable it is.

1. The LLM answers the query without retrieval.
2. Intrinsic checks (`intrinsic.evaluate`): verification rounds, NLI confidence and entropy, self-refinement,
   an internal-contradiction check and in-context verification.
3. If the intrinsic score (NLI verification taken as 1) is below the threshold, evidence is retrieved
   (`retrieval.retrieve`) and NLI verification becomes the fraction of top-k documents that entail the answer.
4. The reliability score combines both; it is read as low, medium or high confidence.
5. The sentences of the refined answer form a fact graph, grouped into communities.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hallumat.detector import extrinsic, graph, intrinsic, plots, retrieval, scoring
from hallumat.detector.config import DetectorConfig
from hallumat.detector.intrinsic import IntrinsicScores
from hallumat.detector.models import Embedder, NLIModel, TextGenerator
from hallumat.detector.retrieval import Document, EvidenceSource

log = logging.getLogger(__name__)


@dataclass
class DetectionReport:
    """Everything the detector found for one query."""

    query: str
    answer: str
    intrinsic: IntrinsicScores
    intrinsic_score: float
    refined_answer: str
    retrieved: bool
    documents: list[Document]
    evidence: list[Document]
    nli_verification: float
    reliability: float
    confidence: scoring.Confidence
    reasons: list[str]
    facts: list[str]
    communities: dict[int, list[str]] = field(default_factory=dict)
    fact_graph: Any = None


class HalluMatDetector:
    """The multi-stage detector over a text generator, an NLI model, an embedder and evidence sources."""

    def __init__(
        self,
        generator: TextGenerator,
        nli: NLIModel,
        embedder: Embedder,
        sources: Sequence[EvidenceSource] = (),
        config: DetectorConfig | None = None,
    ) -> None:
        self.generator = generator
        self.nli = nli
        self.embedder = embedder
        self.sources = list(sources)
        self.config = config or DetectorConfig()

    @classmethod
    def from_pretrained(cls, config: DetectorConfig | None = None) -> HalluMatDetector:
        """Load the configured Hugging Face models; add Google/Scholar search when $SERPAPI_API_KEY is set."""
        from hallumat.detector.models import HFNLIModel, HFTextGenerator, SentenceEmbedder

        config = config or DetectorConfig()
        sources: list[EvidenceSource] = [retrieval.WikipediaSource(config.max_wikipedia_lookups)]
        serpapi = retrieval.SerpApiSource.from_environment(config.search_results, config.search_timeout)
        if serpapi is None:
            log.info("$%s is not set: retrieving from Wikipedia only", retrieval.SERPAPI_KEY_ENV)
        else:
            sources.append(serpapi)
        return cls(
            HFTextGenerator(config.llm_model, config.max_new_tokens),
            HFNLIModel(config.nli_model),
            SentenceEmbedder(config.embedding_model),
            sources,
            config,
        )

    def detect(self, query: str, seed: int | None = None) -> DetectionReport:
        """Run the full pipeline on query. seed fixes the community detection; generation follows the model's
        own sampling settings."""
        config = self.config
        answer = intrinsic.answer(self.generator, query, config)
        scores, refined = intrinsic.evaluate(self.generator, self.nli, answer, config)
        intrinsic_score = scoring.score(scores, 1.0, config)

        documents: list[Document] = []
        evidence: list[Document] = []
        retrieved = intrinsic_score < config.intrinsic_threshold
        if retrieved:
            log.info(
                "intrinsic score %.2f is below %.2f: retrieving evidence", intrinsic_score, config.intrinsic_threshold
            )
            documents, evidence = retrieval.retrieve(query, self.sources, self.embedder, config.top_k)
            verification = extrinsic.nli_verification(self.nli, answer, [doc.text for doc in evidence])
        else:
            verification = 1.0

        reliability = scoring.score(scores, verification, config)
        level = scoring.confidence_level(reliability, config)
        report = DetectionReport(
            query=query,
            answer=answer,
            intrinsic=scores,
            intrinsic_score=intrinsic_score,
            refined_answer=refined,
            retrieved=retrieved,
            documents=documents,
            evidence=evidence,
            nli_verification=verification,
            reliability=reliability,
            confidence=level,
            reasons=scoring.low_confidence_reasons(scores, verification) if level is scoring.Confidence.LOW else [],
            facts=graph.extract_facts(refined, config.min_fact_length),
        )
        if report.facts:
            report.fact_graph = graph.build_fact_graph(report.facts, self.embedder, config.similarity_threshold)
            graph.assign_communities(report.fact_graph, seed=seed)
            report.communities = graph.facts_by_community(report.fact_graph)
        return report


def paraphrase_check(detector: HalluMatDetector, query: str, answer: str, rng: random.Random | None = None) -> int:
    """Optional extra check: 1 if a reworded query gets the identical answer."""
    return intrinsic.paraphrase_consistency(detector.generator, query, answer, detector.config, rng)


def format_report(report: DetectionReport) -> str:
    """The report as plain text."""
    s = report.intrinsic
    lines = [
        f"Query: {report.query}",
        "",
        "Answer (no retrieval):",
        report.answer,
        "",
        "Intrinsic checks:",
        f"  self-consistency:          {s.self_consistency}",
        f"  confidence:                {s.max_confidence:.2f} (std {s.confidence_std:.2f})",
        f"  entropy:                   {s.avg_entropy:.2f} (std {s.entropy_std:.2f})",
        f"  no internal contradiction: {s.internal_consistency}",
        f"  in-context verification:   {s.in_context_verification}",
        f"  intrinsic score:           {report.intrinsic_score:.2f}",
    ]
    if report.retrieved:
        lines += ["", f"Retrieved {len(report.documents)} document(s); top {len(report.evidence)} used as evidence:"]
        for doc in report.evidence:
            lines.append(f"  [{doc.source}] {doc.link or 'no link'}")
            lines.append(f"    {doc.text[:500]}")
        lines.append(f"NLI verification: {report.nli_verification:.2f}")
    lines += ["", f"Reliability score: {report.reliability:.2f} ({report.confidence.value} confidence)"]
    lines += [f"  - {reason}" for reason in report.reasons]
    if report.communities:
        lines += ["", "Facts of the refined answer, by community:", graph.format_communities(report.communities)]
    else:
        lines += ["", "No sentence of the refined answer was long enough to form a fact graph."]
    return "\n".join(lines)


def save_plots(report: DetectionReport, out: Path, show: bool = False) -> list[Path]:
    """Write fact_graph_3d.png, fact_graph_2d.png and fact_similarity.png under out (those that can be drawn)."""
    if report.fact_graph is None:
        return []
    written = [
        plots.graph_3d(report.fact_graph, out / "fact_graph_3d.png", show),
        plots.graph_2d(report.fact_graph, out / "fact_graph_2d.png", show),
        plots.similarity_heatmap(report.facts, out / "fact_similarity.png", show),
    ]
    return [path for path in written if path is not None]
