"""The published tables of the extended dataset.

export/questions.parquet     one row per question, with its five phrasings and its source
export/questions.csv         the same, for the GitHub repository
export/responses/*.parquet   one row per (phrasing, model): the answer, its similarity and hallucination level
export/summary.csv           per model and source: answers, Low/Medium/High counts and mean similarity
export/phcs.csv              per model and source: mean and share of non-zero PHCS over questions
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from hallumat.dataset import LEVELS, level_codes
from hallumat.errors import HallumatError
from hallumat.extend.records import read_jsonl
from hallumat.extend.stages import RESPONDERS, merge_shards

log = logging.getLogger(__name__)

ROWS_PER_FILE = 200_000

QUESTION_COLUMNS = [
    "question_id",
    "source",
    "topic",
    "subject",
    "question",
    "phrasings",
    "reference_answer",
    "short_answer",
    "source_id",
    "source_url",
]
RESPONSE_COLUMNS = [
    "id",
    "question_id",
    "phrasing",
    "question",
    "model",
    "response",
    "reference_answer",
    "similarity",
    "hallucination_level",
]


def question_table(work: Path) -> pd.DataFrame:
    """Questions that have all five phrasings."""
    questions = pd.DataFrame(read_jsonl(work / "questions.jsonl")).rename(columns={"id": "question_id"})
    phrasings = pd.DataFrame(merge_shards(work, "phrasings"))
    if questions.empty or phrasings.empty:
        raise HallumatError(f"no questions or phrasings in {work}")
    grouped = phrasings.sort_values(["question_id", "phrasing"]).groupby("question_id")["text"].agg(list)
    questions["phrasings"] = questions["question_id"].map(grouped)
    questions = questions[questions["phrasings"].map(lambda p: isinstance(p, list) and len(p) == 5)]
    if "short_answer" not in questions:
        questions["short_answer"] = None
    return questions[QUESTION_COLUMNS].sort_values("question_id").reset_index(drop=True)


def response_table(work: Path, model: str, questions: pd.DataFrame) -> pd.DataFrame:
    responses = pd.DataFrame(merge_shards(work / "responses", model))
    labels = pd.DataFrame(read_jsonl(work / "labels" / f"{model}.jsonl"))
    phrasings = pd.DataFrame(merge_shards(work, "phrasings")).rename(columns={"id": "phrasing_id", "text": "question"})
    if responses.empty or labels.empty:
        return pd.DataFrame(columns=RESPONSE_COLUMNS)
    table = (
        responses.drop_duplicates("phrasing_id")
        .merge(labels[["phrasing_id", "similarity", "level"]].drop_duplicates("phrasing_id"), on="phrasing_id")
        .merge(phrasings[["phrasing_id", "phrasing", "question"]], on="phrasing_id")
        .merge(questions[["question_id", "reference_answer"]], on="question_id")
        .rename(columns={"level": "hallucination_level"})
    )
    table["id"] = table["phrasing_id"] + "|" + model
    return table[RESPONSE_COLUMNS]


def summarise(responses: pd.DataFrame, questions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = responses.merge(questions[["question_id", "source"]], on="question_id")
    counts = data.pivot_table(
        index=["model", "source"], columns="hallucination_level", values="id", aggfunc="count", fill_value=0
    ).reindex(columns=list(LEVELS), fill_value=0)
    summary = counts.assign(
        answers=counts.sum(axis=1), mean_similarity=data.groupby(["model", "source"])["similarity"].mean().round(4)
    ).reset_index()
    data["code"] = level_codes(data["hallucination_level"], start=1)
    per_question = data.groupby(["model", "source", "question_id"])["code"].std()
    phcs = (
        per_question.groupby(["model", "source"])
        .agg(mean_phcs="mean", inconsistent_share=lambda s: float((s > 0).mean()))
        .round(4)
        .reset_index()
    )
    return summary, phcs


def export(work: Path, out: Path) -> int:
    """Write the tables and return the number of answers exported."""
    out.mkdir(parents=True, exist_ok=True)
    questions = question_table(work)
    questions.to_parquet(out / "questions.parquet", index=False)
    questions.assign(phrasings=questions["phrasings"].map(" || ".join)).to_csv(
        out / "questions.csv", index=False, lineterminator="\n"
    )
    frames = [response_table(work, model, questions) for model in RESPONDERS]
    responses = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    responses = responses.sort_values(["question_id", "phrasing", "model"]).reset_index(drop=True)
    directory = out / "responses"
    directory.mkdir(exist_ok=True)
    for old in directory.glob("*.parquet"):
        old.unlink()
    n_files = max(1, -(-len(responses) // ROWS_PER_FILE))
    for i in range(n_files):
        part = responses.iloc[i * ROWS_PER_FILE : (i + 1) * ROWS_PER_FILE]
        part.to_parquet(directory / f"test-{i:05d}-of-{n_files:05d}.parquet", index=False)
    summary, phcs = summarise(responses, questions)
    summary.to_csv(out / "summary.csv", index=False, lineterminator="\n")
    phcs.to_csv(out / "phcs.csv", index=False, lineterminator="\n")
    log.info("exported %d questions and %d answers to %s", len(questions), len(responses), out)
    return len(responses)
