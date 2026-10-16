"""The build stages over one work directory. Every stage appends to its output and skips work already done, so a
preempted job can simply be started again.

Work directory layout:
    mp_materials.jsonl, wiki_passages.jsonl        fetched sources
    candidates_mp.jsonl, candidates_wiki.jsonl     candidate questions
    questions.jsonl                                the selected questions
    phrasings.jsonl                                five phrasings per question
    responses/<model>.jsonl                        answers, one file per model (or per shard)
    labels/<model>.jsonl                           similarity and level per answer
    export/                                        the published tables
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Final

from hallumat.extend import materials_project, paraphrase, questions, select
from hallumat.extend.llm import ChatBackend, Sampling
from hallumat.extend.records import Record, append_jsonl, done_ids, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

RESPONDERS: Final = {
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2-9b": "google/gemma-2-9b-it",
}
"""Answering models by short name. Llama-2-7B-chat is the model of the original HalluMat responses."""

WRITER: Final = "Qwen/Qwen2.5-32B-Instruct-AWQ"
"""Model that writes the Wikipedia questions and the paraphrases."""

ANSWER_SAMPLING: Final = Sampling(temperature=0.7, top_p=0.9, max_tokens=256)
WRITER_SAMPLING: Final = Sampling(temperature=0.7, top_p=0.9, max_tokens=1024)


def seed_of(key: str) -> int:
    """A sampling seed derived from a record id, so each generation is reproducible on its own."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


def chunks(items: Sequence[Record], size: int) -> Iterator[Sequence[Record]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def shard(items: Sequence[Record], index: int, count: int) -> list[Record]:
    """Every count-th item starting at index, for splitting a stage across jobs."""
    if not 0 <= index < count:
        raise ValueError(f"shard {index} of {count}")
    return [item for i, item in enumerate(items) if i % count == index]


def mp_questions(work: Path) -> int:
    materials = list(read_jsonl(work / "mp_materials.jsonl"))
    chosen = materials_project.ground_states(materials)
    log.info("%d materials, %d formulas", len(materials), len(chosen))
    return write_jsonl(work / "candidates_mp.jsonl", materials_project.make_questions(chosen))


def wiki_questions(
    work: Path, backend: ChatBackend, chunk_size: int = 2000, shard_index: int = 0, shard_count: int = 1
) -> int:
    """Questions from every passage not yet processed. Passages that yield nothing are recorded too, so they are
    not retried."""
    suffix = "" if shard_count == 1 else f".{shard_index}"
    out, seen_file = work / f"candidates_wiki{suffix}.jsonl", work / f"wiki_done{suffix}.jsonl"
    done = done_ids(seen_file)
    todo = [
        p
        for p in shard(list(read_jsonl(work / "wiki_passages.jsonl")), shard_index, shard_count)
        if p["id"] not in done
    ]
    written = 0
    for batch in chunks(todo, chunk_size):
        replies = backend.chat(
            [questions.conversation(p) for p in batch], WRITER_SAMPLING, [seed_of(p["id"]) for p in batch]
        )
        written += append_jsonl(
            out, (q for p, r in zip(batch, replies, strict=True) for q in questions.parse_questions(r, p))
        )
        append_jsonl(seen_file, ({"id": p["id"]} for p in batch))
        log.info(
            "wiki questions: %d passages left, %d questions written",
            len(todo) - len(done_ids(seen_file)) + len(done),
            written,
        )
    return written


def merge_shards(work: Path, stem: str) -> list[Record]:
    """Records of stem.jsonl and every stem.N.jsonl."""
    records = list(read_jsonl(work / f"{stem}.jsonl"))
    for path in sorted(work.glob(f"{stem}.*.jsonl")):
        records += list(read_jsonl(path))
    return records


def select_questions(work: Path, embedder_factory: Callable[[], object], total: int = 50_000, seed: int = 0) -> int:
    mp = select.drop_exact_duplicates(list(read_jsonl(work / "candidates_mp.jsonl")))
    wiki = select.drop_exact_duplicates(merge_shards(work, "candidates_wiki"))
    wiki = select.drop_near_duplicates(wiki, embedder_factory())  # type: ignore[arg-type]
    chosen = select.choose(mp, wiki, total, seed)
    for record in chosen:
        if not record.get("keep_terms"):
            record["keep_terms"] = questions.keep_terms(record["question"])
    log.info(
        "selected %d questions (%d MP, %d Wikipedia)",
        len(chosen),
        sum(r["source"] == "materials_project" for r in chosen),
        sum(r["source"] == "wikipedia" for r in chosen),
    )
    return write_jsonl(work / "questions.jsonl", chosen)


def paraphrase_questions(
    work: Path,
    backend: ChatBackend,
    chunk_size: int = 2000,
    attempts: int = 3,
    shard_index: int = 0,
    shard_count: int = 1,
) -> int:
    """Five phrasings per question; a question whose rewordings fail validation is retried with new seeds, and
    after `attempts` tries it is recorded in phrasings_failed.jsonl."""
    suffix = "" if shard_count == 1 else f".{shard_index}"
    out, failed_file = work / f"phrasings{suffix}.jsonl", work / f"phrasings_failed{suffix}.jsonl"
    done = done_ids(out, "question_id") | done_ids(failed_file)
    todo = [
        q for q in shard(list(read_jsonl(work / "questions.jsonl")), shard_index, shard_count) if q["id"] not in done
    ]
    for batch in chunks(todo, chunk_size):
        pending = list(batch)
        for attempt in range(attempts):
            if not pending:
                break
            replies = backend.chat(
                [paraphrase.conversation(q) for q in pending],
                WRITER_SAMPLING,
                [seed_of(f"{q['id']}#{attempt}") for q in pending],
            )
            retry = []
            for q, reply in zip(pending, replies, strict=True):
                rewordings = paraphrase.accept(reply, q)
                if rewordings is None:
                    retry.append(q)
                else:
                    append_jsonl(out, paraphrase.phrasings(q, rewordings))
            pending = retry
        append_jsonl(failed_file, ({"id": q["id"]} for q in pending))
        log.info("paraphrase: %d questions left", len(todo) - len(done_ids(out, "question_id")) + len(done))
    return len(done_ids(out, "question_id"))


def respond(
    work: Path, model: str, backend: ChatBackend, chunk_size: int = 5000, shard_index: int = 0, shard_count: int = 1
) -> int:
    """Answer every phrasing with one model; the question is the only message."""
    suffix = "" if shard_count == 1 else f".{shard_index}"
    out = work / "responses" / f"{model}{suffix}.jsonl"
    done = done_ids(out, "phrasing_id")
    rows = shard(merge_shards(work, "phrasings"), shard_index, shard_count)
    todo = [row for row in rows if row["id"] not in done]
    for batch in chunks(todo, chunk_size):
        replies = backend.chat(
            [[{"role": "user", "content": row["text"]}] for row in batch],
            ANSWER_SAMPLING,
            [seed_of(f"{row['id']}|{model}") for row in batch],
        )
        append_jsonl(
            out,
            (
                {"phrasing_id": row["id"], "question_id": row["question_id"], "model": model, "response": reply}
                for row, reply in zip(batch, replies, strict=True)
            ),
        )
        log.info("%s: %d of %d answered", model, len(done_ids(out, "phrasing_id")), len(rows))
    return len(done_ids(out, "phrasing_id"))


def label(work: Path, model: str, embed: Callable[[Sequence[str]], object], chunk_size: int = 20_000) -> int:
    """Similarity and hallucination level of every answer of one model against its reference answer."""
    from hallumat.extend.labels import label_pairs

    references = {q["id"]: q["reference_answer"] for q in read_jsonl(work / "questions.jsonl")}
    out = work / "labels" / f"{model}.jsonl"
    done = done_ids(out, "phrasing_id")
    rows = [r for r in merge_shards(work / "responses", model) if r["phrasing_id"] not in done]
    for batch in chunks(rows, chunk_size):
        results = label_pairs([r["response"] for r in batch], [references[r["question_id"]] for r in batch], embed)
        append_jsonl(
            out,
            (
                {"phrasing_id": r["phrasing_id"], "model": model, "similarity": sim, "level": level}
                for r, (sim, level) in zip(batch, results, strict=True)
            ),
        )
        log.info("%s: %d labelled", model, len(done_ids(out, "phrasing_id")))
    return len(done_ids(out, "phrasing_id"))
