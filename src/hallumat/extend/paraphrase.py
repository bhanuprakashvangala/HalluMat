"""Four rewordings of each question, so every question is asked five ways (the original first)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from hallumat.extend.llm import Message, parse_json
from hallumat.extend.records import Record, normalise

PHRASINGS: Final = 5

PROMPT: Final = """Rewrite the question below in {n} different ways. Each rewording must ask for exactly the same
information and be answerable with the same answer. Vary the wording and sentence structure. Keep every material
name, chemical formula, number and unit exactly as written{terms}. Do not add hints or extra conditions.

Question: {question}

Return only a JSON array of {n} strings."""


def conversation(question: Record, n: int = PHRASINGS - 1) -> list[Message]:
    terms = question.get("keep_terms") or []
    listed = f" (in particular: {', '.join(terms)})" if terms else ""
    return [{"role": "user", "content": PROMPT.format(n=n, question=question["question"], terms=listed)}]


def accept(reply: str, question: Record, n: int = PHRASINGS - 1) -> list[str] | None:
    """n distinct rewordings that keep the required terms and differ from the original, or None."""
    data = parse_json(reply)
    if not isinstance(data, list):
        return None
    original = normalise(question["question"])
    seen = {original}
    kept: list[str] = []
    for item in data:
        text = str(item).strip()
        key = normalise(text)
        if not text.endswith("?") or key in seen or not 4 <= len(text.split()) <= 60:
            continue
        if any(term not in text for term in question.get("keep_terms") or []):
            continue
        seen.add(key)
        kept.append(text)
    return kept[:n] if len(kept) >= n else None


def phrasings(question: Record, rewordings: Sequence[str]) -> list[Record]:
    """The question's five rows: phrasing 0 is the original."""
    texts = [question["question"], *rewordings]
    return [
        {"id": f"{question['id']}-{i}", "question_id": question["id"], "phrasing": i, "text": text}
        for i, text in enumerate(texts)
    ]
