"""Questions written from Wikipedia passages, kept only when the answer is stated in the passage."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Final

from hallumat.extend.llm import Message, parse_json
from hallumat.extend.records import Record, normalise, stable_id

PROMPT: Final = """You are writing quiz questions about materials science from an encyclopedia passage.

Article: {title}
Passage:
\"\"\"{text}\"\"\"

Write up to {n} questions that a materials scientist could answer, each answered by a fact stated in the passage.
Rules:
- Each question must stand on its own: name the material, compound or concept explicitly. Never write "the passage",
  "the article", "this material", "it" or "they" in a question.
- Ask about one fact per question (a property, structure, use, composition, value, process or relationship).
- "short_answer" must be copied word for word from the passage (a few words or a number with its unit).
- "answer" is one complete sentence that answers the question using only the passage.
- Do not ask about people, companies, dates of discovery, history or etymology.
- If the passage is not about materials science, chemistry or physics of materials, return [].

Return only a JSON array: [{{"question": "...", "short_answer": "...", "answer": "..."}}]"""

VAGUE: Final = re.compile(
    r"\b(the passage|the article|the text|this material|this compound|this substance|mentioned above)\b", re.I
)


def conversation(passage: Record, n: int = 3) -> list[Message]:
    return [{"role": "user", "content": PROMPT.format(title=passage["title"], text=passage["text"], n=n)}]


def accept(item: Any, passage: Record) -> Record | None:
    """The question as a record, or None if it is malformed, vague, gives its answer away, or is not grounded in the
    passage."""
    if not isinstance(item, dict):
        return None
    question = str(item.get("question", "")).strip()
    short = str(item.get("short_answer", "")).strip()
    answer = str(item.get("answer", "")).strip()
    if not (question.endswith("?") and 6 <= len(question.split()) <= 40 and short and answer):
        return None
    if VAGUE.search(question) or len(answer.split()) > 60:
        return None
    norm_short, norm_text = normalise(short), normalise(passage["text"])
    if not norm_short or norm_short not in norm_text or norm_short in normalise(question):
        return None
    if norm_short not in normalise(answer):
        return None
    return {
        "id": stable_id("wp", passage["id"], normalise(question)),
        "source": "wikipedia",
        "topic": "encyclopedic",
        "subject": passage["title"],
        "question": question,
        "reference_answer": answer,
        "short_answer": short,
        "source_id": passage["id"],
        "source_url": passage["url"],
        "keep_terms": [],
    }


def parse_questions(reply: str, passage: Record) -> list[Record]:
    data = parse_json(reply)
    if not isinstance(data, list):
        return []
    return [q for q in (accept(item, passage) for item in data) if q is not None]


def keep_terms(question: str, extra: Sequence[str] = ()) -> list[str]:
    """Terms a paraphrase must keep: numbers with units, chemical formulas and any extra terms."""
    terms = list(extra)
    terms += re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:°C|K|GPa|MPa|eV|nm|g/cm³|%)?", question)
    terms += [t for t in re.findall(r"\b(?:[A-Z][a-z]?\d*){2,}\b", question) if any(c.isdigit() for c in t)]
    return sorted({t.strip() for t in terms if t.strip()})
