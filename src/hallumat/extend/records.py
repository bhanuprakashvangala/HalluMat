"""JSON-lines files, stable ids and text normalisation shared by the build stages."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def read_jsonl(path: Path) -> Iterator[Record]:
    """The records of a JSON-lines file; a missing file has none. A truncated last line (an interrupted write) is
    skipped."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def append_jsonl(path: Path, records: Iterable[Record]) -> int:
    """Append records to path (created with its directory if needed) and return how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
    return count


def write_jsonl(path: Path, records: Iterable[Record]) -> int:
    """Write records to path, replacing it atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    count = append_jsonl(tmp, records)
    tmp.replace(path)
    return count


def done_ids(path: Path, key: str = "id") -> set[str]:
    """The ids already written to path, for resuming a stage."""
    return {str(record[key]) for record in read_jsonl(path) if key in record}


def normalise(text: str) -> str:
    """Lower case, NFKC, punctuation dropped and whitespace collapsed; for comparing and de-duplicating text."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s.+-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_id(prefix: str, *parts: str) -> str:
    """A short id that depends only on its parts, e.g. 'mp-3f2a91c07d'."""
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"
