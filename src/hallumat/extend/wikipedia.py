"""Wikipedia passages about materials science.

`crawl_categories` walks the category tree under a set of root categories (breadth first, to a depth limit,
skipping categories about people, organisations, lists and the like). `fetch_leads` downloads the lead section of
each article as plain text, 20 articles per request. `passages` splits the leads into passages of at most about
200 words on sentence boundaries.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from collections.abc import Iterator, Sequence
from typing import Any, Final

from hallumat._optional import require
from hallumat.extend.records import Record, stable_id

log = logging.getLogger(__name__)

API: Final = "https://en.wikipedia.org/w/api.php"
USER_AGENT: Final = "HalluMat/2.0 (https://github.com/bhanuprakashvangala/HalluMat)"

ROOT_CATEGORIES: Final = (
    "Materials science",
    "Materials",
    "Metallurgy",
    "Alloys",
    "Ceramic materials",
    "Polymers",
    "Semiconductor materials",
    "Composite materials",
    "Crystallography",
    "Nanomaterials",
    "Corrosion",
    "Glass",
    "Minerals",
    "Inorganic compounds",
    "Superconductors",
    "Magnetic materials",
    "Building materials",
    "Thin films",
    "Solid-state chemistry",
    "Condensed matter physics",
)

SKIP_CATEGORY: Final = re.compile(
    r"people|scientists|engineers|chemists|physicists|metallurgists|companies|manufacturers|organi[sz]ations|"
    r"societies|journals|awards|lists?\b|museums|by country|by city|births|deaths|history of|stubs|templates|"
    r"images|wikipedia|redirects|disambiguation|fiction|films|songs|albums|brands|products|economy|trade|"
    r"mines\b|mining companies|locations|geography|of the united states|in the united states",
    re.IGNORECASE,
)
SKIP_TITLE: Final = re.compile(r"^(list of|lists of|index of|outline of|glossary of|timeline of)", re.IGNORECASE)

MIN_WORDS: Final = 60
MAX_WORDS: Final = 200


class WikiClient:
    """A polite MediaWiki API client: descriptive User-Agent, maxlag, and retries with back-off."""

    def __init__(self, session: Any = None, pause: float = 0.1) -> None:
        if session is None:
            session = require("requests", "requests", "detector").Session()
            session.headers["User-Agent"] = USER_AGENT
        self.session = session
        self.pause = pause

    def query(self, **params: Any) -> Iterator[dict[str, Any]]:
        """Every page of a query, following 'continue'."""
        params = {"action": "query", "format": "json", "formatversion": 2, "maxlag": 5, **params}
        cont: dict[str, Any] = {}
        while True:
            data = self._get({**params, **cont})
            yield data
            if "continue" not in data:
                return
            cont = data["continue"]

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(6):
            try:
                response = self.session.get(API, params=params, timeout=60)
                if response.status_code == 200:
                    data: dict[str, Any] = response.json()
                    if data.get("error", {}).get("code") != "maxlag":
                        time.sleep(self.pause)
                        return data
            except Exception as exc:
                log.warning("Wikipedia request failed (%s); retrying", exc)
            time.sleep(min(60, 2**attempt))
        raise RuntimeError(f"Wikipedia request kept failing: {params}")


def crawl_categories(client: WikiClient, roots: Sequence[str] = ROOT_CATEGORIES, depth: int = 3) -> list[str]:
    """Article titles in the roots' category trees, to the given depth, in the order found."""
    seen_categories: set[str] = set()
    titles: dict[str, None] = {}
    queue: deque[tuple[str, int]] = deque((root, 0) for root in roots)
    while queue:
        category, level = queue.popleft()
        if category in seen_categories:
            continue
        seen_categories.add(category)
        for data in client.query(
            list="categorymembers", cmtitle=f"Category:{category}", cmlimit=500, cmtype="page|subcat"
        ):
            for member in data.get("query", {}).get("categorymembers", []):
                title = member["title"]
                if member["ns"] == 14:
                    name = title.split(":", 1)[1]
                    if level < depth and not SKIP_CATEGORY.search(name):
                        queue.append((name, level + 1))
                elif member["ns"] == 0 and not SKIP_TITLE.search(title):
                    titles.setdefault(title, None)
        if len(seen_categories) % 100 == 0:
            log.info("%d categories visited, %d articles found", len(seen_categories), len(titles))
    return list(titles)


def fetch_leads(client: WikiClient, titles: Sequence[str], batch: int = 20) -> Iterator[Record]:
    """{title, url, text} for the lead section of every existing article."""
    for start in range(0, len(titles), batch):
        chunk = titles[start : start + batch]
        for data in client.query(
            prop="extracts|info", titles="|".join(chunk), exintro=1, explaintext=1, exlimit=batch, inprop="url"
        ):
            for page in data.get("query", {}).get("pages", []):
                text = (page.get("extract") or "").strip()
                if text and not page.get("missing"):
                    yield {"title": page["title"], "url": page.get("fullurl", ""), "text": text}


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def passages(article: Record) -> list[Record]:
    """The article's lead split into passages of MIN_WORDS to about MAX_WORDS words on sentence boundaries."""
    text = re.sub(r"\s+", " ", article["text"]).strip()
    chunks: list[list[str]] = [[]]
    for sentence in _SENTENCE.split(text):
        words = sum(len(s.split()) for s in chunks[-1])
        if chunks[-1] and words + len(sentence.split()) > MAX_WORDS:
            chunks.append([])
        chunks[-1].append(sentence)
    result = []
    for index, chunk in enumerate(chunks):
        body = " ".join(chunk)
        if len(body.split()) >= MIN_WORDS:
            result.append(
                {
                    "id": stable_id("wp", article["title"], str(index)),
                    "title": article["title"],
                    "url": article["url"],
                    "text": body,
                }
            )
    return result
