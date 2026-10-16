"""Materials Project facts, turned into questions with exact reference answers.

`fetch_materials` pages through the summary endpoint for materials matched to an experimental (ICSD) structure.
`ground_states` keeps one entry per formula, the one lowest above the convex hull, so a question can name the
formula alone ("What is the band gap of GaN?"). `make_questions` asks one property per formula, cycling through
the properties so they stay balanced, and states the database value in the reference answer.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

from hallumat._optional import require
from hallumat.errors import HallumatError
from hallumat.extend.records import Record, stable_id

log = logging.getLogger(__name__)

SUMMARY_URL: Final = "https://api.materialsproject.org/materials/summary/"
KEY_ENV: Final = "MP_API_KEY"
FIELDS: Final = (
    "material_id",
    "formula_pretty",
    "nelements",
    "symmetry",
    "band_gap",
    "is_gap_direct",
    "density",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "ordering",
    "total_magnetization",
    "bulk_modulus",
    "theoretical",
)
MATERIAL_URL: Final = "https://next-gen.materialsproject.org/materials/{}"

ORDERING_NAMES: Final = {
    "NM": "non-magnetic",
    "FM": "ferromagnetic",
    "FiM": "ferrimagnetic",
    "AFM": "antiferromagnetic",
}


def api_key() -> str:
    """The Materials Project API key from $MP_API_KEY or ~/.mp_api_key."""
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        path = os.path.expanduser("~/.mp_api_key")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                key = handle.read().strip()
    if not key:
        raise HallumatError(f"no Materials Project API key: set ${KEY_ENV} or write it to ~/.mp_api_key")
    return key


def fetch_materials(key: str, page_size: int = 1000, limit: int | None = None, session: Any = None) -> Iterator[Record]:
    """Summary documents of non-deprecated materials with an experimental structure, page by page."""
    http = session or require("requests", "requests", "detector")
    skip = 0
    while limit is None or skip < limit:
        size = page_size if limit is None else min(page_size, limit - skip)
        response = http.get(
            SUMMARY_URL,
            params={
                "_fields": ",".join(FIELDS),
                "_limit": size,
                "_skip": skip,
                "theoretical": "false",
                "deprecated": "false",
            },
            headers={"X-API-KEY": key, "accept": "application/json"},
            timeout=120,
        )
        if response.status_code != 200:
            raise HallumatError(f"Materials Project returned HTTP {response.status_code}: {response.text[:200]}")
        page = response.json().get("data", [])
        yield from page
        log.info("fetched %d Materials Project documents", skip + len(page))
        if len(page) < size:
            return
        skip += size


def ground_states(materials: Sequence[Record]) -> list[Record]:
    """One material per formula: the lowest energy above hull, ties broken by material id."""
    best: dict[str, Record] = {}
    for doc in materials:
        formula = doc.get("formula_pretty")
        if not formula or doc.get("energy_above_hull") is None:
            continue
        key = (doc["energy_above_hull"], str(doc["material_id"]))
        current = best.get(formula)
        if current is None or key < (current["energy_above_hull"], str(current["material_id"])):
            best[formula] = doc
    return [best[formula] for formula in sorted(best)]


@dataclass(frozen=True)
class Fact:
    prop: str
    question: str
    answer: str


def _crystal(doc: Record, f: str) -> Fact | None:
    sym = doc.get("symmetry") or {}
    system, group = sym.get("crystal_system"), sym.get("symbol")
    if not system or not group:
        return None
    return Fact(
        "crystal_structure",
        f"What crystal system does {f} crystallize in?",
        f"{f} crystallizes in the {system.lower()} crystal system, with space group {group}.",
    )


def _band_gap(doc: Record, f: str) -> Fact | None:
    gap = doc.get("band_gap")
    if gap is None:
        return None
    if gap < 0.005:
        answer = f"{f} has no band gap; it is computed to be metallic (0 eV)."
    else:
        kind = "direct" if doc.get("is_gap_direct") else "indirect"
        answer = f"The computed band gap of {f} is {gap:.2f} eV, and it is {kind}."
    return Fact("band_gap", f"What is the band gap of {f}?", answer)


def _density(doc: Record, f: str) -> Fact | None:
    density = doc.get("density")
    if density is None:
        return None
    return Fact("density", f"What is the density of {f}?", f"The density of {f} is {density:.2f} g/cm³.")


def _formation(doc: Record, f: str) -> Fact | None:
    energy = doc.get("formation_energy_per_atom")
    if energy is None or doc.get("nelements", 1) < 2:
        return None
    return Fact(
        "formation_energy",
        f"What is the formation energy per atom of {f}?",
        f"The computed formation energy of {f} is {energy:.3f} eV per atom.",
    )


def _stability(doc: Record, f: str) -> Fact | None:
    above = doc.get("energy_above_hull")
    if above is None:
        return None
    if doc.get("is_stable") or above < 1e-6:
        answer = (
            f"Yes. {f} lies on the convex hull (0 eV per atom above it), "
            "so it is predicted to be thermodynamically stable."
        )
    else:
        answer = (
            f"No. {f} lies {above:.3f} eV per atom above the convex hull, "
            "so it is predicted to be metastable rather than the most stable phase."
        )
    return Fact("stability", f"Is {f} thermodynamically stable?", answer)


def _magnetism(doc: Record, f: str) -> Fact | None:
    name = ORDERING_NAMES.get(str(doc.get("ordering")))
    if name is None:
        return None
    answer = f"{f} is predicted to be {name}."
    moment = doc.get("total_magnetization")
    if name in ("ferromagnetic", "ferrimagnetic") and moment:
        answer = f"{f} is predicted to be {name}, with a total magnetization of {moment:.2f} μB per formula unit."
    return Fact(
        "magnetic_ordering", f"Is {f} magnetic, and if so, what kind of magnetic ordering does it have?", answer
    )


def _bulk_modulus(doc: Record, f: str) -> Fact | None:
    modulus = doc.get("bulk_modulus")
    value = modulus.get("vrh") if isinstance(modulus, dict) else None
    if value is None or value <= 0:
        return None
    return Fact(
        "bulk_modulus",
        f"What is the bulk modulus of {f}?",
        f"The bulk modulus of {f} is about {value:.0f} GPa (Voigt-Reuss-Hill average).",
    )


PROPERTIES: Final = (_band_gap, _crystal, _density, _formation, _stability, _magnetism, _bulk_modulus)


def make_questions(materials: Sequence[Record]) -> list[Record]:
    """One question per formula, cycling through the properties; a formula lacking the property in turn gets the
    next one it has."""
    questions = []
    for index, doc in enumerate(materials):
        formula = str(doc["formula_pretty"])
        for offset in range(len(PROPERTIES)):
            fact = PROPERTIES[(index + offset) % len(PROPERTIES)](doc, formula)
            if fact is not None:
                break
        else:
            continue
        questions.append(
            {
                "id": stable_id("mp", formula, fact.prop),
                "source": "materials_project",
                "topic": fact.prop,
                "subject": formula,
                "question": fact.question,
                "reference_answer": fact.answer,
                "source_id": str(doc["material_id"]),
                "source_url": MATERIAL_URL.format(doc["material_id"]),
                "keep_terms": [formula],
                "n_elements": doc.get("nelements"),
            }
        )
    return questions
