"""The dataset-extension stages with a scripted LLM and fake embedders (no GPU, no network)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hallumat.extend import materials_project as mp
from hallumat.extend import paraphrase, questions, select, stages, wikipedia
from hallumat.extend.labels import cosine_rows, label_pairs
from hallumat.extend.llm import Sampling, parse_json, resolve_seeds
from hallumat.extend.records import append_jsonl, done_ids, normalise, read_jsonl, stable_id, write_jsonl
from tests.fakes import KeywordEmbedder

GAN = {
    "material_id": "mp-804",
    "formula_pretty": "GaN",
    "nelements": 2,
    "symmetry": {"crystal_system": "Hexagonal", "symbol": "P6_3mc"},
    "band_gap": 1.73,
    "is_gap_direct": True,
    "density": 6.08,
    "formation_energy_per_atom": -0.66,
    "energy_above_hull": 0.0,
    "is_stable": True,
    "ordering": "NM",
    "total_magnetization": 0.0,
    "bulk_modulus": {"vrh": 172.0},
}


def test_records_round_trip_and_resume(tmp_path: Path):
    path = tmp_path / "a" / "x.jsonl"
    assert append_jsonl(path, [{"id": "1"}, {"id": "2"}]) == 2
    with path.open("a", encoding="utf-8") as f:
        f.write('{"id": "3"')  # an interrupted write
    assert [r["id"] for r in read_jsonl(path)] == ["1", "2"]
    assert done_ids(path) == {"1", "2"}
    write_jsonl(path, [{"id": "9"}])
    assert list(read_jsonl(path)) == [{"id": "9"}]
    assert list(read_jsonl(tmp_path / "missing.jsonl")) == []


def test_normalise_and_stable_id():
    assert normalise("  What is  GaN's band-gap?? ") == "what is gan s band-gap"
    assert stable_id("mp", "GaN", "band_gap") == stable_id("mp", "GaN", "band_gap")
    assert stable_id("mp", "GaN", "band_gap") != stable_id("mp", "GaN", "density")


def test_ground_states_keep_the_lowest_polymorph():
    docs = [
        {"material_id": "mp-2", "formula_pretty": "TiO2", "energy_above_hull": 0.04},
        {"material_id": "mp-1", "formula_pretty": "TiO2", "energy_above_hull": 0.0},
        {"material_id": "mp-3", "formula_pretty": "Si", "energy_above_hull": 0.0},
        {"material_id": "mp-4", "formula_pretty": "X", "energy_above_hull": None},
    ]
    assert [d["material_id"] for d in mp.ground_states(docs)] == ["mp-3", "mp-1"]


@pytest.mark.parametrize(
    ("prop", "question", "answer"),
    [
        (mp._band_gap, "What is the band gap of GaN?", "The computed band gap of GaN is 1.73 eV, and it is direct."),
        (
            mp._crystal,
            "What crystal system does GaN crystallize in?",
            "hexagonal crystal system, with space group P6_3mc",
        ),
        (mp._density, "What is the density of GaN?", "6.08 g/cm³"),
        (mp._formation, "What is the formation energy per atom of GaN?", "-0.660 eV per atom"),
        (mp._stability, "Is GaN thermodynamically stable?", "Yes. GaN lies on the convex hull"),
        (mp._magnetism, "Is GaN magnetic", "GaN is predicted to be non-magnetic."),
        (mp._bulk_modulus, "What is the bulk modulus of GaN?", "about 172 GPa"),
    ],
)
def test_property_templates(prop, question: str, answer: str):
    fact = prop(GAN, "GaN")
    assert fact.question.startswith(question)
    assert answer in fact.answer


def test_templates_handle_metals_metastable_and_missing_values():
    metal = dict(GAN, band_gap=0.0, is_stable=False, energy_above_hull=0.051, ordering="FM", total_magnetization=2.2)
    assert "no band gap" in mp._band_gap(metal, "Fe").answer
    assert mp._stability(metal, "Fe").answer.startswith("No. Fe lies 0.051 eV")
    assert "2.20 μB" in mp._magnetism(metal, "Fe").answer
    assert mp._bulk_modulus(dict(GAN, bulk_modulus=None), "GaN") is None
    assert mp._formation(dict(GAN, nelements=1), "Si") is None


def test_make_questions_cycles_properties_and_records_the_source():
    docs = [dict(GAN, formula_pretty=f"Ga{i}N", material_id=f"mp-{i}") for i in range(7)]
    made = mp.make_questions(docs)
    assert len({q["topic"] for q in made}) == 7
    assert made[0]["source_url"] == "https://next-gen.materialsproject.org/materials/mp-0"
    assert made[0]["keep_terms"] == ["Ga0N"]


def test_api_key_reads_env_or_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MP_API_KEY", " abc ")
    assert mp.api_key() == "abc"
    monkeypatch.delenv("MP_API_KEY")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    with pytest.raises(Exception, match="no Materials Project API key"):
        mp.api_key()
    (tmp_path / ".mp_api_key").write_text("from-file\n", encoding="utf-8")
    assert mp.api_key() == "from-file"


class FakeResponse:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self) -> dict:
        return self._body


class FakeMPSession:
    def __init__(self, pages: list[list[dict]]) -> None:
        self.pages, self.calls = pages, []

    def get(self, url, params, headers, timeout):
        self.calls.append(params)
        return FakeResponse(200, {"data": self.pages[len(self.calls) - 1]})


def test_fetch_materials_pages_until_a_short_page():
    session = FakeMPSession([[GAN, GAN], [GAN]])
    assert len(list(mp.fetch_materials("k", page_size=2, session=session))) == 3
    assert [c["_skip"] for c in session.calls] == [0, 2]
    assert session.calls[0]["theoretical"] == "false"


def test_fetch_materials_reports_http_errors():
    class Denied:
        def get(self, *args, **kwargs):
            return FakeResponse(401, {"error": "bad key"})

    with pytest.raises(Exception, match="HTTP 401"):
        list(mp.fetch_materials("k", session=Denied()))


def test_passages_split_on_sentences():
    sentence = "Silicon carbide is a hard semiconductor used in power electronics and abrasives. "
    article = {"title": "Silicon carbide", "url": "u", "text": sentence * 30}
    parts = wikipedia.passages(article)
    assert len(parts) >= 2
    assert all(wikipedia.MIN_WORDS <= len(p["text"].split()) <= wikipedia.MAX_WORDS + 20 for p in parts)
    assert wikipedia.passages({"title": "t", "url": "u", "text": "Too short."}) == []


def test_category_filters():
    assert wikipedia.SKIP_CATEGORY.search("Materials scientists")
    assert wikipedia.SKIP_CATEGORY.search("Lists of alloys")
    assert not wikipedia.SKIP_CATEGORY.search("Ceramic materials")
    assert wikipedia.SKIP_TITLE.search("List of alloys")


PASSAGE = {
    "id": "wp-1",
    "title": "Boron nitride",
    "url": "https://en.wikipedia.org/wiki/Boron_nitride",
    "text": "Boron nitride is a refractory compound with the chemical formula BN. Its hexagonal form is used as a "
    "lubricant, and cubic boron nitride is the second hardest material after diamond.",
}


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_json("Sure! [1, 2] done") == [1, 2]
    assert parse_json("no json here") is None


def test_wiki_question_must_be_grounded_and_self_contained():
    good = {
        "question": "Which material is cubic boron nitride second only to in hardness?",
        "short_answer": "diamond",
        "answer": "Cubic boron nitride is the second hardest material after diamond.",
    }
    record = questions.accept(good, PASSAGE)
    assert record is not None
    assert record["source_url"] == PASSAGE["url"]
    assert questions.accept(dict(good, short_answer="graphite", answer="graphite"), PASSAGE) is None
    assert questions.accept(dict(good, question="What is it second hardest after in the passage?"), PASSAGE) is None
    assert questions.accept(dict(good, question="Is diamond harder than cubic boron nitride or not?"), PASSAGE) is None
    assert questions.accept("not a dict", PASSAGE) is None
    assert questions.parse_questions(json.dumps([good, {"question": "?"}]), PASSAGE) == [record]


def test_keep_terms_finds_formulas_and_numbers():
    assert "Fe2O3" in questions.keep_terms("What is the density of Fe2O3 at 300 K?")
    assert "300 K" in questions.keep_terms("What is the density of Fe2O3 at 300 K?")
    assert questions.keep_terms("What is steel?") == []


def test_paraphrase_validation():
    q = {"id": "q1", "question": "What is the band gap of GaN?", "keep_terms": ["GaN"]}
    good = ["How large is the band gap of GaN?", "GaN has what band gap?", "What band gap does GaN have?",
            "Can you state the band gap of GaN?"]  # fmt: skip
    assert paraphrase.accept(json.dumps(good), q) == good
    assert paraphrase.accept(json.dumps(good[:3]), q) is None
    assert paraphrase.accept(json.dumps([*good[:3], "What is the band gap of gallium nitride?"]), q) is None
    assert paraphrase.accept(json.dumps([*good[:3], "What is the band gap of GaN?"]), q) is None
    rows = paraphrase.phrasings(q, good)
    assert [r["phrasing"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[0]["text"] == q["question"]
    assert rows[1]["id"] == "q1-1"


def test_select_dedupes_and_balances():
    pytest.importorskip("faiss")
    wiki = [{"question": f"What is property {i} of steel?", "source": "wikipedia", "topic": "t"} for i in range(6)]
    wiki.append({"question": "What is property 0 of steel?", "source": "wikipedia", "topic": "t"})
    assert len(select.drop_exact_duplicates(wiki)) == 6
    embedder = KeywordEmbedder(["steel", "glass"])
    assert len(select.drop_near_duplicates(wiki[:6], embedder)) == 1
    mp_pool = [{"question": f"q{i}", "source": "materials_project", "topic": f"p{i % 3}", "n_elements": 2 + i % 2}
               for i in range(30)]  # fmt: skip
    chosen = select.choose(mp_pool, wiki[:6], total=10)
    assert len(chosen) == 10
    assert sum(r["source"] == "wikipedia" for r in chosen) == 5
    assert select.choose(mp_pool, [], total=10, seed=1) == select.choose(mp_pool, [], total=10, seed=1)


def test_strip_reasoning():
    from hallumat.extend.llm import strip_reasoning

    assert strip_reasoning("<think>\nhmm\n</think>\n\nGaN is hexagonal.") == "GaN is hexagonal."
    assert strip_reasoning("  plain  ") == "plain"


def test_resolve_seeds():
    assert resolve_seeds(3, Sampling(seed=10), None) == [10, 11, 12]
    assert resolve_seeds(2, Sampling(), [7, 8]) == [7, 8]
    with pytest.raises(ValueError, match="2 seeds for 3"):
        resolve_seeds(3, Sampling(), [1, 2])


def test_labels_match_the_thresholds():
    vectors = {"same": [1.0, 0.0], "close": [1.0, 0.6], "far": [0.0, 1.0]}

    def embed(texts):
        return np.asarray([vectors[t] for t in texts], dtype=np.float32)

    results = label_pairs(["same", "close", "far", ""], ["same", "same", "same", "same"], embed)
    assert [level for _, level in results] == ["Low", "Low", "High", "High"]
    assert results[0][0] == pytest.approx(1.0)
    assert results[3] == (0.0, "High")
    assert cosine_rows(np.zeros((1, 2), np.float32), np.ones((1, 2), np.float32)).tolist() == [0.0]


class ScriptedWriter:
    """Answers question-writing, paraphrasing and answering prompts with valid output."""

    name = "fake"

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def chat(self, conversations, sampling, seeds=None):
        self.seeds += list(seeds or [])
        replies = []
        for conversation in conversations:
            prompt = conversation[-1]["content"]
            if prompt.startswith("You are writing quiz questions"):
                replies.append(json.dumps([{
                    "question": "Which material is cubic boron nitride second only to in hardness?",
                    "short_answer": "diamond",
                    "answer": "Cubic boron nitride is the second hardest material after diamond.",
                }]))  # fmt: skip
            elif prompt.startswith("Rewrite the question"):
                original = prompt.split("Question: ", 1)[1].split("\n", 1)[0]
                replies.append(json.dumps([f"{original[:-1]}, variant {i}?" for i in range(4)]))
            else:
                replies.append(f"Answer to: {prompt}")
        return replies


def test_pipeline_end_to_end(tmp_path: Path):
    pytest.importorskip("faiss")
    pytest.importorskip("pyarrow")
    work = tmp_path / "work"
    docs = [dict(GAN, formula_pretty=f"Ga{i}N", material_id=f"mp-{i}") for i in range(12)]
    write_jsonl(work / "mp_materials.jsonl", docs)
    write_jsonl(work / "wiki_passages.jsonl", [PASSAGE])
    writer = ScriptedWriter()

    assert stages.mp_questions(work) == 12
    assert stages.wiki_questions(work, writer) == 1
    assert stages.wiki_questions(work, writer) == 0, "a second run must not redo passages"
    assert stages.select_questions(work, lambda: KeywordEmbedder(["boron", "diamond"]), total=6) == 6
    assert stages.paraphrase_questions(work, writer) == 6
    assert len(list(read_jsonl(work / "phrasings.jsonl"))) == 30
    assert stages.respond(work, "qwen3-8b", writer, shard_index=0, shard_count=2) == 15
    assert stages.respond(work, "qwen3-8b", writer, shard_index=1, shard_count=2) == 15

    def embed(texts):
        return np.asarray([[len(t) % 7 + 1.0, 1.0] for t in texts], dtype=np.float32)

    assert stages.label(work, "qwen3-8b", embed) == 30

    from hallumat.extend.export import export

    out = tmp_path / "export"
    assert export(work, out) == 30
    table = pd.concat(pd.read_parquet(p) for p in sorted((out / "responses").glob("*.parquet")))
    assert set(table["model"]) == {"qwen3-8b"}
    assert table["phrasing"].value_counts().to_dict() == {i: 6 for i in range(5)}
    assert set(table["hallucination_level"]) <= {"Low", "Medium", "High"}
    qs = pd.read_parquet(out / "questions.parquet")
    assert len(qs) == 6
    assert qs["phrasings"].map(len).eq(5).all()
    assert (out / "summary.csv").read_text(encoding="utf-8").startswith("model,source,Low,Medium,High,answers")
    assert stages.seed_of("x") == stages.seed_of("x")
