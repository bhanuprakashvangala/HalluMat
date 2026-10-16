# HalluMat

Code and data for **HalluMat: Detecting Hallucinations in LLM-Generated Materials Science Content Through Multi-Stage Verification**
Bhanu Prakash Vangala, Sajid Mahmud, Pawan Neupane, Joel Selvaraj, Jianlin Cheng.
AAAI 2025 Spring Symposium on AI for Engineering and Scientific Discoveries. [arXiv:2512.22396](https://arxiv.org/abs/2512.22396)

HalluMat is a benchmark of LLM answers to materials science questions, labelled by how much each answer hallucinates.
Each question is asked in several paraphrased forms, so you can check both whether a model is wrong and whether it stays
consistent when the question is reworded. The repository also has HalluMatDetector, which checks an answer with
self-consistency, retrieval-backed entailment and a fact graph.

**Dataset:** [huggingface.co/datasets/bhanuprakashvangala/HalluMat](https://huggingface.co/datasets/bhanuprakashvangala/HalluMat) (a copy is in `data/`).

## Install

Python 3.11 or later:

```bash
git clone https://github.com/bhanuprakashvangala/HalluMat.git
cd HalluMat
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the `hallumat` package and the `hallumat` command. The base install analyses the dataset; extras add
the model-based parts:

| Extra | Installs | Needed for |
|---|---|---|
| `models` | torch, transformers, accelerate, sentencepiece | `hallumat generate`, `hallumat label` |
| `detector` | `models` plus sentence-transformers, faiss-cpu, rank-bm25, wikipedia-api, networkx, python-louvain, requests | `hallumat detect` |
| `dev` | pytest, ruff, mypy, the detector's light dependencies, pre-commit | tests and checks |

A command whose extra is missing stops with the `pip install` command that adds it.

## Commands

| Command | What it does |
|---|---|
| `hallumat reproduce` | the whole dataset analysis: prints every table and writes `results/` (no GPU, a few seconds) |
| `hallumat summary` | Table 1, the dataset composition |
| `hallumat consistency` | PHCS and Table 2, the ANOVA across question groups and the Welch t-test |
| `hallumat agreement` | agreement of the recomputed with the assigned hallucination levels |
| `hallumat generate --questions Q.csv --out R.csv` | answers questions with Llama-2-7B-chat |
| `hallumat label --responses R.csv --out L.csv` | labels responses by BERT similarity to the reference answer |
| `hallumat detect "QUERY" [--plots DIR]` | runs HalluMatDetector on one question |

Global options come before the command: `--root DIR` (the folder holding `data/` and `results/`, default `.`) and
`-v`/`-vv` for more log output. Errors print one line, `hallumat: error: ...`, and exit with status 1.

## Reproduce the dataset statistics and figures

```bash
hallumat reproduce
```

This prints Table 1, the ANOVA and t-test, the dataset overview, Table 2 and the agreement metrics, and writes
`results/agreement_metrics.csv` and the four figures in `results/figures/`. The tables and the metrics are the same on
every platform. The committed figures were drawn with matplotlib 3.10.8 and Pillow 11.3.0; to redraw them byte for byte,
install with `pip install -c constraints/reproduce.txt -e .`.

| | Count |
|---|---|
| Questions (groups) | 655 |
| Paraphrased questions / responses | 3,269 |
| Low hallucination | 57 |
| Medium hallucination | 872 |
| High hallucination | 2,340 |

Differences from the paper:
- The original CSV had 6 more rows with an empty question and response, which the paper counted as High (2,346).
  They are removed here, so High is 2,340.
- Table 2 in the paper lists 0-based group indices, so paper group 423 is `Group_ID` 424 in the data. The PHCS scores match.
  Many groups tie at 0.5477, so the last rows of that table depend on sort order.

## Run HalluMatDetector

The detector uses Llama-2-7B-chat, DeBERTa-v3 for NLI and MiniLM embeddings, so it needs a GPU and access to the
[Llama 2 weights](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

```bash
pip install -e ".[detector]"
export HF_TOKEN=your_hugging_face_token
export SERPAPI_API_KEY=your_serpapi_key       # optional: adds Google Search and Google Scholar evidence
hallumat detect "Which is more ductile, aluminium or alumina?" --plots results/detector
```

It prints the answer, every intrinsic check, the evidence it used and the reliability score, groups the facts of the
refined answer into communities, and with `--plots` saves the 3D and 2D fact graphs and a fact-similarity heatmap.
Without `SERPAPI_API_KEY` it retrieves from Wikipedia only.

From Python:

```python
from hallumat.detector import DetectorConfig, HalluMatDetector, format_report

detector = HalluMatDetector.from_pretrained(DetectorConfig())
report = detector.detect("Which is more ductile, aluminium or alumina?", seed=0)
print(format_report(report))
```

Every model sits behind a small interface (`hallumat.detector.models`), so another LLM, NLI model or embedder can be
passed to `HalluMatDetector(...)` directly. All weights and thresholds are fields of `DetectorConfig`.

## Build responses for your own questions

```bash
pip install -e ".[models]"
hallumat generate --questions my_questions.csv --out my_responses.csv   # columns Question, Answer, Explanation
hallumat label --responses my_responses.csv --out my_labelled.csv
```

## Package layout

```
src/hallumat/
  dataset.py          loading and validating HalluMatData.csv, the Low/Medium/High encodings
  analysis/           composition (Table 1), consistency (PHCS, Table 2, ANOVA, t-test), agreement, figures, report
  generation.py       response generation with a Hugging Face causal LM
  labeling.py         BERT-similarity labels: above 0.8 Low, 0.5 to 0.8 Medium, below 0.5 High
  detector/
    config.py         models, weights and thresholds (DetectorConfig)
    models.py         the LLM, NLI and embedding interfaces and their Hugging Face implementations
    intrinsic.py      verification rounds, NLI confidence and entropy, self-refinement, contradiction checks
    retrieval.py      query chunks, BM25 ranking, Wikipedia and SerpAPI sources, FAISS dense retrieval
    extrinsic.py      NLI entailment against the retrieved evidence
    scoring.py        the reliability score and its confidence level
    graph.py          the fact graph and its Louvain communities
    plots.py          fact-graph figures
    pipeline.py       HalluMatDetector, DetectionReport
  cli.py              the hallumat command
data/                 HalluMatData.csv and the per-question summary tables
results/              agreement_metrics.csv and figures/
tests/                unit tests (fake models, no GPU) and the reproduction tests with their goldens
```

The scripts of the first release (`analysis/*.py`, `detector/*.py`) are at tag `v1.1.0`; see `CHANGELOG.md` for
how each maps to a command.

## Development

```bash
pip install -c constraints/reproduce.txt -e ".[dev]"
pytest                 # about 15 seconds; no GPU or model download
ruff check . && ruff format --check . && mypy
```

## Related

[HalluFormer](https://github.com/bhanuprakashvangala/HalluFormer), a transformer classifier for hallucination detection from the same symposium.

## Citation

```bibtex
@misc{vangala2025hallumatdetectinghallucinationsllmgenerated,
      title={HalluMat: Detecting Hallucinations in LLM-Generated Materials Science Content Through Multi-Stage Verification},
      author={Bhanu Prakash Vangala and Sajid Mahmud and Pawan Neupane and Joel Selvaraj and Jianlin Cheng},
      year={2025},
      eprint={2512.22396},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.22396},
}
```

## License

Code: MIT. Data: CC BY 4.0.
This work comes from the [Bioinformatics and Machine Learning Lab](https://calla.rnet.missouri.edu/cheng/) at the University of Missouri.
