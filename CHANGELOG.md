# Changelog

## 2.0.0

The scripts of the first release are now the installable Python package `hallumat`, with one `hallumat` command,
tests and CI. The dataset analysis gives the same numbers: `hallumat reproduce` prints the same tables and writes
`results/agreement_metrics.csv` and the four figures byte for byte (with the plotting stack in
`constraints/reproduce.txt`).

### Commands

| 1.x | 2.0.0 |
|---|---|
| `pip install -r requirements.txt` | `pip install -e .` |
| `pip install -r requirements-detector.txt` | `pip install -e ".[detector]"` |
| `python analysis/dataset_summary.py` | `hallumat summary` |
| `python analysis/plots.py` | `hallumat consistency` (tables) and `hallumat reproduce` (tables and figures) |
| `python detector/agreement_metrics.py` | `hallumat agreement` (table) and `hallumat reproduce` (writes the CSV) |
| `python detector/confusion_matrix.py` | `hallumat reproduce` |
| `python detector/generate_responses.py` | `hallumat generate --questions Q.csv --out R.csv` |
| `python detector/label_responses.py` | `hallumat label --responses R.csv --out L.csv` |
| `python detector/hallu_mat_detector.py` | `hallumat detect "QUERY"` |

### Changes

- Nothing runs at import time. Models load only when a command needs them, and importing `hallumat` loads no
  optional dependency.
- The detector is split into stages (`intrinsic`, `retrieval`, `extrinsic`, `scoring`, `graph`, `plots`) behind
  `HalluMatDetector.detect()`, which returns a `DetectionReport` instead of printing. Its prompts, model calls, their
  order, the score weights and every threshold are unchanged and now live in `DetectorConfig`.
- The detector's figures are saved as PNG files (`--plots DIR`) instead of only being shown on screen (`--show`).
- Google Search and Google Scholar retrieval through SerpAPI, part of the original detector, is back as an optional
  source that reads its key from `$SERPAPI_API_KEY`.
- Query chunks are de-duplicated in a fixed order, so BM25 ties and Wikipedia lookups no longer depend on Python's
  hash seed. `--max-wikipedia-lookups` can cap the lookups (default: all, as before), and `--seed` fixes the
  community detection.
- LangChain, NLTK and rouge-score are no longer dependencies: only LangChain's embedding wrapper was used, and it is
  replaced by sentence-transformers with the same preprocessing; the other two were imported but unused. statsmodels,
  imported but unused by the analysis, is dropped too.
- `hallumat label` writes to the path given by `--out`, where `label_responses.py` wrote `Hallucination_Analysis.csv`
  into the current directory, and `hallumat generate` takes its input and output paths as options.
- The Hugging Face token is passed as `token=` (the `use_auth_token=` argument is deprecated in transformers).
- `agreement_metrics.py` printed the dataset's column names before the metrics; `hallumat agreement` prints only the
  metrics.
- `results/agreement_metrics.csv` has LF line ends on every platform.
- Groups with equal PHCS are ranked in `Group_ID` order. The 1.x scripts used numpy's default sort, which is not
  stable and orders ties differently on different platforms, so the last rows of Table 2 and the Welch t-test (which
  picks the 10 least inconsistent of the 461 groups tied at 0) changed from machine to machine: on Windows the t-test
  gave t = -4.8736, p = 0.000009 and on Linux t = -6.0023, p < 0.000001. With the fixed order it is t = -1.5946,
  p = 0.114987 everywhere. The paper reports neither test.
