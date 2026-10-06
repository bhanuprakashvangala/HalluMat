# HalluMat

Code and data for **HalluMat: Detecting Hallucinations in LLM-Generated Materials Science Content Through Multi-Stage Verification**
Bhanu Prakash Vangala, Sajid Mahmud, Pawan Neupane, Joel Selvaraj, Jianlin Cheng.
AAAI 2025 Spring Symposium on AI for Engineering and Scientific Discoveries. [arXiv:2512.22396](https://arxiv.org/abs/2512.22396)

HalluMat is a benchmark of LLM answers to materials science questions, labelled by how much each answer hallucinates.
Each question is asked in several paraphrased forms, so you can check both whether a model is wrong and whether it stays
consistent when the question is reworded. The repository also has HalluMatDetector, which checks an answer with
self-consistency, Wikipedia retrieval and a contradiction graph.

**Dataset:** [huggingface.co/datasets/bhanuprakashvangala/HalluMat](https://huggingface.co/datasets/bhanuprakashvangala/HalluMat) (a copy is in `data/`).

## Layout

```
data/        HalluMatData.csv and the per-question summary tables
analysis/    dataset statistics and the paper's tables and figures
detector/    response generation, labelling, and HalluMatDetector
results/     outputs of the analysis scripts
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the dataset statistics and figures

No GPU needed; each script takes a few seconds.

```bash
python analysis/dataset_summary.py      # Table 1: dataset composition
python analysis/plots.py                # Table 2 (PHCS) and the level distribution figures
python detector/agreement_metrics.py    # agreement between the assigned and recomputed levels
python detector/confusion_matrix.py     # confusion matrix figure
```

Figures go to `results/figures/` and metrics to `results/agreement_metrics.csv`.

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
pip install -r requirements-detector.txt
export HF_TOKEN=your_hugging_face_token
python detector/hallu_mat_detector.py   # asks for a query, then prints the reliability score and graphs
```

To build responses for your own questions, put `Paraphrased_Questions.csv` (columns `Question, Answer, Explanation`)
in `data/`, then run `detector/generate_responses.py` followed by `detector/label_responses.py`.

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
