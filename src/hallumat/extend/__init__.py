"""Build the extended HalluMat dataset: 50,000 unique materials-science questions with sourced reference answers,
five phrasings of each, answers from several LLMs, and a hallucination label for every answer.

Stages (each reads and writes files in one work directory, and each can be rerun to resume):

1. `fetch-mp`, `fetch-wiki`: facts from the Materials Project and passages from Wikipedia.
2. `mp-questions`, `wiki-questions`: candidate questions with reference answers and their source.
3. `select`: removes duplicates and near-duplicates and picks the final questions.
4. `paraphrase`: four rewordings per question, so each question has five phrasings.
5. `respond`: answers every phrasing with each LLM.
6. `label`: BERT similarity of every answer to the reference answer, and its hallucination level.
7. `export`: the published tables.

Stages 2 (Wikipedia), 4 and 5 run an LLM through vLLM and need a GPU; the others run on a CPU.
"""
