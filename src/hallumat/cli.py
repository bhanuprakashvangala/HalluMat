"""The `hallumat` command.

    hallumat reproduce      the dataset analysis: prints every table and writes results/ (no GPU)
    hallumat summary        Table 1, the dataset composition
    hallumat consistency    PHCS, Table 2, the ANOVA and the t-test
    hallumat agreement      agreement of the two labelling passes
    hallumat generate       answer paraphrased questions with Llama-2 (GPU, `models` extra)
    hallumat label          label responses by BERT similarity (`models` extra)
    hallumat detect QUERY   run HalluMatDetector on one query (GPU, `detector` extra)

Errors print one line, 'hallumat: error: ...', and exit with status 1; usage errors exit with status 2.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from hallumat import __version__
from hallumat.errors import HallumatError
from hallumat.paths import Paths, require_file

log = logging.getLogger("hallumat")

Command = Callable[[argparse.Namespace], int]


def _paths(args: argparse.Namespace) -> Paths:
    return Paths(args.root)


def _analysis(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from hallumat.analysis import analyse
    from hallumat.dataset import load_dataset

    return analyse(load_dataset(args.dataset or _paths(args).dataset))


def _reproduce(args: argparse.Namespace) -> int:
    from hallumat.analysis import format_report, write_outputs

    result = _analysis(args)
    print(format_report(result))
    out = args.out or _paths(args).results
    written = write_outputs(result, out)
    print(f"\nWrote {len(written)} files to {out}", file=sys.stderr)
    return 0


def _summary(args: argparse.Namespace) -> int:
    from hallumat.analysis.report import format_composition

    print(format_composition(_analysis(args)))
    return 0


def _consistency(args: argparse.Namespace) -> int:
    from hallumat.analysis.report import format_consistency

    print(format_consistency(_analysis(args)))
    return 0


def _agreement(args: argparse.Namespace) -> int:
    from hallumat.analysis.report import format_agreement

    print(format_agreement(_analysis(args)))
    return 0


def _generate(args: argparse.Namespace) -> int:
    from hallumat.generation import GenerationSettings, HFResponseGenerator, generate_responses

    questions = pd.read_csv(require_file(args.questions, "questions table"))
    settings = GenerationSettings(args.max_length, args.temperature, args.top_p)
    responses = generate_responses(questions, HFResponseGenerator(args.model, settings))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    responses.to_csv(args.out, index=False)
    print(f"Wrote {len(responses)} responses to {args.out}", file=sys.stderr)
    return 0


def _label(args: argparse.Namespace) -> int:
    from hallumat.labeling import BertEmbedder, label_responses

    frame = pd.read_csv(require_file(args.responses, "responses table"))
    labelled = label_responses(frame, BertEmbedder(args.model))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(args.out, index=False)
    print(f"Wrote {len(labelled)} labelled responses to {args.out}", file=sys.stderr)
    return 0


def _detect(args: argparse.Namespace) -> int:
    from hallumat.detector import DetectorConfig, HalluMatDetector, format_report, save_plots

    query = " ".join(args.query).strip() or input("Enter your query: ").strip()
    if not query:
        raise HallumatError("the query is empty")
    config = DetectorConfig(max_wikipedia_lookups=args.max_wikipedia_lookups)
    report = HalluMatDetector.from_pretrained(config).detect(query, seed=args.seed)
    print(format_report(report))
    if args.plots or args.show:
        out = args.plots or _paths(args).results / "detector"
        for path in save_plots(report, out, show=args.show):
            print(f"Wrote {path}", file=sys.stderr)
    return 0


def _extend(args: argparse.Namespace) -> int:
    """The dataset-extension stages; see hallumat.extend."""
    from hallumat.extend import stages
    from hallumat.extend.records import write_jsonl

    work: Path = args.work
    stage = args.stage
    if stage == "respond" and not args.model:
        raise HallumatError("respond needs --model")
    work.mkdir(parents=True, exist_ok=True)
    if stage == "fetch-mp":
        from hallumat.extend.materials_project import api_key, fetch_materials

        count = write_jsonl(work / "mp_materials.jsonl", fetch_materials(api_key(), limit=args.limit))
    elif stage == "fetch-wiki":
        from hallumat.extend import wikipedia

        client = wikipedia.WikiClient()
        titles = wikipedia.crawl_categories(client, depth=args.depth)
        articles = wikipedia.fetch_leads(client, titles[: args.limit] if args.limit else titles)
        count = write_jsonl(
            work / "wiki_passages.jsonl", (p for article in articles for p in wikipedia.passages(article))
        )
    elif stage == "download":
        from huggingface_hub import snapshot_download

        patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.jinja"]
        for model in [
            stages.WRITER,
            *stages.RESPONDERS.values(),
            "sentence-transformers/all-MiniLM-L6-v2",
            "google-bert/bert-base-uncased",
        ]:
            log.info("downloading %s", model)
            snapshot_download(model, allow_patterns=patterns)
        count = len(stages.RESPONDERS) + 3
    elif stage == "mp-questions":
        count = stages.mp_questions(work)
    elif stage == "select":
        from hallumat.detector.models import SentenceEmbedder

        count = stages.select_questions(
            work, lambda: SentenceEmbedder("sentence-transformers/all-MiniLM-L6-v2"), total=args.total
        )
    elif stage in ("wiki-questions", "paraphrase", "respond"):
        from hallumat.extend.llm import make_backend

        model = stages.RESPONDERS[args.model] if stage == "respond" else (args.writer or stages.WRITER)
        options: dict[str, object] = {"max_model_len": args.max_model_len}
        if args.engine == "vllm":
            options["tensor_parallel_size"] = args.tensor_parallel
            if stage == "respond" and args.model in stages.CHAT_TEMPLATE_KWARGS:
                options["chat_template_kwargs"] = stages.CHAT_TEMPLATE_KWARGS[args.model]
        backend = make_backend(model, engine=args.engine, **options)
        shard_index, shard_count = args.shard
        if stage == "wiki-questions":
            count = stages.wiki_questions(work, backend, shard_index=shard_index, shard_count=shard_count)
        elif stage == "paraphrase":
            count = stages.paraphrase_questions(work, backend, shard_index=shard_index, shard_count=shard_count)
        else:
            count = stages.respond(work, args.model, backend, shard_index=shard_index, shard_count=shard_count)
    elif stage == "label":
        from hallumat.extend.labels import BatchBertEmbedder

        embedder = BatchBertEmbedder(batch_size=args.batch_size)
        count = sum(stages.label(work, model, embedder.embed_many) for model in args.models)
    else:  # export
        from hallumat.extend.export import export

        count = export(work, args.out or work / "export")
    print(f"{stage}: {count} records", file=sys.stderr)
    return 0


def _shard(text: str) -> tuple[int, int]:
    index, _, count = text.partition("/")
    try:
        parsed = int(index), int(count)
    except ValueError:
        raise argparse.ArgumentTypeError("expected INDEX/COUNT, e.g. 0/4") from None
    if not 0 <= parsed[0] < parsed[1]:
        raise argparse.ArgumentTypeError("INDEX must be between 0 and COUNT - 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hallumat", description="HalluMat dataset analysis and the HalluMatDetector hallucination detector."
    )
    parser.add_argument("--version", action="version", version=f"hallumat {__version__}")
    parser.add_argument("--root", type=Path, default=Path(), help="repository root holding data/ and results/")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more log output (-vv for debug)")
    parser.set_defaults(handler=None)
    commands = parser.add_subparsers(title="commands", metavar="COMMAND")

    def analysis_command(name: str, handler: Command, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("--dataset", type=Path, help="HalluMatData.csv (default: <root>/data/HalluMatData.csv)")
        sub.set_defaults(handler=handler)
        return sub

    reproduce = analysis_command(
        "reproduce", _reproduce, "print every table and write agreement_metrics.csv and the figures"
    )
    reproduce.add_argument("--out", type=Path, help="output directory (default: <root>/results)")
    analysis_command("summary", _summary, "Table 1: the dataset composition")
    analysis_command("consistency", _consistency, "PHCS and Table 2, the ANOVA and the t-test")
    analysis_command("agreement", _agreement, "agreement of the recomputed with the assigned levels")

    generate = commands.add_parser("generate", help="answer paraphrased questions with an LLM")
    generate.add_argument("--questions", type=Path, required=True, help="CSV with Question, Answer, Explanation")
    generate.add_argument("--out", type=Path, required=True, help="CSV to write")
    generate.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    generate.add_argument("--max-length", type=int, default=128, help="tokens, prompt included")
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.set_defaults(handler=_generate)

    label = commands.add_parser("label", help="label responses by BERT similarity to the reference answer")
    label.add_argument("--responses", type=Path, required=True, help="CSV with Generated_Response, Original_Answer")
    label.add_argument("--out", type=Path, required=True, help="CSV to write")
    label.add_argument("--model", default="google-bert/bert-base-uncased")
    label.set_defaults(handler=_label)

    detect = commands.add_parser("detect", help="run HalluMatDetector on a query")
    detect.add_argument("query", nargs="*", help="the question (asked for interactively when omitted)")
    detect.add_argument("--plots", type=Path, help="directory for the fact-graph figures")
    detect.add_argument("--show", action="store_true", help="also open the figures in a window")
    detect.add_argument("--seed", type=int, help="seed for the community detection")
    detect.add_argument(
        "--max-wikipedia-lookups", type=int, help="look up at most this many query chunks on Wikipedia (default: all)"
    )
    detect.set_defaults(handler=_detect)

    from hallumat.extend.stages import RESPONDERS

    extend = commands.add_parser("extend", help="build the extended dataset (see hallumat.extend)")
    extend.add_argument(
        "stage",
        choices=[
            "download",
            "fetch-mp",
            "fetch-wiki",
            "mp-questions",
            "wiki-questions",
            "select",
            "paraphrase",
            "respond",
            "label",
            "export",
        ],
    )
    extend.add_argument("--work", type=Path, required=True, help="work directory shared by the stages")
    extend.add_argument("--model", choices=sorted(RESPONDERS), help="answering model (respond)")
    extend.add_argument("--models", nargs="+", default=sorted(RESPONDERS), help="models to label (label)")
    extend.add_argument("--writer", help="model that writes questions and paraphrases (default Qwen2.5-32B AWQ)")
    extend.add_argument("--engine", choices=["vllm", "transformers"], default="vllm")
    extend.add_argument("--max-model-len", type=int, default=4096)
    extend.add_argument("--tensor-parallel", type=int, default=1, help="GPUs per model (vLLM)")
    extend.add_argument("--shard", type=_shard, default=(0, 1), metavar="INDEX/COUNT", help="split a stage across jobs")
    extend.add_argument("--total", type=int, default=50_000, help="questions to select (select)")
    extend.add_argument("--depth", type=int, default=2, help="category depth (fetch-wiki)")
    extend.add_argument("--limit", type=int, help="fetch at most this many documents (fetch-mp, fetch-wiki)")
    extend.add_argument("--batch-size", type=int, default=64, help="BERT batch size (label)")
    extend.add_argument("--out", type=Path, help="export directory (default <work>/export)")
    extend.set_defaults(handler=_extend)
    return parser


def configure_logging(verbosity: int) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.WARNING - 10 * min(verbosity, 2))
    log.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command: Command | None = args.handler
    if command is None:
        parser.print_help(sys.stderr)
        return 2
    configure_logging(args.verbose)
    try:
        return command(args)
    except HallumatError as exc:
        log.debug("the command failed", exc_info=True)
        print(f"hallumat: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
