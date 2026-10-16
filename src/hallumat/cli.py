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
    label.add_argument("--model", default="bert-base-uncased")
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
