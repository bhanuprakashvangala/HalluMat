"""`hallumat reproduce` on the released dataset gives the committed outputs.

The printed report must equal tests/golden/reproduce_stdout.txt and agreement_metrics.csv must equal the committed
file byte for byte (after line-end normalisation of the checkout). The PNGs are compared with
tests/golden/figures.sha256 only on the plotting stack pinned in constraints/reproduce.txt, since other matplotlib
or Pillow releases draw slightly different pixels.
"""

import hashlib
from importlib.metadata import version
from pathlib import Path

import pytest

from hallumat.cli import main
from tests.conftest import GOLDEN

PINNED = {"matplotlib": "3.10.8", "pillow": "11.3.0"}


@pytest.fixture(scope="module")
def reproduced(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    import contextlib
    import io

    out = tmp_path_factory.mktemp("reproduce")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert main(["--root", str(repo_root), "reproduce", "--out", str(out)]) == 0
    return out, stdout.getvalue()


def lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def test_report_matches_the_golden(reproduced: tuple[Path, str]):
    _, stdout = reproduced
    assert stdout == (GOLDEN / "reproduce_stdout.txt").read_text(encoding="utf-8")


def test_table_one_and_table_two(reproduced: tuple[Path, str]):
    _, stdout = reproduced
    for line in [
        "Total Unique Queries    655",
        "Total Paraphrased Queries   3269",
        "Medium Hallucination Responses    872",
        "F-statistic: 10.0936, p-value: 0.000000",
        "T-statistic: -1.5946, p-value: 0.114987",
        "|        424 |     1.09545  |",
    ]:
        assert line in stdout


def test_metrics_csv_is_byte_identical(reproduced: tuple[Path, str], repo_root: Path):
    out, _ = reproduced
    written = (out / "agreement_metrics.csv").read_bytes()
    assert b"\r" not in written
    assert written == lf((repo_root / "results" / "agreement_metrics.csv").read_bytes())


def test_every_figure_is_written(reproduced: tuple[Path, str]):
    out, _ = reproduced
    names = [line.split()[1] for line in (GOLDEN / "figures.sha256").read_text(encoding="utf-8").splitlines()]
    assert all((out / name).stat().st_size > 10_000 for name in names)


def test_figures_are_byte_identical_on_the_pinned_stack(reproduced: tuple[Path, str]):
    installed = {name: version(name) for name in PINNED}
    if installed != PINNED:
        pytest.skip(f"plotting stack {installed} is not the pinned {PINNED} (constraints/reproduce.txt)")
    out, _ = reproduced
    for line in (GOLDEN / "figures.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name
