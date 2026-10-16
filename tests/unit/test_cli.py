import sys
from pathlib import Path

import pytest

from hallumat import __version__
from hallumat.cli import main


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    status = main(list(argv))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_version(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"hallumat {__version__}"


def test_no_command_prints_help_and_exits_2(capsys: pytest.CaptureFixture[str]):
    status, out, err = run(capsys)
    assert (status, out) == (2, "")
    assert "reproduce" in err


def test_summary_of_a_given_dataset(capsys: pytest.CaptureFixture[str], tiny_csv: Path):
    status, out, err = run(capsys, "summary", "--dataset", str(tiny_csv))
    assert (status, err) == (0, "")
    assert out.startswith("Dataset composition (Table 1)\n")
    assert "Total Unique Queries      3" in out


def test_consistency_and_agreement_sections(capsys: pytest.CaptureFixture[str], tiny_csv: Path):
    assert "Table 2" in run(capsys, "consistency", "--dataset", str(tiny_csv))[1]
    assert "Precision (macro)" in run(capsys, "agreement", "--dataset", str(tiny_csv))[1]


def test_reproduce_writes_under_root_results(capsys: pytest.CaptureFixture[str], tiny_csv: Path, tmp_path: Path):
    (tmp_path / "data").mkdir()
    tiny_csv.rename(tmp_path / "data" / "HalluMatData.csv")
    status, _, err = run(capsys, "--root", str(tmp_path), "reproduce")
    assert status == 0
    assert "Wrote 5 files" in err
    assert (tmp_path / "results" / "agreement_metrics.csv").is_file()
    assert len(list((tmp_path / "results" / "figures").glob("*.png"))) == 4


def test_missing_dataset_is_a_one_line_error(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    status, out, err = run(capsys, "--root", str(tmp_path), "summary")
    assert (status, out) == (1, "")
    assert err.startswith("hallumat: error: HalluMat dataset not found: ")
    assert err.count("\n") == 1


@pytest.mark.parametrize(
    ("argv", "package"),
    [
        (["detect", "what", "is", "steel"], "transformers"),
        (["label", "--responses", "{csv}", "--out", "{out}"], "transformers"),
        (["generate", "--questions", "{questions}", "--out", "{out}"], "transformers"),
    ],
)
def test_model_commands_name_the_missing_extra(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_csv: Path,
    argv: list[str],
    package: str,
):
    questions = tmp_path / "questions.csv"
    questions.write_text("Question,Answer,Explanation\nq,a,e\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, package, None)
    args = [a.format(csv=tiny_csv, out=tmp_path / "out.csv", questions=questions) for a in argv]
    status, out, err = run(capsys, *args)
    assert (status, out) == (1, "")
    assert err == f'hallumat: error: {package} is required for this command: pip install -e ".[models]"\n'


def test_label_needs_an_existing_input(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    status, _, err = run(capsys, "label", "--responses", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "o.csv"))
    assert status == 1
    assert "responses table not found" in err


def test_empty_detect_query_is_an_error(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "   ")
    status, _, err = run(capsys, "detect")
    assert status == 1
    assert err == "hallumat: error: the query is empty\n"
