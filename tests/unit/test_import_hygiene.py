"""Importing hallumat never loads the heavy optional dependencies; they load only when a command needs them."""

import subprocess
import sys

import pytest

HEAVY = [
    "torch",
    "transformers",
    "sentence_transformers",
    "faiss",
    "rank_bm25",
    "wikipediaapi",
    "networkx",
    "community",
]


@pytest.mark.parametrize(
    "module",
    [
        "hallumat",
        "hallumat.cli",
        "hallumat.analysis",
        "hallumat.labeling",
        "hallumat.generation",
        "hallumat.detector",
        "hallumat.detector.pipeline",
    ],
)
def test_import_loads_no_heavy_dependency(module: str):
    code = f"import sys, {module}; print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == ""
