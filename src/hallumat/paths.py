"""Where the released data and the outputs live, relative to a repository root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hallumat.errors import DataNotFoundError


@dataclass(frozen=True)
class Paths:
    """File locations under root (default: the current directory)."""

    root: Path = Path()

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def dataset(self) -> Path:
        """HalluMatData.csv: one row per paraphrased question and its generated response."""
        return self.data / "HalluMatData.csv"

    @property
    def results(self) -> Path:
        return self.root / "results"


def require_file(path: Path, what: str) -> Path:
    """Return path if it is an existing file, else raise DataNotFoundError naming what it should hold."""
    if not path.is_file():
        raise DataNotFoundError(f"{what} not found: {path}")
    return path
