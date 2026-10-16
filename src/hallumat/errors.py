"""Errors that the command line reports as one line ('hallumat: error: ...', exit status 1)."""

from __future__ import annotations


class HallumatError(Exception):
    """Base class for the errors hallumat expects and reports without a traceback."""


class DataNotFoundError(HallumatError, FileNotFoundError):
    """A required input file does not exist."""


class DatasetError(HallumatError, ValueError):
    """An input table lacks a required column or holds a value outside its allowed set."""


class MissingDependencyError(HallumatError, ImportError):
    """An optional dependency is not installed; the message names the extra that provides it."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(f'{package} is required for this command: pip install -e ".[{extra}]"')
        self.package = package
        self.extra = extra

    def __reduce__(self) -> tuple[type[MissingDependencyError], tuple[str, str]]:
        return type(self), (self.package, self.extra)
