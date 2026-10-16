"""Import an optional dependency, or raise MissingDependencyError naming the extra that installs it."""

from __future__ import annotations

import importlib
from types import ModuleType

from hallumat.errors import MissingDependencyError


def require(module: str, package: str, extra: str) -> ModuleType:
    """Import module; if it is not installed, raise MissingDependencyError(package, extra)."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(package, extra) from exc
