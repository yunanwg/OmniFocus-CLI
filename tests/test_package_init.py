"""Tests for :mod:`omnifocus.__init__`."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "omnifocus" / "__init__.py"


def _load_init_module() -> ModuleType:
    """Load ``omnifocus.__init__`` under a temporary module name."""
    spec = importlib.util.spec_from_file_location("test_omnifocus_init", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_reads_package_metadata() -> None:
    """Package version should come from installed package metadata."""
    with patch("importlib.metadata.version", return_value="1.2.0"):
        module = _load_init_module()

    assert module.__version__ == "1.2.0"


def test_version_falls_back_when_metadata_missing() -> None:
    """Missing package metadata should use the local fallback version."""
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        module = _load_init_module()

    assert module.__version__ == "0+unknown"
