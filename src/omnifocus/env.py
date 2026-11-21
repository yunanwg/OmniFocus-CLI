"""Safe configuration readers for environment variables and secret files."""

from __future__ import annotations

import os
from pathlib import Path

from omnifocus.errors import OFError


def read_env_or_file[ErrorT: OFError](name: str, *, error_type: type[ErrorT]) -> str | None:
    """Read *name* directly or from ``<name>_FILE`` without exposing its value.

    The ``_FILE`` form is intended for container secret mounts.  Supplying both
    forms is rejected so a stale environment variable cannot silently override
    a rotated secret file.  A single final line ending is removed because
    secret-management CLIs commonly write one.
    """

    file_name = f"{name}_FILE"
    value = os.environ.get(name)
    file_path = os.environ.get(file_name)
    if value is not None and file_path is not None:
        raise error_type(f"{name} and {file_name} cannot both be set")
    if file_path is None:
        return value

    try:
        secret = Path(file_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise error_type(f"Unable to read secret file configured by {file_name}") from exc
    return secret.removesuffix("\n").removesuffix("\r")
