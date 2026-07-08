"""Container launcher for OmniFocus CLI, MCP, and HTTPS API modes.

This module keeps native Python console scripts unchanged while making the
container UX simpler:

- no args -> MCP server mode
- ``mcp`` -> explicit MCP server mode
- ``http`` -> explicit HTTPS API mode
- CLI commands like ``sync`` or ``add`` -> Click CLI mode
"""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import sys
from collections.abc import Sequence
from typing import NoReturn

import click

from omnifocus.cli import cli
from omnifocus.http_api import main as http_main
from omnifocus.mcp_server import main as mcp_main
from omnifocus.mcp_server import run_http as mcp_run_http

_CLI_FLAGS = {"--help", "-h", "--version"}
_CLI_COMMANDS = set(cli.commands)
_USAGE = """Usage:
  podman run --rm of sync
  podman run --rm of add "Buy milk"
  podman run --rm -i of
  podman run --rm -i of mcp
  podman run --rm of http
"""


def _is_cli_invocation(argv: Sequence[str]) -> bool:
    """Return True when *argv* should dispatch to the Click CLI."""
    return bool(argv) and (argv[0] in _CLI_FLAGS or argv[0] in _CLI_COMMANDS)


def _raise_usage_error(message: str) -> NoReturn:
    """Raise a ClickException with usage guidance for the container launcher."""
    raise click.ClickException(f"{message}\n\n{_USAGE}")


def _parse_mcp_http_flags(flags: Sequence[str]) -> tuple[str, int, bool]:
    """Parse the optional ``--host``/``--port``/``--stateful`` flags for ``mcp --http``.

    Returns ``(host, port, stateless)``. The Streamable HTTP transport defaults to
    STATELESS (see :func:`omnifocus.mcp_server.build_http_app`); ``--stateful``
    opts back into per-session state.
    """
    host = "0.0.0.0"  # noqa: S104 — container listens on all interfaces (see mcp_server.run_http)
    port = 8096
    stateless = True
    rest = list(flags)
    while rest:
        flag = rest.pop(0)
        if flag == "--host":
            if not rest:
                _raise_usage_error("--host requires a value.")
            host = rest.pop(0)
        elif flag == "--port":
            if not rest:
                _raise_usage_error("--port requires a value.")
            value = rest.pop(0)
            try:
                port = int(value)
            except ValueError:
                _raise_usage_error(f"--port must be an integer, got {value!r}.")
        elif flag == "--stateful":
            stateless = False
        else:
            _raise_usage_error(f"Unknown 'mcp --http' option: {flag!r}.")
    return host, port, stateless


def _run_mcp(argv: Sequence[str]) -> None:
    """Dispatch the ``mcp`` launcher mode: stdio by default, Streamable HTTP with ``--http``.

    Bare ``mcp`` keeps the stdio transport (used by the fallback wrapper). ``mcp --http``
    serves the in-process Streamable HTTP transport that replaces the supergateway bridge.
    """
    if not argv:
        mcp_main()
        return
    if argv[0] != "--http":
        _raise_usage_error(
            "The 'mcp' mode takes no positional arguments; pass --http for Streamable HTTP."
        )
    host, port, stateless = _parse_mcp_http_flags(argv[1:])
    mcp_run_http(host=host, port=port, stateless=stateless)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch container arguments to the CLI, MCP server, or HTTPS API entrypoint."""
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        mcp_main()
        return

    if args[0] == "mcp":
        _run_mcp(args[1:])
        return

    if args[0] == "http":
        http_main(args[1:])
        return

    if args[0] == "of":
        _raise_usage_error(
            "Legacy container syntax detected. Drop the extra 'of' and run the subcommand directly."
        )

    if _is_cli_invocation(args):
        cli.main(args=args, prog_name="of", standalone_mode=True)
        return

    _raise_usage_error(f"Unknown launcher mode or command: {args[0]!r}")


def console_main() -> None:
    """Run the launcher with Click-style user-facing error handling and exit codes."""
    try:
        main()
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":  # pragma: no cover
    console_main()
