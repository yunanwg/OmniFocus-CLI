"""Write contract for agent-created tasks, enforced by the MCP server.

Agents add tasks for the human to do. Left alone they also schedule them: a due
date here, a flag there, a project guessed from the name. The contract keeps an
agent's task in the Inbox with no project, tags, dates or flag, and starts its
note with a source line (``via <source> · YYYY-MM-DD``) so every agent task can
be traced back and recognised later.

It is off unless ``OF_WRITE_CONTRACT`` is set. Its value is a JSON object of
source exceptions, ``{}`` for none::

    {"*": {"tags": true, "due": "any"},
     "errands": {"project": "Errands", "due": "today"},
     "groceries": {"project": "Errands", "due": "any"}}

A key matches when it appears in the caller's ``source`` (case-insensitive).
``project`` names the one project that source may use; ``due`` is ``today``
(only a due date of today) or ``any``; ``tags`` (``true``) allows tags.
The ``*`` key applies to every source, but only to a write that leaves the
task in the Inbox (no project set): agents may suggest tags and a due date
there and the user triages. It cannot name a project. A source's own key and
``*`` add up. Defer and flag have no exception.
A call that carries ``user_asked`` (the user's own words asking for the field)
bypasses the field checks; it is logged.

``OF_WRITE_LOG`` names an append-only JSON Lines file that records every
add_task and update_task the MCP server performs.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import click

from omnifocus.dateparse import parse_due

logger = logging.getLogger(__name__)

CONTRACT_ENV = "OF_WRITE_CONTRACT"
INBOX_KEY = "*"
LOG_ENV = "OF_WRITE_LOG"
SOURCE_PREFIX = "via "
_SEP = " · "


class ContractError(ValueError):
    """Raised when the contract configuration itself is malformed."""


@dataclass(frozen=True)
class SourceException:
    """What one source may set beyond the plain Inbox task."""

    project: str | None = None
    due: str | None = None  # "today" or "any"
    tags: bool = False


@dataclass(frozen=True)
class WriteContract:
    """The parsed contract: source exceptions plus the optional write log."""

    exceptions: Mapping[str, SourceException] = field(default_factory=dict)
    log_path: Path | None = None

    # -- source line ---------------------------------------------------------

    @staticmethod
    def source_line(source: str, today: date) -> str:
        """Return note line 1 for ``source`` on ``today``."""
        clean = source.strip().removeprefix(SOURCE_PREFIX).strip()
        return f"{SOURCE_PREFIX}{clean}{_SEP}{today.isoformat()}"

    @classmethod
    def stamp_note(cls, note: str, source: str, today: date) -> str:
        """Put the source line first, replacing one the caller already wrote."""
        lines = note.split("\n") if note else []
        if lines and lines[0].startswith(SOURCE_PREFIX):
            lines = lines[1:]
        body = "\n".join(lines).strip("\n")
        line = cls.source_line(source, today)
        return f"{line}\n{body}" if body else line

    @staticmethod
    def is_agent_task(note: str) -> bool:
        """True when a note starts with a source line, i.e. an agent wrote the task."""
        return note.startswith(SOURCE_PREFIX)

    # -- field checks --------------------------------------------------------

    def exception_for(self, source: str) -> SourceException:
        """Return the exception whose key appears in ``source``, or an empty one."""
        lowered = source.lower()
        for key, exc in self.exceptions.items():
            if key != INBOX_KEY and key.lower() in lowered:
                return exc
        return SourceException()

    def rejected_fields(
        self,
        *,
        source: str,
        project_name: str | None = None,
        due: str | None = None,
        defer: str | None = None,
        flagged: bool = False,
        tags: bool = False,
        today: date,
    ) -> list[str]:
        """Return the fields this source may not set, in a stable order.

        Empty ``due``/``defer`` strings clear a date and are always allowed.
        The ``*`` exception counts only when no project is set.
        """
        exc = self.exception_for(source)
        inbox = self.exceptions.get(INBOX_KEY) if project_name is None else None
        rejected: list[str] = []
        if project_name is not None and (
            exc.project is None or exc.project.casefold() != project_name.casefold()
        ):
            rejected.append("project")
        if due and not (
            _due_allowed(exc.due, due, today)
            or (inbox is not None and _due_allowed(inbox.due, due, today))
        ):
            rejected.append("due")
        if defer:
            rejected.append("defer")
        if flagged:
            rejected.append("flagged")
        if tags and not (exc.tags or (inbox is not None and inbox.tags)):
            rejected.append("tags")
        return rejected

    @staticmethod
    def rejection(fields: list[str], *, update: bool = False) -> str:
        """The error an agent sees, naming the contract and the way out."""
        what = "change" if update else "set"
        target = "an agent task still in the Inbox" if update else "a new task"
        return (
            f"Rejected by the OmniFocus write contract: agents may not {what} "
            f"{', '.join(fields)} on {target}. Agent tasks go to the Inbox with no "
            "project, defer or flag; tags and a due date only where the contract "
            "allows them for this source. Leave filing to the user. Pass user_asked "
            "with the user's own words only when the user explicitly asked for this "
            "field."
        )

    # -- log -----------------------------------------------------------------

    def log(self, entry: dict[str, Any]) -> None:
        """Append one JSON line; a failed log write never fails the task write."""
        if self.log_path is None:
            return
        record = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), **entry}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("write log append failed: %s", self.log_path, exc_info=True)


def _due_allowed(rule: str | None, due: str, today: date) -> bool:
    if rule == "any":
        return True
    if rule != "today":
        return False
    try:
        parsed = parse_due(due)
    except click.BadParameter:
        try:
            parsed = datetime.fromisoformat(due)
        except ValueError:
            return False
    return parsed.date() == today


def load_contract(environ: Mapping[str, str] | None = None) -> WriteContract | None:
    """Read the contract from the environment; ``None`` when it is off."""
    env = os.environ if environ is None else environ
    raw = env.get(CONTRACT_ENV)
    if raw is None:
        return None
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ContractError(f"{CONTRACT_ENV} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ContractError(f"{CONTRACT_ENV} must be a JSON object")
    exceptions: dict[str, SourceException] = {}
    for key, value in data.items():
        if not isinstance(value, dict) or set(value) - {"project", "due", "tags"}:
            raise ContractError(f"{CONTRACT_ENV}[{key!r}] allows only 'project', 'due' and 'tags'")
        if value.get("due") not in (None, "today", "any"):
            raise ContractError(f"{CONTRACT_ENV}[{key!r}].due must be 'today' or 'any'")
        if not isinstance(value.get("tags", False), bool):
            raise ContractError(f"{CONTRACT_ENV}[{key!r}].tags must be true or false")
        if key == INBOX_KEY and "project" in value:
            raise ContractError(f"{CONTRACT_ENV}[{INBOX_KEY!r}] is Inbox-only: no 'project'")
        exceptions[key] = SourceException(
            project=value.get("project"), due=value.get("due"), tags=value.get("tags", False)
        )
    log_path = env.get(LOG_ENV)
    return WriteContract(exceptions=exceptions, log_path=Path(log_path) if log_path else None)
