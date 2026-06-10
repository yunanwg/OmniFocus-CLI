"""Transport-neutral task filtering shared by the CLI, MCP, and HTTP surfaces.

Centralising the filter semantics keeps the CLI table view, the MCP tools, and
the HTTP API in agreement — in particular for the timezone handling that
"completed today" and "due today" depend on.

Timezone model (see :mod:`omnifocus.models`):

* ``Task.completed`` is timezone-aware UTC.
* ``Task.due`` is a *naive* local wall-clock time in the user's timezone.

So "today" must be evaluated in the user's local timezone, and completion
timestamps must be converted to that zone before their calendar date is taken.
The zone comes from ``OF_TIMEZONE`` or ``TZ``; if neither resolves, the system
local timezone is used.
"""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import os
from datetime import UTC, date, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from omnifocus.api_common import matching_tag_ids
from omnifocus.models import OFModel, Task

VALID_TASK_STATUS = ("active", "completed", "dropped", "all")


def user_tz() -> tzinfo:
    """Return the user's local timezone from ``OF_TIMEZONE`` / ``TZ``, else system local."""
    name = os.environ.get("OF_TIMEZONE") or os.environ.get("TZ")
    if name:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError, ValueError:
            pass
    return datetime.now().astimezone().tzinfo  # type: ignore[return-value]


def parse_filter_date(value: str, *, tz: tzinfo | None = None) -> date:
    """Parse an ISO ``YYYY-MM-DD`` date or a relative word into a calendar date.

    Relative words ``today`` / ``yesterday`` / ``tomorrow`` are resolved in *tz*
    (defaulting to :func:`user_tz`). Raises :class:`ValueError` on bad input.
    """
    tz = tz or user_tz()
    token = value.strip().lower()
    today = datetime.now(tz).date()
    if token in ("today", "tod"):
        return today
    if token in ("yesterday", "yd"):
        return today - timedelta(days=1)
    if token in ("tomorrow", "tom"):
        return today + timedelta(days=1)
    return date.fromisoformat(value.strip())


def _completed_local_date(task: Task, tz: tzinfo) -> date | None:
    """Return the task's completion date in *tz*, or ``None`` when not completed."""
    completed = task.completed
    if completed is None:
        return None
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=UTC)
    return completed.astimezone(tz).date()


def _tasks_for_status(model: OFModel, status: str) -> list[Task]:
    """Return the base task set for the requested status keyword."""
    if status == "active":
        return model.active_tasks
    if status == "completed":
        return [task for task in model.tasks.values() if task.completed is not None]
    if status == "dropped":
        return [task for task in model.tasks.values() if task.hidden is not None]
    if status == "all":
        return list(model.tasks.values())
    raise ValueError(f"Invalid status: {status!r}")


def folder_subtree_project_ids(model: OFModel, folder_query: str) -> set[str]:
    """Return ids of projects under any folder matching *folder_query* (substring).

    The match covers the whole subtree: projects inside descendant folders of a
    matched folder are included too, so a single folder name isolates a branch
    of the library (e.g. all work projects under a "Work" folder).
    """
    needle = folder_query.lower()
    subtree = {fid for fid, folder in model.folders.items() if needle in folder.name.lower()}
    if not subtree:
        return set()
    changed = True
    while changed:
        changed = False
        for fid, folder in model.folders.items():
            if fid not in subtree and folder.parent_folder_id in subtree:
                subtree.add(fid)
                changed = True
    return {pid for pid, project in model.projects.items() if project.folder_id in subtree}


def filter_tasks(
    model: OFModel,
    *,
    status: str = "active",
    inbox: bool = False,
    today: bool = False,
    flagged: bool = False,
    due: bool = False,
    project: str | None = None,
    tag: str | None = None,
    tag_id: str | None = None,
    folder: str | None = None,
    completed_on: str | None = None,
    completed_since: str | None = None,
    tz: tzinfo | None = None,
) -> list[Task]:
    """Filter tasks with AND logic across every supplied criterion.

    Substring matching is used for ``project``, ``tag`` and ``folder``;
    ``tag_id`` is an exact identifier. ``completed_on`` / ``completed_since``
    accept an ISO date or ``today`` / ``yesterday``; supplying either promotes a
    default ``active`` status to ``completed`` so the natural "what did I finish
    today" query works without also passing ``status``. Raises
    :class:`ValueError` on an unknown status or an unparseable date.
    """
    tz = tz or user_tz()

    effective_status = status
    if status == "active" and (completed_on or completed_since):
        effective_status = "completed"
    tasks = _tasks_for_status(model, effective_status)

    if completed_on is not None:
        target = parse_filter_date(completed_on, tz=tz)
        tasks = [task for task in tasks if _completed_local_date(task, tz) == target]
    if completed_since is not None:
        since = parse_filter_date(completed_since, tz=tz)
        tasks = [
            task
            for task in tasks
            if (local := _completed_local_date(task, tz)) is not None and local >= since
        ]

    if inbox:
        tasks = [task for task in tasks if task.inbox]
    if today:
        today_local = datetime.now(tz).date()
        tasks = [task for task in tasks if task.due is not None and task.due.date() <= today_local]
    if flagged:
        tasks = [task for task in tasks if task.flagged]
    if due:
        tasks = [task for task in tasks if task.due is not None]
    if project:
        needle = project.lower()
        matching = {pid for pid, item in model.projects.items() if needle in item.name.lower()}
        tasks = [task for task in tasks if task.project_id in matching]
    if tag_id:
        tasks = [task for task in tasks if tag_id in task.tag_ids]
    if tag:
        matches = matching_tag_ids(model, tag)
        tasks = [task for task in tasks if matches.intersection(task.tag_ids)]
    if folder:
        project_ids = folder_subtree_project_ids(model, folder)
        tasks = [task for task in tasks if task.project_id in project_ids]
    return tasks
