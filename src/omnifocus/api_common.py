"""Shared API helpers for MCP and HTTPS transports."""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import dataclasses
from datetime import UTC, datetime
from typing import Any

from omnifocus.models import Folder, OFModel, Perspective, Project, Tag, Task
from omnifocus.review import ProjectReviewState, compute_project_review_state


def serialise_json(data: Any) -> Any:
    """Convert nested dataclasses and datetimes to JSON-safe values."""
    if isinstance(data, datetime):
        return data.isoformat()
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        return {key: serialise_json(value) for key, value in dataclasses.asdict(data).items()}
    if isinstance(data, dict):
        return {key: serialise_json(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [serialise_json(item) for item in data]
    return data


def parse_optional_date(value: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string or return ``None``."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_optional_utc_datetime(value: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp and normalise it to UTC."""
    parsed = parse_optional_date(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def matching_tag_ids(
    model: OFModel,
    query: str,
    *,
    include_hidden: bool = False,
) -> set[str]:
    """Return all tag ids whose names contain the provided substring."""
    needle = query.lower()
    return {
        tag.id
        for tag in model.tags.values()
        if (include_hidden or tag.hidden is None) and needle in tag.name.lower()
    }


def task_summary(task: Task, model: OFModel) -> dict[str, Any]:
    """Return a concise dict representation of a task.

    ``start`` is OmniFocus's internal name for the defer date; ``defer`` is returned as an explicit
    alias so callers do not have to remember the historical naming mismatch. ``project_status`` and
    ``project_folder_name`` are included so review agents can filter without a separate lookup.
    """
    project = model.projects.get(task.project_id or "")
    folder = model.folders.get(project.folder_id or "") if project else None
    start_iso = task.start.isoformat() if task.start else None
    return {
        "id": task.id,
        "name": task.name,
        "project": project.name if project else None,
        "project_id": task.project_id,
        "project_status": project.status if project else None,
        "project_folder_name": folder.name if folder else None,
        "inbox": task.inbox,
        "flagged": task.flagged,
        "due": task.due.isoformat() if task.due else None,
        "start": start_iso,
        "defer": start_iso,
        "completed": task.completed.isoformat() if task.completed else None,
        "note": task.note,
        "tag_ids": list(task.tag_ids),
        "tag_names": [model.tags[tag_id].name for tag_id in task.tag_ids if tag_id in model.tags],
    }


def project_summary(
    project: Project,
    model: OFModel,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a concise dict representation of a project."""
    folder = model.folders.get(project.folder_id or "")
    review_state = compute_project_review_state(project, now=now)
    return {
        **dataclasses.asdict(project),
        "folder_name": folder.name if folder is not None else None,
        "tag_names": [
            model.tags[tag_id].name for tag_id in project.tag_ids if tag_id in model.tags
        ],
        "review_due": review_state.due,
        "review_basis": review_state.basis,
    }


def project_review_sort_key(
    summary: dict[str, Any],
    review_state: ProjectReviewState,
) -> tuple[int, datetime, int, str, str]:
    """Return a stable ordering key for project review queues."""
    due_at = review_state.due_at or datetime.max.replace(tzinfo=UTC)
    if review_state.basis == "unknown":
        bucket = 2
    elif review_state.due:
        bucket = 0
    else:
        bucket = 1
    rank = int(summary.get("rank", 0))
    name = str(summary.get("name", "")).lower()
    project_id = str(summary.get("id", ""))
    return (bucket, due_at, rank, name, project_id)


def folder_summary(folder: Folder, model: OFModel) -> dict[str, Any]:
    """Return a concise dict representation of a folder."""
    child_folders = sorted(
        (
            candidate
            for candidate in model.folders.values()
            if candidate.parent_folder_id == folder.id
        ),
        key=lambda item: (item.rank, item.name.lower(), item.id),
    )
    child_projects = sorted(
        (project for project in model.projects.values() if project.folder_id == folder.id),
        key=lambda item: (item.rank, item.name.lower(), item.id),
    )
    return {
        **dataclasses.asdict(folder),
        "child_folder_ids": [candidate.id for candidate in child_folders],
        "project_ids": [project.id for project in child_projects],
    }


def tag_summary(tag: Tag, model: OFModel) -> dict[str, Any]:
    """Return a concise dict representation of a tag."""
    child_tags = sorted(
        (candidate for candidate in model.tags.values() if candidate.parent_tag_id == tag.id),
        key=lambda item: (item.rank, item.name.lower(), item.id),
    )
    return {
        **dataclasses.asdict(tag),
        "parent_name": (
            model.tags[tag.parent_tag_id].name if tag.parent_tag_id in model.tags else None
        ),
        "child_tag_ids": [candidate.id for candidate in child_tags],
    }


def _collect_strings(value: Any, out: set[str]) -> None:
    """Add every string leaf of nested dict/list *value* to *out*."""
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, out)


def perspective_summary(perspective: Perspective, model: OFModel) -> dict[str, Any]:
    """Return a perspective with the tags, projects and folders it names resolved.

    Filter rules and focus reference entities by stable id. Rather than assume
    which plist keys hold ids, every string leaf that matches a known tag,
    project or folder id is listed under ``references`` with its type and name.
    """
    strings: set[str] = set()
    _collect_strings(perspective.definition, strings)
    references: dict[str, dict[str, str]] = {}
    for value in sorted(strings):
        if value in model.tags:
            references[value] = {"type": "tag", "name": model.tags[value].name}
        elif value in model.projects:
            references[value] = {"type": "project", "name": model.projects[value].name}
        elif value in model.folders:
            references[value] = {"type": "folder", "name": model.folders[value].name}
    return {**serialise_json(perspective), "references": references}


def validate_folder_parent_change(
    *,
    model: OFModel,
    folder_id: str,
    parent_folder_id: str | None,
    clear_parent: bool,
) -> str | None:
    """Validate requested folder reparenting."""
    if parent_folder_id and clear_parent:
        return "parent_folder_id and clear_parent cannot be combined"
    if parent_folder_id is None:
        return None
    if parent_folder_id not in model.folders:
        return f"Folder not found: {parent_folder_id}"
    if parent_folder_id == folder_id:
        return "Folder cannot be its own parent"
    seen: set[str] = {folder_id}
    current_id: str | None = parent_folder_id
    while current_id is not None:
        if current_id in seen:
            return "Folder move would create a cycle"
        seen.add(current_id)
        parent = model.folders.get(current_id)
        current_id = None if parent is None else parent.parent_folder_id
    return None


def validate_tag_parent_change(
    *,
    model: OFModel,
    tag_id: str,
    parent_tag_id: str | None,
    clear_parent: bool,
) -> str | None:
    """Validate requested tag reparenting."""
    if parent_tag_id and clear_parent:
        return "parent_tag_id and clear_parent cannot be combined"
    if parent_tag_id is None:
        return None
    if parent_tag_id not in model.tags:
        return f"Tag not found: {parent_tag_id}"
    if parent_tag_id == tag_id:
        return "Tag cannot be its own parent"
    seen: set[str] = {tag_id}
    current_id: str | None = parent_tag_id
    while current_id is not None:
        if current_id in seen:
            return "Tag move would create a cycle"
        seen.add(current_id)
        parent = model.tags.get(current_id)
        current_id = None if parent is None else parent.parent_tag_id
    return None
