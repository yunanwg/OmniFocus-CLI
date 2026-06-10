"""Shared store-backed service layer for MCP and HTTPS transports."""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import dataclasses
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any, Protocol

import click

from omnifocus.api_common import (
    folder_summary,
    matching_tag_ids,
    parse_optional_date,
    parse_optional_utc_datetime,
    project_review_sort_key,
    project_summary,
    tag_summary,
    task_summary,
    validate_folder_parent_change,
    validate_tag_parent_change,
)
from omnifocus.dateparse import parse_due
from omnifocus.errors import OFHTTPError
from omnifocus.filters import VALID_TASK_STATUS, filter_tasks
from omnifocus.formatting import build_folder_tree_data
from omnifocus.fuzzy import find_tasks
from omnifocus.models import Folder, OFModel, Project, Tag, Task
from omnifocus.review import compute_project_review_state, mark_project_reviewed
from omnifocus.store import OFocusStore


class StoreContextFactory(Protocol):
    """Callable protocol returning an async OFocusStore context manager."""

    def __call__(self) -> AbstractAsyncContextManager[OFocusStore]:
        """Return a context manager yielding an ``OFocusStore``."""


class StoreBackedApiService:
    """Transport-neutral service layer backed by ``OFocusStore``.

    This class is the shared public contract boundary for both MCP and HTTPS transports.
    It owns request-level validation, stable-ID lookup rules, and summary serialization,
    while delegating persistence and sync mechanics to :class:`OFocusStore`.
    """

    def __init__(
        self,
        *,
        load_model: Callable[[bool], Awaitable[OFModel]],
        store_factory: StoreContextFactory,
    ) -> None:
        """Initialise the service with injectable store/model hooks."""
        self._load_model = load_model
        self._store_factory = store_factory

    async def list_tasks(
        self,
        *,
        inbox: bool = False,
        today: bool = False,
        flagged: bool = False,
        due: bool = False,
        project: str | None = None,
        tag: str | None = None,
        tag_id: str | None = None,
        status: str = "active",
        completed_on: str | None = None,
        completed_since: str | None = None,
        folder: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List tasks using the shared filter model for MCP and HTTP.

        Filtering semantics intentionally match the user-facing surfaces: all supplied filters are
        applied with AND logic; `project`, `tag` and `folder` use substring matching, and `tag_id`
        uses an exact stable identifier. `status` selects the base set
        (active/completed/dropped/all); `completed_on` / `completed_since` accept an ISO date or
        `today`/`yesterday` and report on finished work (they imply `status=completed`).
        """
        if status not in VALID_TASK_STATUS:
            raise OFHTTPError("Invalid status filter", status_code=422, code="validation_error")
        model = await self._load_model(False)
        if tag_id:
            self._require_tag(model, tag_id)
        try:
            tasks = filter_tasks(
                model,
                status=status,
                inbox=inbox,
                today=today,
                flagged=flagged,
                due=due,
                project=project,
                tag=tag,
                tag_id=tag_id,
                folder=folder,
                completed_on=completed_on,
                completed_since=completed_since,
            )
        except ValueError as exc:
            raise OFHTTPError(str(exc), status_code=422, code="validation_error") from exc
        return [task_summary(task, model) for task in tasks[:limit]]

    async def search_tasks(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search tasks by fuzzy name or stable ID and return scored summaries."""
        model = await self._load_model(False)
        results = find_tasks(query, model.active_tasks, limit=limit)
        return [
            {"score": round(result.score, 3), **task_summary(result.task, model)}
            for result in results
        ]

    async def get_task(self, *, task_id: str) -> dict[str, Any]:
        """Return a single task summary by stable OmniFocus task ID."""
        model = await self._load_model(False)
        task = self._require_task(model, task_id)
        return task_summary(task, model)

    async def add_task(
        self,
        *,
        name: str,
        project_id: str | None = None,
        due: str | None = None,
        flagged: bool = False,
        note: str = "",
    ) -> dict[str, str]:
        """Create a task, optionally resolving it into an active project container."""
        if not name:
            raise OFHTTPError("name is required", status_code=422, code="validation_error")
        due_dt = self._parse_due_like(due, field="due")
        parent_task_id: str | None = None
        inbox = True
        if project_id is not None:
            model = await self._load_model(False)
            project = self._require_project(model, project_id)
            if project.status != "active":
                raise OFHTTPError(
                    f"Project is not active: {project_id}",
                    status_code=409,
                    code="conflict",
                )
            parent_task_id = project.id
            inbox = False
        async with self._store_factory() as store:
            return await store.add_task(
                name=name,
                parent_task_id=parent_task_id,
                inbox=inbox,
                flagged=flagged,
                due_dt=due_dt,
                note=note,
            )

    async def update_task(
        self,
        *,
        task_id: str,
        name: str | None = None,
        project_id: str | None = None,
        clear_project: bool = False,
        inbox: bool | None = None,
        due: str | None = None,
        defer: str | None = None,
        flagged: bool | None = None,
        note: str | None = None,
        estimate: int | str | None = None,
        tag_ids: tuple[str, ...] | None = None,
        clear_tags: bool = False,
        dropped: bool | None = None,
    ) -> dict[str, str]:
        """Update a task by stable ID using the shared mutation semantics.

        The method validates conflicting move operations, tag replacement versus clearing,
        due/defer parsing, estimate coercion, and dropped/hidden transitions before persistence.
        """
        model = await self._load_model(False)
        task = self._require_task(model, task_id)
        updated = self._build_updated_task(
            model=model,
            task=task,
            name=name,
            project_id=project_id,
            clear_project=clear_project,
            inbox=inbox,
            due=due,
            defer=defer,
            flagged=flagged,
            note=note,
            estimate=estimate,
            tag_ids=tag_ids,
            clear_tags=clear_tags,
            dropped=dropped,
        )
        async with self._store_factory() as store:
            if dropped is True:
                return await store.drop_task(updated)
            return await store.update_task(updated)

    async def complete_task(self, *, task_id: str) -> dict[str, str]:
        """Mark a task complete by stable ID."""
        model = await self._load_model(False)
        task = self._require_task(model, task_id)
        async with self._store_factory() as store:
            return await store.complete_task(task)

    async def drop_task(self, *, task_id: str) -> dict[str, str]:
        """Drop a task by stable ID via the same update path used by other transports."""
        return await self.update_task(task_id=task_id, dropped=True)

    async def list_projects(
        self,
        *,
        status: str = "active",
        tag: str | None = None,
        tag_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List projects with the current transport-shared filter semantics.

        Status filtering is exact, while `tag` uses substring matching and `tag_id` uses an exact
        stable identifier.
        """
        if status not in {"active", "all", "inactive", "done", "dropped"}:
            raise OFHTTPError("Invalid status filter", status_code=422, code="validation_error")
        model = await self._load_model(False)
        projects = [
            project
            for project in model.projects.values()
            if status == "all" or project.status == status
        ]
        if tag_id:
            self._require_tag(model, tag_id)
            projects = [project for project in projects if tag_id in project.tag_ids]
        if tag:
            matches = matching_tag_ids(model, tag)
            projects = [project for project in projects if matches.intersection(project.tag_ids)]
        summaries = [project_summary(project, model) for project in projects]
        return summaries if limit is None else summaries[:limit]

    async def get_project(self, *, project_id: str) -> dict[str, Any]:
        """Return a single project summary by stable OmniFocus project ID."""
        model = await self._load_model(False)
        project = self._require_project(model, project_id)
        return project_summary(project, model)

    async def add_project(
        self,
        *,
        name: str,
        folder_id: str | None = None,
        due: str | None = None,
        defer: str | None = None,
        flagged: bool = False,
        note: str = "",
        status: str = "active",
    ) -> dict[str, str]:
        """Create a project, optionally assigning it to a folder by stable ID."""
        if not name:
            raise OFHTTPError("name is required", status_code=422, code="validation_error")
        if status not in {"active", "inactive"}:
            raise OFHTTPError(
                "status must be active or inactive",
                status_code=422,
                code="validation_error",
            )
        if folder_id is not None:
            model = await self._load_model(False)
            self._require_folder(model, folder_id)
        async with self._store_factory() as store:
            return await store.add_project(
                name=name,
                folder_id=folder_id,
                status=status,
                flagged=flagged,
                due_dt=self._parse_due_like(due, field="due"),
                start_dt=self._parse_due_like(defer, field="defer"),
                note=note,
            )

    async def update_project(
        self,
        *,
        project_id: str,
        name: str | None = None,
        folder_id: str | None = None,
        clear_folder: bool = False,
        due: str | None = None,
        defer: str | None = None,
        flagged: bool | None = None,
        note: str | None = None,
        status: str | None = None,
        tag_ids: tuple[str, ...] | None = None,
        clear_tags: bool = False,
    ) -> dict[str, str]:
        """Update a project by stable ID using shared validation and status semantics.

        The method validates folder moves, tag replacement versus clearing, due/defer parsing, and
        status transitions including `done` and `dropped`.
        """
        model = await self._load_model(False)
        project = self._require_project(model, project_id)
        updated = self._build_updated_project(
            model=model,
            project=project,
            name=name,
            folder_id=folder_id,
            clear_folder=clear_folder,
            due=due,
            defer=defer,
            flagged=flagged,
            note=note,
            status=status,
            tag_ids=tag_ids,
            clear_tags=clear_tags,
        )
        async with self._store_factory() as store:
            if updated.status == "done":
                return await store.complete_project(updated)
            if updated.status == "dropped":
                return await store.drop_project(updated)
            return await store.update_project(updated)

    async def complete_project(self, *, project_id: str) -> dict[str, str]:
        """Mark a project complete by stable ID."""
        model = await self._load_model(False)
        project = self._require_project(model, project_id)
        async with self._store_factory() as store:
            return await store.complete_project(project)

    async def list_projects_for_review(
        self,
        *,
        due_only: bool = True,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List projects for review, sorted by urgency and review metadata.

        By default only active and inactive projects that are currently due for review are
        returned; callers can opt into a wider review queue via `due_only=False`.
        """
        model = await self._load_model(False)
        now = datetime.now(UTC)
        candidates = [
            project
            for project in model.projects.values()
            if project.status in {"active", "inactive"}
        ]
        summaries = [
            (
                project_summary(project, model, now=now),
                compute_project_review_state(project, now=now),
            )
            for project in candidates
        ]
        if due_only:
            summaries = [item for item in summaries if item[1].due]
        summaries.sort(key=lambda item: project_review_sort_key(item[0], item[1]))
        return [summary for summary, _state in summaries[:limit]]

    async def mark_project_reviewed(
        self,
        *,
        project_id: str,
        reviewed_at: str | None = None,
    ) -> dict[str, Any]:
        """Stamp a project as reviewed and return its updated review-aware summary."""
        review_dt = parse_optional_utc_datetime(reviewed_at)
        if reviewed_at is not None and review_dt is None:
            raise OFHTTPError(
                f"Invalid reviewed_at timestamp: {reviewed_at!r}",
                status_code=422,
                code="validation_error",
            )
        model = await self._load_model(False)
        project = self._require_project(model, project_id)
        updated_project, recalculated = mark_project_reviewed(project, reviewed_at=review_dt)
        async with self._store_factory() as store:
            await store.mark_project_reviewed(project, reviewed_at=review_dt)
        summary = project_summary(updated_project, model, now=review_dt or datetime.now(UTC))
        summary["next_review_recalculated"] = recalculated
        summary["status"] = "reviewed"
        return summary

    async def list_folders(self) -> list[dict[str, Any]]:
        """List all folders with direct child folder and project references."""
        model = await self._load_model(False)
        folders = sorted(
            model.folders.values(),
            key=lambda folder: (folder.rank, folder.name.lower()),
        )
        return [folder_summary(folder, model) for folder in folders]

    async def get_folder(self, *, folder_id: str) -> dict[str, Any]:
        """Return a single folder summary by stable OmniFocus folder ID."""
        model = await self._load_model(False)
        folder = self._require_folder(model, folder_id)
        return folder_summary(folder, model)

    async def get_folder_tree(self) -> dict[str, Any]:
        """Return the nested folder tree used by CLI and HTTP folder views."""
        model = await self._load_model(False)
        return build_folder_tree_data(model.folders, model.projects)

    async def add_folder(
        self,
        *,
        name: str,
        parent_folder_id: str | None = None,
    ) -> dict[str, str]:
        """Create a folder, optionally attaching it to a parent by stable ID."""
        if not name:
            raise OFHTTPError("name is required", status_code=422, code="validation_error")
        if parent_folder_id is not None:
            model = await self._load_model(False)
            self._require_folder(model, parent_folder_id)
        async with self._store_factory() as store:
            return await store.add_folder(name=name, parent_folder_id=parent_folder_id)

    async def update_folder(
        self,
        *,
        folder_id: str,
        name: str | None = None,
        parent_folder_id: str | None = None,
        clear_parent: bool = False,
    ) -> dict[str, str]:
        """Rename or move a folder after validating parent existence and cycles."""
        model = await self._load_model(False)
        folder = self._require_folder(model, folder_id)
        validation_error = validate_folder_parent_change(
            model=model,
            folder_id=folder_id,
            parent_folder_id=parent_folder_id,
            clear_parent=clear_parent,
        )
        if validation_error is not None:
            raise OFHTTPError(validation_error, status_code=409, code="conflict")
        new_parent_folder_id = (
            parent_folder_id
            if parent_folder_id is not None
            else (None if clear_parent else folder.parent_folder_id)
        )
        updated = dataclasses.replace(
            folder,
            name=name if name is not None else folder.name,
            parent_folder_id=new_parent_folder_id,
            modified=datetime.now(UTC),
        )
        async with self._store_factory() as store:
            return await store.update_folder(updated)

    async def drop_folder(self, *, folder_id: str) -> dict[str, str]:
        """Drop a folder by stable ID."""
        model = await self._load_model(False)
        folder = self._require_folder(model, folder_id)
        async with self._store_factory() as store:
            return await store.drop_folder(folder)

    async def list_tags(self, *, include_hidden: bool = False) -> list[dict[str, Any]]:
        """List tags, excluding hidden or dropped tags by default."""
        model = await self._load_model(False)
        tags = [tag for tag in model.tags.values() if include_hidden or tag.hidden is None]
        tags.sort(key=lambda item: (item.rank, item.name.lower(), item.id))
        return [tag_summary(tag, model) for tag in tags]

    async def get_tag(self, *, tag_id: str) -> dict[str, Any]:
        """Return a single tag summary by stable OmniFocus tag ID."""
        model = await self._load_model(False)
        tag = self._require_tag(model, tag_id)
        return tag_summary(tag, model)

    async def add_tag(
        self,
        *,
        name: str,
        parent_tag_id: str | None = None,
        note: str = "",
    ) -> dict[str, str]:
        """Create a tag, optionally attaching it to a parent by stable ID."""
        if not name:
            raise OFHTTPError("name is required", status_code=422, code="validation_error")
        if parent_tag_id is not None:
            model = await self._load_model(False)
            self._require_tag(model, parent_tag_id)
        async with self._store_factory() as store:
            return await store.add_tag(name=name, parent_tag_id=parent_tag_id, note=note)

    async def update_tag(
        self,
        *,
        tag_id: str,
        name: str | None = None,
        parent_tag_id: str | None = None,
        clear_parent: bool = False,
        note: str | None = None,
    ) -> dict[str, str]:
        """Rename or move a tag after validating parent existence and cycles."""
        model = await self._load_model(False)
        tag = self._require_tag(model, tag_id)
        validation_error = validate_tag_parent_change(
            model=model,
            tag_id=tag_id,
            parent_tag_id=parent_tag_id,
            clear_parent=clear_parent,
        )
        if validation_error is not None:
            raise OFHTTPError(validation_error, status_code=409, code="conflict")
        new_parent_tag_id = (
            parent_tag_id
            if parent_tag_id is not None
            else (None if clear_parent else tag.parent_tag_id)
        )
        updated = dataclasses.replace(
            tag,
            name=name if name is not None else tag.name,
            parent_tag_id=new_parent_tag_id,
            modified=datetime.now(UTC),
            note=note if note is not None else tag.note,
        )
        async with self._store_factory() as store:
            return await store.update_tag(updated)

    async def drop_tag(self, *, tag_id: str) -> dict[str, str]:
        """Drop a tag by stable ID."""
        model = await self._load_model(False)
        tag = self._require_tag(model, tag_id)
        async with self._store_factory() as store:
            return await store.drop_tag(tag)

    async def sync_now(self) -> dict[str, Any]:
        """Force a fresh sync and return top-level object counts for operator surfaces."""
        model = await self._load_model(True)
        return {
            "status": "synced",
            "tasks": len(model.tasks),
            "projects": len(model.projects),
            "folders": len(model.folders),
            "tags": len(model.tags),
        }

    def _require_task(self, model: OFModel, task_id: str) -> Task:
        """Return a task by id or raise an HTTP 404 error."""
        task = model.tasks.get(task_id)
        if task is None:
            raise OFHTTPError(f"Task not found: {task_id}", status_code=404, code="not_found")
        return task

    def _require_project(self, model: OFModel, project_id: str) -> Project:
        """Return a project by id or raise an HTTP 404 error."""
        project = model.projects.get(project_id)
        if project is None:
            raise OFHTTPError(
                f"Project not found: {project_id}",
                status_code=404,
                code="not_found",
            )
        return project

    def _require_folder(self, model: OFModel, folder_id: str) -> Folder:
        """Return a folder by id or raise an HTTP 404 error."""
        folder = model.folders.get(folder_id)
        if folder is None:
            raise OFHTTPError(f"Folder not found: {folder_id}", status_code=404, code="not_found")
        return folder

    def _require_tag(self, model: OFModel, tag_id: str) -> Tag:
        """Return a tag by id or raise an HTTP 404 error."""
        tag = model.tags.get(tag_id)
        if tag is None:
            raise OFHTTPError(f"Tag not found: {tag_id}", status_code=404, code="not_found")
        return tag

    def _parse_due_like(self, value: str | None, *, field: str) -> datetime | None:
        """Parse ISO or CLI-style natural date inputs."""
        if value is None or value == "":
            return None
        try:
            return parse_due(value)
        except click.BadParameter:
            parsed = parse_optional_date(value)
            if parsed is None:
                raise OFHTTPError(
                    f"Invalid {field} date: {value!r}",
                    status_code=422,
                    code="validation_error",
                ) from None
            return parsed

    def _build_updated_task(
        self,
        *,
        model: OFModel,
        task: Task,
        name: str | None,
        project_id: str | None,
        clear_project: bool,
        inbox: bool | None,
        due: str | None,
        defer: str | None,
        flagged: bool | None,
        note: str | None,
        estimate: int | str | None,
        tag_ids: tuple[str, ...] | None,
        clear_tags: bool,
        dropped: bool | None,
    ) -> Task:
        """Return a validated updated task object."""
        if project_id and clear_project:
            raise OFHTTPError(
                "project_id and clear_project cannot be combined",
                status_code=409,
                code="conflict",
            )
        if project_id and inbox is True:
            raise OFHTTPError(
                "project_id and inbox=true cannot be combined",
                status_code=409,
                code="conflict",
            )
        if clear_project and inbox is False:
            raise OFHTTPError(
                "clear_project cannot be combined with inbox=false",
                status_code=409,
                code="conflict",
            )
        if clear_tags and tag_ids is not None:
            raise OFHTTPError(
                "tag_ids and clear_tags cannot be combined",
                status_code=409,
                code="conflict",
            )

        new_parent_task_id = task.parent_task_id
        new_project_id = task.project_id
        new_inbox = task.inbox

        if project_id:
            project = self._require_project(model, project_id)
            if project.status != "active":
                raise OFHTTPError(
                    f"Project is not active: {project_id}",
                    status_code=409,
                    code="conflict",
                )
            new_parent_task_id = project.id
            new_project_id = project.id
            new_inbox = False
        elif clear_project or inbox is True:
            new_parent_task_id = None
            new_project_id = None
            new_inbox = True

        estimate_value = task.estimated_minutes
        if estimate is not None:
            if estimate == "":
                estimate_value = None
            else:
                try:
                    estimate_value = int(estimate)
                except TypeError, ValueError:
                    raise OFHTTPError(
                        f"Invalid estimate: {estimate!r}",
                        status_code=422,
                        code="validation_error",
                    ) from None

        if clear_tags:
            resolved_tag_ids: tuple[str, ...] = ()
        elif tag_ids is not None:
            missing = [tag_id for tag_id in tag_ids if tag_id not in model.tags]
            if missing:
                raise OFHTTPError(
                    f"Unknown tag IDs: {', '.join(missing)}",
                    status_code=404,
                    code="not_found",
                )
            resolved_tag_ids = tag_ids
        else:
            resolved_tag_ids = task.tag_ids

        hidden_value = task.hidden
        if dropped is True:
            hidden_value = datetime.now(UTC)
        elif dropped is False:
            hidden_value = None

        return dataclasses.replace(
            task,
            name=name if name is not None else task.name,
            parent_task_id=new_parent_task_id,
            project_id=new_project_id,
            inbox=new_inbox,
            flagged=task.flagged if flagged is None else flagged,
            note=task.note if note is None else note,
            due=self._parse_due_like(due, field="due") if due is not None else task.due,
            start=self._parse_due_like(defer, field="defer") if defer is not None else task.start,
            estimated_minutes=estimate_value,
            tag_ids=resolved_tag_ids,
            hidden=hidden_value,
            modified=datetime.now(UTC),
        )

    def _build_updated_project(
        self,
        *,
        model: OFModel,
        project: Project,
        name: str | None,
        folder_id: str | None,
        clear_folder: bool,
        due: str | None,
        defer: str | None,
        flagged: bool | None,
        note: str | None,
        status: str | None,
        tag_ids: tuple[str, ...] | None,
        clear_tags: bool,
    ) -> Project:
        """Return a validated updated project object."""
        if folder_id and clear_folder:
            raise OFHTTPError(
                "folder_id and clear_folder cannot be combined",
                status_code=409,
                code="conflict",
            )
        if clear_tags and tag_ids is not None:
            raise OFHTTPError(
                "tag_ids and clear_tags cannot be combined",
                status_code=409,
                code="conflict",
            )
        if status is not None and status not in {"active", "inactive", "done", "dropped"}:
            raise OFHTTPError("Invalid project status", status_code=422, code="validation_error")

        if folder_id is not None:
            self._require_folder(model, folder_id)
            new_folder_id = folder_id
        elif clear_folder:
            new_folder_id = None
        else:
            new_folder_id = project.folder_id

        if clear_tags:
            resolved_tag_ids: tuple[str, ...] = ()
        elif tag_ids is not None:
            missing = [tag_id for tag_id in tag_ids if tag_id not in model.tags]
            if missing:
                raise OFHTTPError(
                    f"Unknown tag IDs: {', '.join(missing)}",
                    status_code=404,
                    code="not_found",
                )
            resolved_tag_ids = tag_ids
        else:
            resolved_tag_ids = project.tag_ids

        now = datetime.now(UTC)
        updated = dataclasses.replace(
            project,
            name=name if name is not None else project.name,
            folder_id=new_folder_id,
            status=status if status is not None else project.status,
            modified=now,
            flagged=project.flagged if flagged is None else flagged,
            due=self._parse_due_like(due, field="due") if due is not None else project.due,
            start=(
                self._parse_due_like(defer, field="defer") if defer is not None else project.start
            ),
            note=project.note if note is None else note,
            tag_ids=resolved_tag_ids,
        )
        if status == "done" and updated.completed is None:
            updated = dataclasses.replace(updated, completed=now)
        return updated


def default_api_service() -> StoreBackedApiService:
    """Return the default production API service instance.

    The returned service uses environment-derived store configuration and the normal model-loading
    path used in production CLI, MCP, and HTTP deployments.
    """

    async def _load(force_refresh: bool = False) -> OFModel:
        async with OFocusStore.from_env() as store:
            return await store.load(force_refresh=force_refresh)

    return StoreBackedApiService(load_model=_load, store_factory=OFocusStore.from_env)
