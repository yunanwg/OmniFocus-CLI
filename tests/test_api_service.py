"""Tests for :mod:`omnifocus.api_service`."""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from omnifocus.api_service import StoreBackedApiService, default_api_service
from omnifocus.errors import OFHTTPError
from omnifocus.models import Folder, OFModel, Project, Tag, Task

NOW = datetime(2026, 4, 6, 12, 0, 0, tzinfo=UTC)


def _make_model() -> OFModel:
    model = OFModel()
    model.folders["f1"] = Folder(
        id="f1",
        name="Work",
        parent_folder_id=None,
        rank=100,
        added=NOW,
        modified=NOW,
    )
    model.folders["f2"] = Folder(
        id="f2",
        name="Parent",
        parent_folder_id=None,
        rank=200,
        added=NOW,
        modified=NOW,
    )
    model.tags["tag1"] = Tag(
        id="tag1",
        name="@home",
        parent_tag_id=None,
        rank=100,
        added=NOW,
        modified=NOW,
        note="",
        hidden=None,
    )
    model.tags["tag2"] = Tag(
        id="tag2",
        name="@desk",
        parent_tag_id=None,
        rank=200,
        added=NOW,
        modified=NOW,
        note="",
        hidden=None,
    )
    model.projects["p1"] = Project(
        id="p1",
        name="Active project",
        folder_id="f1",
        status="active",
        singleton=False,
        rank=100,
        added=NOW,
        modified=NOW,
        flagged=False,
        due=None,
        start=None,
        note="",
        completed=None,
        tag_ids=("tag1",),
    )
    model.projects["p2"] = Project(
        id="p2",
        name="Inactive project",
        folder_id="f1",
        status="inactive",
        singleton=False,
        rank=200,
        added=NOW,
        modified=NOW,
        flagged=False,
        due=None,
        start=None,
        note="",
        completed=None,
        tag_ids=(),
    )
    model.tasks["t1"] = Task(
        id="t1",
        name="Task",
        parent_task_id="p1",
        project_id="p1",
        inbox=False,
        completed=None,
        flagged=False,
        due=None,
        start=None,
        hidden=None,
        note="",
        rank=100,
        repetition_rule=None,
        estimated_minutes=30,
        tag_ids=("tag1",),
        added=NOW,
        modified=NOW,
    )
    return model


@asynccontextmanager
async def _store_context(store: AsyncMock) -> AsyncIterator[AsyncMock]:
    yield store


def _service(
    *,
    model: OFModel | None = None,
    store: AsyncMock | None = None,
) -> StoreBackedApiService:
    current_model = model or _make_model()
    current_store = store or AsyncMock()

    async def _load(_force_refresh: bool = False) -> OFModel:
        return current_model

    return StoreBackedApiService(
        load_model=_load,
        store_factory=lambda: _store_context(current_store),
    )


class TestStoreBackedApiService:
    @pytest.mark.asyncio
    async def test_add_task_rejects_inactive_project(self) -> None:
        service = _service()

        with pytest.raises(OFHTTPError, match="Project is not active"):
            await service.add_task(name="Task", project_id="p2")

    @pytest.mark.asyncio
    async def test_complete_task_uses_store(self) -> None:
        store = AsyncMock()
        store.complete_task.return_value = {"status": "completed", "task_id": "t1"}
        service = _service(store=store)

        result = await service.complete_task(task_id="t1")

        assert result == {"status": "completed", "task_id": "t1"}
        store.complete_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drop_task_delegates_to_update_task(self) -> None:
        service = _service()
        service.update_task = AsyncMock(return_value={"status": "dropped", "task_id": "t1"})  # type: ignore[method-assign]

        result = await service.drop_task(task_id="t1")

        assert result == {"status": "dropped", "task_id": "t1"}
        service.update_task.assert_awaited_once_with(task_id="t1", dropped=True)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_list_projects_rejects_invalid_status(self) -> None:
        service = _service()

        with pytest.raises(OFHTTPError, match="Invalid status filter"):
            await service.list_projects(status="weird")

    @pytest.mark.asyncio
    async def test_add_project_rejects_invalid_status(self) -> None:
        service = _service()

        with pytest.raises(OFHTTPError, match="status must be active or inactive"):
            await service.add_project(name="Project", status="done")

    @pytest.mark.asyncio
    async def test_add_folder_with_parent_id_calls_store(self) -> None:
        store = AsyncMock()
        store.add_folder.return_value = {"status": "created", "folder_id": "f3"}
        service = _service(store=store)

        result = await service.add_folder(name="Child", parent_folder_id="f2")

        assert result == {"status": "created", "folder_id": "f3"}
        store.add_folder.assert_awaited_once_with(name="Child", parent_folder_id="f2")

    @pytest.mark.asyncio
    async def test_add_folder_without_parent_calls_store(self) -> None:
        store = AsyncMock()
        store.add_folder.return_value = {"status": "created", "folder_id": "f3"}
        service = _service(store=store)

        result = await service.add_folder(name="Root")

        assert result == {"status": "created", "folder_id": "f3"}
        store.add_folder.assert_awaited_once_with(name="Root", parent_folder_id=None)

    @pytest.mark.asyncio
    async def test_add_tag_with_parent_id_calls_store(self) -> None:
        store = AsyncMock()
        store.add_tag.return_value = {"status": "created", "tag_id": "tag3"}
        service = _service(store=store)

        result = await service.add_tag(name="@child", parent_tag_id="tag2", note="hello")

        assert result == {"status": "created", "tag_id": "tag3"}
        store.add_tag.assert_awaited_once_with(name="@child", parent_tag_id="tag2", note="hello")

    @pytest.mark.asyncio
    async def test_add_tag_without_parent_calls_store(self) -> None:
        store = AsyncMock()
        store.add_tag.return_value = {"status": "created", "tag_id": "tag3"}
        service = _service(store=store)

        result = await service.add_tag(name="@root")

        assert result == {"status": "created", "tag_id": "tag3"}
        store.add_tag.assert_awaited_once_with(name="@root", parent_tag_id=None, note="")

    def test_build_updated_project_rejects_conflicting_tag_fields(self) -> None:
        model = _make_model()
        service = _service(model=model)

        with pytest.raises(OFHTTPError, match="tag_ids and clear_tags cannot be combined"):
            service._build_updated_project(  # noqa: SLF001
                model=model,
                project=model.projects["p1"],
                name=None,
                folder_id=None,
                clear_folder=False,
                due=None,
                defer=None,
                flagged=None,
                note=None,
                status=None,
                tag_ids=("tag1",),
                clear_tags=True,
            )

    def test_build_updated_project_rejects_invalid_status(self) -> None:
        model = _make_model()
        service = _service(model=model)

        with pytest.raises(OFHTTPError, match="Invalid project status"):
            service._build_updated_project(  # noqa: SLF001
                model=model,
                project=model.projects["p1"],
                name=None,
                folder_id=None,
                clear_folder=False,
                due=None,
                defer=None,
                flagged=None,
                note=None,
                status="paused",
                tag_ids=None,
                clear_tags=False,
            )

    def test_build_updated_project_can_clear_tags(self) -> None:
        model = _make_model()
        service = _service(model=model)

        updated = service._build_updated_project(  # noqa: SLF001
            model=model,
            project=model.projects["p1"],
            name=None,
            folder_id=None,
            clear_folder=False,
            due=None,
            defer=None,
            flagged=None,
            note=None,
            status=None,
            tag_ids=None,
            clear_tags=True,
        )

        assert updated.tag_ids == ()

    def test_build_updated_project_rejects_unknown_tag_ids(self) -> None:
        model = _make_model()
        service = _service(model=model)

        with pytest.raises(OFHTTPError, match="Unknown tag IDs: missing"):
            service._build_updated_project(  # noqa: SLF001
                model=model,
                project=model.projects["p1"],
                name=None,
                folder_id=None,
                clear_folder=False,
                due=None,
                defer=None,
                flagged=None,
                note=None,
                status=None,
                tag_ids=("missing",),
                clear_tags=False,
            )

    def test_build_updated_project_accepts_valid_tag_ids(self) -> None:
        model = _make_model()
        service = _service(model=model)

        updated = service._build_updated_project(  # noqa: SLF001
            model=model,
            project=model.projects["p1"],
            name=None,
            folder_id=None,
            clear_folder=False,
            due=None,
            defer=None,
            flagged=None,
            note=None,
            status=None,
            tag_ids=("tag2",),
            clear_tags=False,
        )

        assert updated.tag_ids == ("tag2",)


class TestDefaultApiService:
    @pytest.mark.asyncio
    async def test_default_api_service_uses_ofocusstore_context(self) -> None:
        model = _make_model()
        store = AsyncMock()
        store.load.return_value = model

        @asynccontextmanager
        async def _context() -> AsyncIterator[AsyncMock]:
            yield store

        with patch("omnifocus.api_service.OFocusStore.from_env", side_effect=lambda: _context()):
            service = default_api_service()
            result = await service.sync_now()

        assert result == {
            "status": "synced",
            "tasks": 1,
            "projects": 2,
            "folders": 2,
            "tags": 2,
        }
        store.load.assert_awaited_once_with(force_refresh=True)


class TestListTasksFiltering:
    @pytest.mark.asyncio
    async def test_invalid_status_raises_422(self) -> None:
        with pytest.raises(OFHTTPError) as excinfo:
            await _service().list_tasks(status="bogus")
        assert excinfo.value.status_code == 422

    @pytest.mark.asyncio
    async def test_unparseable_completed_date_raises_422(self) -> None:
        with pytest.raises(OFHTTPError) as excinfo:
            await _service().list_tasks(completed_on="not-a-date")
        assert excinfo.value.status_code == 422

    @pytest.mark.asyncio
    async def test_status_completed_returns_completed_tasks(self) -> None:
        model = _make_model()
        model.tasks["done"] = Task(
            id="done",
            name="Done task",
            parent_task_id="p1",
            project_id="p1",
            inbox=False,
            completed=datetime(2026, 6, 1, 10, 0, tzinfo=UTC),
            flagged=False,
            due=None,
            start=None,
            hidden=None,
            note="",
            rank=0,
            repetition_rule=None,
            estimated_minutes=None,
            tag_ids=(),
            added=NOW,
            modified=NOW,
        )
        results = await _service(model=model).list_tasks(status="completed")
        assert {row["id"] for row in results} == {"done"}

    @pytest.mark.asyncio
    async def test_folder_filter_matches_subtree(self) -> None:
        results = await _service().list_tasks(folder="Work")
        assert {row["id"] for row in results} == {"t1"}
