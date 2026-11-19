"""Tests for the agent write contract and its enforcement in the MCP server."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from omnifocus.mcp_server import _handle_add_task, _handle_update_task, list_tools
from omnifocus.write_contract import (
    CONTRACT_ENV,
    LOG_ENV,
    ContractError,
    SourceException,
    WriteContract,
    load_contract,
)
from tests.test_mcp_server import _make_model, _mock_store, _parse_response

TODAY = date.today()
EXCEPTIONS = json.dumps(
    {
        "rental": {"project": "Engineering", "due": "today"},
        "hellofresh": {"project": "Engineering", "due": "any"},
    }
)


# ---------------------------------------------------------------------------
# write_contract module
# ---------------------------------------------------------------------------


class TestLoadContract:
    def test_off_when_unset(self) -> None:
        assert load_contract({}) is None

    def test_empty_means_no_exceptions(self) -> None:
        contract = load_contract({CONTRACT_ENV: ""})
        assert contract == WriteContract()

    def test_reads_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(CONTRACT_ENV, "{}")
        monkeypatch.setenv(LOG_ENV, "/var/log/of/x.jsonl")
        contract = load_contract()
        assert contract is not None
        assert contract.log_path == Path("/var/log/of/x.jsonl")

    def test_parses_exceptions(self) -> None:
        contract = load_contract({CONTRACT_ENV: EXCEPTIONS})
        assert contract is not None
        assert contract.exceptions["rental"] == SourceException(project="Engineering", due="today")
        assert contract.log_path is None

    @pytest.mark.parametrize(
        "raw",
        [
            "not json",
            "[]",
            '{"rental": "x"}',
            '{"rental": {"tags": true}}',
            '{"rental": {"due": "tomorrow"}}',
        ],
    )
    def test_rejects_bad_config(self, raw: str) -> None:
        with pytest.raises(ContractError):
            load_contract({CONTRACT_ENV: raw})


class TestSourceLine:
    def test_source_line(self) -> None:
        line = WriteContract.source_line("via Claude Code · rental sweep ", date(2026, 11, 19))
        assert line == "via Claude Code · rental sweep · 2026-11-19"

    def test_stamp_empty_note(self) -> None:
        assert WriteContract.stamp_note("", "Duet · daily", date(2026, 11, 19)) == (
            "via Duet · daily · 2026-11-19"
        )

    def test_stamp_prepends(self) -> None:
        stamped = WriteContract.stamp_note("Call 01 23", "Duet · daily", date(2026, 11, 19))
        assert stamped == "via Duet · daily · 2026-11-19\nCall 01 23"

    def test_stamp_replaces_existing_line(self) -> None:
        stamped = WriteContract.stamp_note(
            "via old · x · 2026-01-01\n\nCall", "Duet · daily", date(2026, 11, 19)
        )
        assert stamped == "via Duet · daily · 2026-11-19\nCall"

    def test_is_agent_task(self) -> None:
        assert WriteContract.is_agent_task("via x · y · 2026-11-19\nbody")
        assert not WriteContract.is_agent_task("my own note")


class TestRejectedFields:
    contract = WriteContract(
        exceptions={
            "rental": SourceException(project="Engineering", due="today"),
            "hellofresh": SourceException(project="Engineering", due="any"),
        }
    )

    def test_plain_inbox_task_passes(self) -> None:
        assert self.contract.rejected_fields(source="Duet · daily", today=TODAY) == []

    def test_everything_rejected_without_exception(self) -> None:
        rejected = self.contract.rejected_fields(
            source="Duet · daily",
            project_name="Engineering",
            due="today",
            defer="today",
            flagged=True,
            tags=True,
            today=TODAY,
        )
        assert rejected == ["project", "due", "defer", "flagged", "tags"]

    def test_clearing_dates_is_allowed(self) -> None:
        assert self.contract.rejected_fields(source="x", due="", defer="", today=TODAY) == []

    def test_rental_project_and_due_today(self) -> None:
        assert (
            self.contract.rejected_fields(
                source="Claude Code · Rental sweep",
                project_name="engineering",
                due="today",
                today=TODAY,
            )
            == []
        )

    def test_rental_iso_due_today(self) -> None:
        iso = f"{TODAY.isoformat()}T18:00:00"
        assert self.contract.rejected_fields(source="rental", due=iso, today=TODAY) == []

    def test_rental_other_project_and_later_due(self) -> None:
        later = (TODAY + timedelta(days=3)).isoformat()
        rejected = self.contract.rejected_fields(
            source="rental", project_name="Operations", due=later, today=TODAY
        )
        assert rejected == ["project", "due"]

    def test_rental_unparseable_due(self) -> None:
        assert self.contract.rejected_fields(source="rental", due="??", today=TODAY) == ["due"]

    def test_rental_still_no_defer_flag_tags(self) -> None:
        rejected = self.contract.rejected_fields(
            source="rental", defer="today", flagged=True, tags=True, today=TODAY
        )
        assert rejected == ["defer", "flagged", "tags"]

    def test_hellofresh_any_due(self) -> None:
        later = (TODAY + timedelta(days=5)).isoformat()
        assert (
            self.contract.rejected_fields(
                source="Duet · HelloFresh weekly picks",
                project_name="Engineering",
                due=later,
                today=TODAY,
            )
            == []
        )

    def test_rejection_message_names_contract(self) -> None:
        msg = WriteContract.rejection(["due", "flagged"])
        assert "OmniFocus write contract" in msg
        assert "set due, flagged on a new task" in msg
        assert "user_asked" in msg
        upd = WriteContract.rejection(["tags"], update=True)
        assert "change tags on an agent task still in the Inbox" in upd


class TestLog:
    def test_no_path_is_noop(self) -> None:
        WriteContract().log({"tool": "add_task"})

    def test_appends_json_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "log.jsonl"
        contract = WriteContract(log_path=path)
        contract.log({"tool": "add_task", "source": "Duet · daily"})
        contract.log({"tool": "update_task"})
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert [line["tool"] for line in lines] == ["add_task", "update_task"]
        assert lines[0]["source"] == "Duet · daily"
        assert "time" in lines[0]

    def test_failed_append_does_not_raise(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("")
        WriteContract(log_path=blocker / "log.jsonl").log({"tool": "add_task"})


# ---------------------------------------------------------------------------
# MCP server enforcement
# ---------------------------------------------------------------------------


@pytest.fixture
def contract_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    log = tmp_path / "write-log.jsonl"
    monkeypatch.setenv(CONTRACT_ENV, EXCEPTIONS)
    monkeypatch.setenv(LOG_ENV, str(log))
    return log


def _log_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def _add(args: dict[str, Any], model: Any = None) -> tuple[Any, Any]:
    mock = _mock_store()
    with patch("omnifocus.mcp_server._load_model", AsyncMock(return_value=model or _make_model())):
        with patch("omnifocus.mcp_server.OFocusStore.from_env", return_value=mock):
            result = await _handle_add_task(args)
    return _parse_response(result), mock


async def _update(args: dict[str, Any], model: Any) -> tuple[Any, Any]:
    mock = _mock_store(model)
    with patch("omnifocus.mcp_server._load_model", AsyncMock(return_value=model)):
        with patch("omnifocus.mcp_server.OFocusStore.from_env", return_value=mock):
            result = await _handle_update_task(args)
    return _parse_response(result), mock


def _agent_model(*, inbox: bool = True, note: str | None = None) -> Any:
    model = _make_model()
    model.tasks["t2"] = dataclasses.replace(
        model.tasks["t2"],
        inbox=inbox,
        note=note if note is not None else "via Duet · daily · 2026-11-19\nCall them",
    )
    return model


class TestSchema:
    @pytest.mark.asyncio
    async def test_schema_unchanged_when_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(CONTRACT_ENV, raising=False)
        tools = {tool.name: tool for tool in await list_tools()}
        assert "source" not in tools["add_task"].inputSchema["properties"]

    @pytest.mark.asyncio
    async def test_schema_requires_source_when_on(self, contract_on: Path) -> None:
        tools = {tool.name: tool for tool in await list_tools()}
        add = tools["add_task"]
        assert add.inputSchema["required"] == ["name", "source"]
        assert "user_asked" in add.inputSchema["properties"]
        assert "write contract" in (add.description or "")
        assert "user_asked" in tools["update_task"].inputSchema["properties"]
        assert "source" not in tools["update_task"].inputSchema["properties"]
        assert tools["add_task"].annotations is not None


class TestAddTaskContract:
    @pytest.mark.asyncio
    async def test_source_required(self, contract_on: Path) -> None:
        data, mock = await _add({"name": "T"})
        assert "source is required" in data["error"]
        mock.add_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inbox_task_gets_source_line_and_log(self, contract_on: Path) -> None:
        data, mock = await _add({"name": "Call bank", "note": "01 23", "source": "Duet · daily"})
        assert data["status"] == "created"
        note = mock.add_task.await_args.kwargs["note"]
        assert note == f"via Duet · daily · {TODAY.isoformat()}\n01 23"
        [entry] = _log_lines(contract_on)
        assert entry["tool"] == "add_task"
        assert entry["source"] == "Duet · daily"
        assert entry["task_id"] == "new-task"
        assert entry["fields"] == ["name", "note"]
        assert "user_asked" not in entry

    @pytest.mark.asyncio
    async def test_rejects_due_flag_project(self, contract_on: Path) -> None:
        data, mock = await _add(
            {
                "name": "T",
                "source": "Duet · daily",
                "due": "today",
                "flagged": True,
                "project": "Operations",
                "defer": "today",
                "tag_ids": ["tag1"],
            }
        )
        assert "OmniFocus write contract" in data["error"]
        assert "project, due, defer, flagged, tags" in data["error"]
        mock.add_task.assert_not_awaited()
        assert _log_lines(contract_on) == []

    @pytest.mark.asyncio
    async def test_user_asked_lifts_checks(self, contract_on: Path) -> None:
        data, mock = await _add(
            {
                "name": "T",
                "source": "Claude · chat",
                "due": "tomorrow",
                "flagged": True,
                "user_asked": "flag it for tomorrow",
            }
        )
        assert data["status"] == "created"
        [entry] = _log_lines(contract_on)
        assert entry["user_asked"] == "flag it for tomorrow"
        assert entry["fields"] == ["due", "flagged", "name"]

    @pytest.mark.asyncio
    async def test_rental_exception(self, contract_on: Path) -> None:
        data, mock = await _add(
            {"name": "Call agency", "source": "rental sweep", "project": "Engin", "due": "today"}
        )
        assert data["status"] == "created"
        assert mock.add_task.await_args.kwargs["parent_task_id"] == "p1"

    @pytest.mark.asyncio
    async def test_failed_write_not_logged(self, contract_on: Path) -> None:
        data, _ = await _add({"name": "", "source": "Duet · daily"})
        assert "error" in data
        assert _log_lines(contract_on) == []


class TestUpdateTaskContract:
    @pytest.mark.asyncio
    async def test_rejects_dates_on_agent_inbox_task(self, contract_on: Path) -> None:
        data, mock = await _update(
            {"task_id": "t2", "due": "today", "flagged": True}, _agent_model()
        )
        assert "agent task still in the Inbox" in data["error"]
        assert "due, flagged" in data["error"]
        mock.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_tags_and_project(self, contract_on: Path) -> None:
        data, _ = await _update(
            {"task_id": "t2", "tag_ids": ["tag1"], "project_id": "p2"}, _agent_model()
        )
        assert "project, tags" in data["error"]

    @pytest.mark.asyncio
    async def test_unknown_project_id_rejected(self, contract_on: Path) -> None:
        data, _ = await _update({"task_id": "t2", "project_id": "nope"}, _agent_model())
        assert "project" in data["error"]

    @pytest.mark.asyncio
    async def test_note_edit_keeps_source_line(self, contract_on: Path) -> None:
        data, mock = await _update({"task_id": "t2", "note": "Called, no answer"}, _agent_model())
        assert data["status"] == "updated"
        updated = mock.update_task.await_args.args[0]
        assert updated.note == "via Duet · daily · 2026-11-19\nCalled, no answer"
        [entry] = _log_lines(contract_on)
        assert entry["source"] == "via Duet · daily · 2026-11-19"
        assert entry["fields"] == ["note"]
        assert entry["task_id"] == "t1"

    @pytest.mark.asyncio
    async def test_empty_note_keeps_source_line(self, contract_on: Path) -> None:
        _, mock = await _update({"task_id": "t2", "note": ""}, _agent_model())
        assert mock.update_task.await_args.args[0].note == "via Duet · daily · 2026-11-19"

    @pytest.mark.asyncio
    async def test_note_with_source_line_kept_as_is(self, contract_on: Path) -> None:
        _, mock = await _update(
            {"task_id": "t2", "note": "via Duet · x · 2026-11-20"}, _agent_model()
        )
        assert mock.update_task.await_args.args[0].note == "via Duet · x · 2026-11-20"

    @pytest.mark.asyncio
    async def test_drop_allowed(self, contract_on: Path) -> None:
        data, _ = await _update({"task_id": "t2", "dropped": True}, _agent_model())
        assert data["status"] == "dropped"

    @pytest.mark.asyncio
    async def test_rental_exception_uses_task_source(self, contract_on: Path) -> None:
        model = _agent_model(note="via Claude Code · rental sweep · 2026-11-19")
        data, _ = await _update({"task_id": "t2", "project_id": "p1", "due": "today"}, model)
        assert data["status"] == "updated"

    @pytest.mark.asyncio
    async def test_user_asked_lifts_checks(self, contract_on: Path) -> None:
        data, _ = await _update(
            {"task_id": "t2", "due": "tomorrow", "user_asked": "due tomorrow please"},
            _agent_model(),
        )
        assert data["status"] == "updated"
        assert _log_lines(contract_on)[0]["user_asked"] == "due tomorrow please"

    @pytest.mark.asyncio
    async def test_filed_task_is_free(self, contract_on: Path) -> None:
        data, _ = await _update({"task_id": "t2", "due": "tomorrow"}, _agent_model(inbox=False))
        assert data["status"] == "updated"

    @pytest.mark.asyncio
    async def test_own_task_is_free(self, contract_on: Path) -> None:
        data, _ = await _update({"task_id": "t2", "flagged": True}, _agent_model(note="mine"))
        assert data["status"] == "updated"
        assert _log_lines(contract_on)[0]["source"] is None

    @pytest.mark.asyncio
    async def test_missing_task_falls_through(self, contract_on: Path) -> None:
        data, _ = await _update({"task_id": "missing", "due": "today"}, _agent_model())
        assert "error" in data
        assert _log_lines(contract_on) == []

    @pytest.mark.asyncio
    async def test_off_skips_everything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(CONTRACT_ENV, raising=False)
        data, _ = await _update({"task_id": "t2", "due": "today"}, _agent_model())
        assert data["status"] == "updated"
