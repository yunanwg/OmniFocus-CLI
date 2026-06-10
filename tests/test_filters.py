"""Tests for :mod:`omnifocus.filters` — the shared task-filtering layer."""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

from datetime import UTC, date, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo

import pytest

from omnifocus.filters import (
    VALID_TASK_STATUS,
    filter_tasks,
    folder_subtree_project_ids,
    parse_filter_date,
    user_tz,
)
from omnifocus.models import Folder, OFModel, Project, Tag, Task

NOW = datetime(2026, 4, 6, 12, 0, 0, tzinfo=UTC)
PARIS = ZoneInfo("Europe/Paris")
# 23:30 UTC on 2026-06-10 is 01:30 on 2026-06-11 in Paris (UTC+2 in summer):
# the canonical case that proves completion dates must be localised.
BOUNDARY_UTC = datetime(2026, 6, 10, 23, 30, 0, tzinfo=UTC)
BOUNDARY_NAIVE = datetime(2026, 6, 10, 23, 30, 0)  # noqa: DTZ001 — intentional naive value
OLD_COMPLETED = datetime(2024, 1, 1, 8, 0, 0, tzinfo=UTC)
PAST_DUE = datetime(2020, 1, 1, 9, 0, 0)  # noqa: DTZ001 — due is naive local wall-clock
FUTURE_DUE = datetime(2090, 1, 1, 9, 0, 0)  # noqa: DTZ001 — due is naive local wall-clock


def _task(
    tid: str,
    *,
    project_id: str | None = None,
    inbox: bool = False,
    completed: datetime | None = None,
    flagged: bool = False,
    due: datetime | None = None,
    hidden: datetime | None = None,
    tag_ids: tuple[str, ...] = (),
) -> Task:
    return Task(
        id=tid,
        name=tid,
        parent_task_id=project_id,
        project_id=project_id,
        inbox=inbox,
        completed=completed,
        flagged=flagged,
        due=due,
        start=None,
        hidden=hidden,
        note="",
        rank=0,
        repetition_rule=None,
        estimated_minutes=None,
        tag_ids=tag_ids,
        added=NOW,
        modified=NOW,
    )


def _folder(fid: str, name: str, parent: str | None) -> Folder:
    return Folder(id=fid, name=name, parent_folder_id=parent, rank=0, added=NOW, modified=NOW)


def _project(pid: str, name: str, folder_id: str | None) -> Project:
    return Project(
        id=pid,
        name=name,
        folder_id=folder_id,
        status="active",
        singleton=False,
        rank=0,
        added=NOW,
        modified=NOW,
        flagged=False,
        due=None,
        start=None,
        note="",
        completed=None,
    )


def _model() -> OFModel:
    model = OFModel()
    model.folders["fw"] = _folder("fw", "Work", None)
    # Child folder name deliberately does NOT contain "Work", so folder="Work"
    # only matches the parent and must reach this via subtree expansion.
    model.folders["fws"] = _folder("fws", "Engineering", "fw")
    model.folders["fp"] = _folder("fp", "Personal", None)
    model.projects["pw"] = _project("pw", "Work Project", "fw")
    model.projects["pws"] = _project("pws", "Work Sub Project", "fws")
    model.projects["pp"] = _project("pp", "Personal Project", "fp")
    model.projects["pl"] = _project("pl", "Loose Project", None)
    model.tags["t_home"] = Tag(id="t_home", name="@home", parent_tag_id=None, rank=0)
    # active
    model.tasks["a_work"] = _task(
        "a_work", project_id="pw", flagged=True, due=PAST_DUE, tag_ids=("t_home",)
    )
    model.tasks["a_inbox"] = _task("a_inbox", project_id="pp", inbox=True, due=FUTURE_DUE)
    model.tasks["a_sub"] = _task("a_sub", project_id="pws")
    # completed
    model.tasks["c_boundary"] = _task("c_boundary", project_id="pw", completed=BOUNDARY_UTC)
    model.tasks["c_naive"] = _task("c_naive", project_id="pw", completed=BOUNDARY_NAIVE)
    model.tasks["c_old"] = _task("c_old", project_id="pp", completed=OLD_COMPLETED)
    # dropped
    model.tasks["d_drop"] = _task("d_drop", project_id="pw", hidden=NOW)
    return model


def _ids(tasks: list[Task]) -> set[str]:
    return {task.id for task in tasks}


# --------------------------------------------------------------------------- status


def test_status_active_is_default() -> None:
    assert _ids(filter_tasks(_model())) == {"a_work", "a_inbox", "a_sub"}


def test_status_completed() -> None:
    assert _ids(filter_tasks(_model(), status="completed")) == {"c_boundary", "c_naive", "c_old"}


def test_status_dropped() -> None:
    assert _ids(filter_tasks(_model(), status="dropped")) == {"d_drop"}


def test_status_all() -> None:
    assert _ids(filter_tasks(_model(), status="all")) == {
        "a_work",
        "a_inbox",
        "a_sub",
        "c_boundary",
        "c_naive",
        "c_old",
        "d_drop",
    }


def test_status_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Invalid status"):
        filter_tasks(_model(), status="bogus")


def test_valid_task_status_constant() -> None:
    assert VALID_TASK_STATUS == ("active", "completed", "dropped", "all")


# --------------------------------------------------------------------------- completed dates


def test_completed_on_promotes_active_to_completed() -> None:
    # Default status is active, but a completed_* filter must reach completed tasks.
    result = filter_tasks(_model(), completed_on="2026-06-11", tz=PARIS)
    assert _ids(result) == {"c_boundary", "c_naive"}


def test_completed_on_localises_to_paris() -> None:
    # In Paris the boundary task lands on the 11th, so the 10th excludes it.
    assert _ids(filter_tasks(_model(), status="all", completed_on="2026-06-10", tz=PARIS)) == set()


def test_completed_on_localises_to_utc() -> None:
    # In UTC the same instant is the 10th — proving the timezone actually matters.
    assert _ids(filter_tasks(_model(), status="all", completed_on="2026-06-10", tz=UTC)) == {
        "c_boundary",
        "c_naive",
    }


def test_completed_since_includes_2026_excludes_2024() -> None:
    result = filter_tasks(_model(), completed_since="2026-01-01", tz=PARIS)
    assert _ids(result) == {"c_boundary", "c_naive"}


def test_completed_since_over_all_skips_uncompleted() -> None:
    # status=all keeps uncompleted tasks in the base set; the since filter must drop them.
    result = filter_tasks(_model(), status="all", completed_since="2026-01-01", tz=PARIS)
    assert _ids(result) == {"c_boundary", "c_naive"}


def test_completed_on_bad_date_raises() -> None:
    with pytest.raises(ValueError, match="Invalid isoformat|does not match"):
        filter_tasks(_model(), completed_on="not-a-date", tz=PARIS)


# --------------------------------------------------------------------------- boolean filters


def test_inbox() -> None:
    assert _ids(filter_tasks(_model(), inbox=True)) == {"a_inbox"}


def test_today_uses_due_date() -> None:
    # Past-due is included, far-future and no-due are not.
    assert _ids(filter_tasks(_model(), today=True, tz=PARIS)) == {"a_work"}


def test_flagged() -> None:
    assert _ids(filter_tasks(_model(), flagged=True)) == {"a_work"}


def test_due_only() -> None:
    assert _ids(filter_tasks(_model(), due=True)) == {"a_work", "a_inbox"}


# --------------------------------------------------------------------------- entity filters


def test_project_substring() -> None:
    assert _ids(filter_tasks(_model(), project="personal")) == {"a_inbox"}


def test_tag_substring() -> None:
    assert _ids(filter_tasks(_model(), tag="home")) == {"a_work"}


def test_tag_id_exact() -> None:
    assert _ids(filter_tasks(_model(), tag_id="t_home")) == {"a_work"}


def test_folder_includes_subtree() -> None:
    # "Work" matches the Work folder and its Work Sub child, so both projects count.
    assert _ids(filter_tasks(_model(), folder="Work")) == {"a_work", "a_sub"}


def test_folder_no_match_returns_nothing() -> None:
    assert filter_tasks(_model(), folder="Nonexistent") == []


# --- folder_subtree_project_ids ---


def test_folder_subtree_project_ids_includes_descendants() -> None:
    assert folder_subtree_project_ids(_model(), "Work") == {"pw", "pws"}


def test_folder_subtree_project_ids_empty_on_no_match() -> None:
    assert folder_subtree_project_ids(_model(), "Nope") == set()


# --------------------------------------------------------------------------- user_tz


def test_user_tz_from_of_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OF_TIMEZONE", "Europe/Paris")
    assert user_tz() == ZoneInfo("Europe/Paris")


def test_user_tz_falls_back_to_tz_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OF_TIMEZONE", raising=False)
    monkeypatch.setenv("TZ", "America/New_York")
    assert user_tz() == ZoneInfo("America/New_York")


def test_user_tz_invalid_name_falls_back_to_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OF_TIMEZONE", "Definitely/Invalid")
    monkeypatch.delenv("TZ", raising=False)
    assert isinstance(user_tz(), tzinfo)


def test_user_tz_no_env_uses_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OF_TIMEZONE", raising=False)
    monkeypatch.delenv("TZ", raising=False)
    assert isinstance(user_tz(), tzinfo)


# --------------------------------------------------------------------------- parse_filter_date


def test_parse_filter_date_iso() -> None:
    assert parse_filter_date("2026-06-10", tz=PARIS) == date(2026, 6, 10)


def test_parse_filter_date_today() -> None:
    assert parse_filter_date("today", tz=PARIS) == datetime.now(PARIS).date()
    assert parse_filter_date("tod", tz=PARIS) == datetime.now(PARIS).date()


def test_parse_filter_date_yesterday() -> None:
    expected = datetime.now(PARIS).date() - timedelta(days=1)
    assert parse_filter_date("yesterday", tz=PARIS) == expected
    assert parse_filter_date("yd", tz=PARIS) == expected


def test_parse_filter_date_tomorrow() -> None:
    assert parse_filter_date("tomorrow", tz=PARIS) > datetime.now(PARIS).date()
    assert parse_filter_date("tom", tz=PARIS) > datetime.now(PARIS).date()


def test_parse_filter_date_default_tz_returns_date() -> None:
    assert isinstance(parse_filter_date("today"), date)


def test_parse_filter_date_bad_raises() -> None:
    with pytest.raises(ValueError, match="Invalid isoformat|does not match"):
        parse_filter_date("garbage", tz=PARIS)


# --- tz defaulting in filter_tasks ---


def test_filter_tasks_defaults_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OF_TIMEZONE", "Europe/Paris")
    # No tz argument: filter_tasks resolves user_tz() internally.
    assert _ids(filter_tasks(_model(), status="all", completed_on="2026-06-11")) == {
        "c_boundary",
        "c_naive",
    }
