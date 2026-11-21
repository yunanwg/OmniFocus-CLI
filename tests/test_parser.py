"""Tests for :mod:`omnifocus.parser`."""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import io
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC

import pytest

from omnifocus.errors import OFParseError
from omnifocus.models import OFModel
from omnifocus.parser import (
    _bool,
    _idref,
    _int,
    _materialize_reference_snapshot,
    _parse_dt_local,
    _parse_dt_utc,
    _text,
    build_model,
    load_xml_from_zip,
)
from omnifocus.recursion_limit import MIN_RECURSION_LIMIT
from tests.conftest import make_zip

NS = "{http://www.omnigroup.com/namespace/OmniFocus/v2}"

# ---------------------------------------------------------------------------
# load_xml_from_zip
# ---------------------------------------------------------------------------


class TestLoadXmlFromZip:
    def test_valid_zip(self, sample_zip: bytes) -> None:
        root = load_xml_from_zip(sample_zip)
        assert root.tag == f"{NS}omnifocus"

    def test_not_a_zip(self) -> None:
        with pytest.raises(OFParseError, match="Invalid ZIP"):
            load_xml_from_zip(b"this is not a zip")

    def test_zip_without_contents_xml(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("other.xml", "<root/>")
        with pytest.raises(OFParseError, match="contents.xml not found"):
            load_xml_from_zip(buf.getvalue())

    def test_malformed_xml(self) -> None:
        zipped = make_zip("<unclosed>")
        with pytest.raises(OFParseError, match="Malformed XML"):
            load_xml_from_zip(zipped)

    def test_invalid_utf8(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("contents.xml", b"\xff\xfe broken bytes")
        with pytest.raises(OFParseError):
            load_xml_from_zip(buf.getvalue())


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

_SAMPLE_EL = ET.fromstring(  # noqa: S314
    f'<task xmlns="{NS[1:-1]}" id="t1">'
    f"  <name>Hello</name>"
    f"  <rank>42</rank>"
    f"  <flagged>true</flagged>"
    f'  <task idref="parent1"/>'
    f"</task>"
)


class TestHelpers:
    def test_text_present(self) -> None:
        assert _text(_SAMPLE_EL, "name") == "Hello"

    def test_text_missing(self) -> None:
        assert _text(_SAMPLE_EL, "nonexistent") == ""

    def test_text_default(self) -> None:
        assert _text(_SAMPLE_EL, "nonexistent", "fallback") == "fallback"

    def test_bool_true(self) -> None:
        assert _bool(_SAMPLE_EL, "flagged") is True

    def test_bool_missing(self) -> None:
        assert _bool(_SAMPLE_EL, "nonexistent") is False

    def test_bool_default_true(self) -> None:
        assert _bool(_SAMPLE_EL, "nonexistent", default=True) is True

    def test_int_present(self) -> None:
        assert _int(_SAMPLE_EL, "rank") == 42

    def test_int_missing(self) -> None:
        assert _int(_SAMPLE_EL, "nonexistent") is None

    def test_int_invalid(self) -> None:
        el = ET.fromstring(f'<task xmlns="{NS[1:-1]}"><rank>notanumber</rank></task>')  # noqa: S314
        assert _int(el, "rank") is None

    def test_idref_present(self) -> None:
        assert _idref(_SAMPLE_EL, "task") == "parent1"

    def test_idref_missing_child(self) -> None:
        assert _idref(_SAMPLE_EL, "folder") is None

    def test_idref_no_idref_attr(self) -> None:
        el = ET.fromstring(f'<task xmlns="{NS[1:-1]}"><context/></task>')  # noqa: S314
        assert _idref(el, "context") is None


class TestParseDt:
    def test_utc_with_z(self) -> None:
        dt = _parse_dt_utc("2026-03-22T15:40:11.347Z")
        assert dt is not None
        assert dt.tzinfo == UTC
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.day == 22

    def test_utc_empty(self) -> None:
        assert _parse_dt_utc("") is None

    def test_utc_invalid(self) -> None:
        assert _parse_dt_utc("not-a-date") is None

    def test_local_present(self) -> None:
        dt = _parse_dt_local("2026-06-01T19:00:00.000")
        assert dt is not None
        assert dt.tzinfo is None
        assert dt.hour == 19

    def test_local_empty(self) -> None:
        assert _parse_dt_local("") is None

    def test_local_invalid(self) -> None:
        assert _parse_dt_local("bad") is None


# ---------------------------------------------------------------------------
# build_model — sample fixture
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_folders_loaded(self, sample_model: OFModel) -> None:
        assert "folder1" in sample_model.folders
        assert "folder2" in sample_model.folders
        assert sample_model.folders["folder1"].name == "Work"
        assert sample_model.folders["folder2"].parent_folder_id == "folder1"

    def test_tags_loaded(self, sample_model: OFModel) -> None:
        assert "tag1" in sample_model.tags
        assert "tag2" in sample_model.tags
        assert sample_model.tags["tag1"].name == "@home"
        assert sample_model.tags["tag2"].parent_tag_id == "tag1"
        assert sample_model.tags["tag1"].added is not None
        assert sample_model.tags["tag1"].modified is not None

    def test_projects_loaded(self, sample_model: OFModel) -> None:
        assert "proj1" in sample_model.projects
        assert "proj2" in sample_model.projects
        p = sample_model.projects["proj1"]
        assert p.name == "🚀 Launch MVP"
        assert p.status == "active"
        assert p.flagged is True
        assert p.folder_id == "folder1"
        assert p.due is not None

    def test_project_singleton(self, sample_model: OFModel) -> None:
        p = sample_model.projects["proj2"]
        assert p.singleton is True
        assert p.folder_id is None
        assert p.status == "inactive"

    def test_project_review_fields_loaded(self, sample_model: OFModel) -> None:
        project = sample_model.projects["proj1"]
        assert project.last_review is not None
        assert project.next_review is not None
        assert project.review_interval == "@1m"

    def test_project_completed(self, sample_model: OFModel) -> None:
        p = sample_model.projects["proj3"]
        assert p.status == "done"
        assert p.completed is not None

    def test_tasks_not_in_projects(self, sample_model: OFModel) -> None:
        # Project ids must not appear in tasks dict
        assert "proj1" not in sample_model.tasks
        assert "proj2" not in sample_model.tasks

    def test_task_loaded(self, sample_model: OFModel) -> None:
        t = sample_model.tasks["task1"]
        assert t.name == "Write tests"
        assert t.project_id == "proj1"
        assert t.parent_task_id == "proj1"
        assert t.flagged is False
        assert t.estimated_minutes == 120
        assert t.repetition_rule == "FREQ=WEEKLY;INTERVAL=1"
        assert "tag1" in t.tag_ids

    def test_task_completed(self, sample_model: OFModel) -> None:
        t = sample_model.tasks["task2"]
        assert t.completed is not None

    def test_nested_task_project_id(self, sample_model: OFModel) -> None:
        # task3 is a subtask of task1, which is under proj1
        t = sample_model.tasks["task3"]
        assert t.project_id == "proj1"
        assert t.parent_task_id == "task1"

    def test_hidden_task(self, sample_model: OFModel) -> None:
        t = sample_model.tasks["task4"]
        assert t.hidden is not None

    def test_inbox_task(self, sample_model: OFModel) -> None:
        t = sample_model.tasks["inbox1"]
        assert t.inbox is True
        assert t.project_id is None
        assert t.name == "Buy milk 🥛"

    def test_active_tasks(self, sample_model: OFModel) -> None:
        active = sample_model.active_tasks
        ids = {t.id for t in active}
        # task1, task3, inbox1 are active; task2 (completed) and task4 (hidden) are not
        assert "task1" in ids
        assert "task3" in ids
        assert "inbox1" in ids
        assert "task2" not in ids
        assert "task4" not in ids

    def test_no_transactions(self, sample_zip: bytes) -> None:
        model = build_model(sample_zip, transaction_bytes_list=None)
        assert len(model.tasks) > 0

    def test_empty_transaction_list(self, sample_zip: bytes) -> None:
        model = build_model(sample_zip, transaction_bytes_list=[])
        assert len(model.tasks) > 0

    def test_raises_recursion_limit_when_too_low(self, sample_zip: bytes) -> None:
        original = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            model = build_model(sample_zip)
            assert len(model.tasks) > 0
            assert sys.getrecursionlimit() >= MIN_RECURSION_LIMIT
        finally:
            sys.setrecursionlimit(original)

    def test_does_not_lower_higher_recursion_limit(self, sample_zip: bytes) -> None:
        original = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(MIN_RECURSION_LIMIT + 10_000)
            build_model(sample_zip)
            assert sys.getrecursionlimit() == MIN_RECURSION_LIMIT + 10_000
        finally:
            sys.setrecursionlimit(original)


# ---------------------------------------------------------------------------
# Transaction merging
# ---------------------------------------------------------------------------

_BASE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="t1">
    <project/>
    <inbox>true</inbox>
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Original name</name>
    <rank>1</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-01T00:00:00.000Z</modified>
  </task>
  <task id="t2">
    <project/>
    <inbox>true</inbox>
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Will be deleted</name>
    <rank>2</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-01T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""

_TX_UPDATE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="t1">
    <project/>
    <inbox>true</inbox>
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Updated name</name>
    <rank>1</rank>
    <flagged>true</flagged>
    <completed/>
    <modified>2026-01-02T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""

_TX_DELETE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="t2">
    <added>2026-01-02T00:00:00.000Z</added>
  </task>
</omnifocus>
"""

_TX_ADD_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="t3">
    <project/>
    <inbox>true</inbox>
    <added>2026-01-03T00:00:00.000Z</added>
    <name>Brand new task</name>
    <rank>3</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-03T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""

_TX_SECOND_UPDATE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="t1">
    <project/>
    <inbox>true</inbox>
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Updated name again</name>
    <rank>5</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-03T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""

_TX_READD_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="t2">
    <project/>
    <inbox>true</inbox>
    <added>2026-01-04T00:00:00.000Z</added>
    <name>Restored task</name>
    <rank>2</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-04T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""

_TASK_TO_PROJECT_BASE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="shape1">
    <project/>
    <inbox>true</inbox>
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Shape shifter</name>
    <rank>1</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-01T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""

_TASK_TO_PROJECT_TX_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="shape1">
    <name>Shape shifter</name>
    <rank>1</rank>
    <flagged>false</flagged>
    <added>2026-01-01T00:00:00.000Z</added>
    <modified>2026-01-02T00:00:00.000Z</modified>
    <project>
      <status>active</status>
      <singleton>false</singleton>
    </project>
  </task>
</omnifocus>
"""

_FOLDER_UPDATE_BASE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <folder id="eYAcOM9PHFo">
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Work</name>
    <rank>1</rank>
    <modified>2026-01-01T00:00:00.000Z</modified>
  </folder>
  <folder id="ptPMm-uIiPn">
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Routines</name>
    <rank>2</rank>
    <modified>2026-01-01T00:00:00.000Z</modified>
  </folder>
  <task id="IPZIKgwH2FI">
    <project>
      <folder idref="eYAcOM9PHFo"/>
      <status>active</status>
      <singleton>false</singleton>
    </project>
    <inbox>false</inbox>
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Certyfikaty 2026</name>
    <rank>10</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-01T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""

_FOLDER_UPDATE_TX_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <folder id="eYAcOM9PHFo" op="update">
    <modified>2026-01-02T00:00:00.000Z</modified>
    <rank>5</rank>
  </folder>
  <folder id="ptPMm-uIiPn" op="update">
    <modified>2026-01-02T00:00:00.000Z</modified>
    <rank>6</rank>
  </folder>
  <task id="IPZIKgwH2FI" op="update">
    <modified>2026-01-02T00:00:00.000Z</modified>
    <project>
      <status>active</status>
      <singleton>false</singleton>
    </project>
  </task>
</omnifocus>
"""

_PROJECT_DIRECT_FOLDER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="direct-folder-project">
    <project>
      <status>active</status>
      <singleton>false</singleton>
    </project>
    <folder idref="folder-direct"/>
    <inbox>false</inbox>
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Direct folder project</name>
    <rank>1</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-01T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""

_REFERENCE_FOLDER_BASE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <folder id="eYAcOM9PHFo">
    <folder/>
    <added>2024-12-15T19:11:43.652Z</added>
    <name>💻 Work</name>
    <note/>
    <rank>240802395</rank>
    <hidden/>
    <modified>2026-01-23T22:16:42.744Z</modified>
  </folder>
  <folder id="ptPMm-uIiPn">
    <folder/>
    <added>2026-01-07T02:47:39.465Z</added>
    <name>🔄 Routines</name>
    <note/>
    <rank>2147136693</rank>
    <hidden/>
    <modified>2026-01-23T22:25:10.259Z</modified>
  </folder>
</omnifocus>
"""

_REFERENCE_FOLDER_TX_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <folder id="eYAcOM9PHFo" op="reference">
    <reference-snapshot>
      <folder/>
      <added>2024-12-15T19:11:43.652Z</added>
      <modified>2026-03-24T13:46:44.118Z</modified>
      <name>💻 Work</name>
      <note/>
      <rank>240802395</rank>
      <hidden/>
    </reference-snapshot>
  </folder>
  <folder id="ptPMm-uIiPn" op="reference">
    <reference-snapshot>
      <folder/>
      <added>2026-01-07T02:47:39.465Z</added>
      <modified>2026-03-24T08:02:16.197Z</modified>
      <name>🔄 Routines</name>
      <note/>
      <rank>2147136693</rank>
      <hidden/>
    </reference-snapshot>
  </folder>
  <task id="jWkUhVYC7Qx" op="update">
    <project>
      <folder idref="ptPMm-uIiPn"/>
      <singleton>true</singleton>
      <status>active</status>
    </project>
    <inbox>false</inbox>
    <task/>
    <added>2026-01-07T02:59:32.726Z</added>
    <modified>2026-03-24T08:02:16.197Z</modified>
    <name>💻 Work</name>
    <rank>1610612736</rank>
  </task>
</omnifocus>
"""

_PROJECT_TO_TASK_BASE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <folder id="jBfVJLuNwe1">
    <folder idref="eYAcOM9PHFo"/>
    <added>2024-12-15T19:22:37.315Z</added>
    <name>🔴 SECURITUM</name>
    <rank>1073741824</rank>
    <modified>2026-03-24T13:46:13.924Z</modified>
  </folder>
  <task id="pI8a3Bp_6Wy">
    <project>
      <folder idref="eYAcOM9PHFo"/>
      <status>active</status>
      <singleton>true</singleton>
    </project>
    <inbox>false</inbox>
    <task/>
    <added>2025-05-24T09:34:43.638Z</added>
    <name>🔴 SECURITUM (CISO)</name>
    <rank>1</rank>
    <modified>2026-03-24T13:41:53.335Z</modified>
  </task>
  <task id="JkUWGlVomMw">
    <project>
      <folder idref="jBfVJLuNwe1"/>
      <status>active</status>
      <singleton>false</singleton>
    </project>
    <inbox>false</inbox>
    <task/>
    <added>2026-03-24T13:41:53.335Z</added>
    <name>Migracja starej infry na nową</name>
    <rank>538220087</rank>
    <order>parallel</order>
    <modified>2026-03-24T13:41:53.335Z</modified>
  </task>
</omnifocus>
"""

_PROJECT_TO_TASK_TX_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="JkUWGlVomMw" op="update">
    <project/>
    <task idref="pI8a3Bp_6Wy"/>
    <added>2026-03-24T13:41:53.335Z</added>
    <modified>2026-03-24T13:46:29.682Z</modified>
    <rank>2147483646</rank>
  </task>
</omnifocus>
"""

_REPEATED_NAME_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="weekly-review-1">
    <project/>
    <inbox>true</inbox>
    <added>2026-03-01T00:00:00.000Z</added>
    <name>Weekly Review</name>
    <rank>1</rank>
    <flagged>false</flagged>
    <due>2026-03-22T19:00:00.000</due>
    <completed/>
    <modified>2026-03-01T00:00:00.000Z</modified>
  </task>
  <task id="weekly-review-2">
    <project/>
    <inbox>true</inbox>
    <added>2026-03-02T00:00:00.000Z</added>
    <name>Weekly Review</name>
    <rank>2</rank>
    <flagged>false</flagged>
    <due>2026-04-05T19:00:00.000</due>
    <completed/>
    <modified>2026-03-02T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""


class TestTransactionMerge:
    def test_update_overwrites_task(self) -> None:
        base = make_zip(_BASE_XML)
        tx = make_zip(_TX_UPDATE_XML)
        model = build_model(base, [tx])
        assert model.tasks["t1"].name == "Updated name"
        assert model.tasks["t1"].flagged is True

    def test_delete_removes_task(self) -> None:
        base = make_zip(_BASE_XML)
        tx = make_zip(_TX_DELETE_XML)
        model = build_model(base, [tx])
        assert "t2" not in model.tasks

    def test_add_new_task(self) -> None:
        base = make_zip(_BASE_XML)
        tx = make_zip(_TX_ADD_XML)
        model = build_model(base, [tx])
        assert "t3" in model.tasks
        assert model.tasks["t3"].name == "Brand new task"

    def test_multiple_transactions_applied_in_order(self) -> None:
        base = make_zip(_BASE_XML)
        txs = [make_zip(_TX_UPDATE_XML), make_zip(_TX_DELETE_XML), make_zip(_TX_ADD_XML)]
        model = build_model(base, txs)
        assert model.tasks["t1"].name == "Updated name"
        assert "t2" not in model.tasks
        assert "t3" in model.tasks

    def test_multiple_upserts_keep_latest_by_id(self) -> None:
        base = make_zip(_BASE_XML)
        txs = [make_zip(_TX_UPDATE_XML), make_zip(_TX_SECOND_UPDATE_XML)]
        model = build_model(base, txs)
        assert len(model.tasks) == 2
        assert model.tasks["t1"].name == "Updated name again"
        assert model.tasks["t1"].rank == 5

    def test_delete_followed_by_readd_restores_single_task(self) -> None:
        base = make_zip(_BASE_XML)
        txs = [make_zip(_TX_DELETE_XML), make_zip(_TX_READD_XML)]
        model = build_model(base, txs)
        assert model.tasks["t2"].name == "Restored task"
        assert list(model.tasks).count("t2") == 1

    def test_task_can_change_shape_into_project(self) -> None:
        base = make_zip(_TASK_TO_PROJECT_BASE_XML)
        tx = make_zip(_TASK_TO_PROJECT_TX_XML)
        model = build_model(base, [tx])
        assert "shape1" not in model.tasks
        assert model.projects["shape1"].name == "Shape shifter"
        assert model.projects["shape1"].status == "active"

    def test_repeated_names_remain_distinct_by_id(self) -> None:
        model = build_model(make_zip(_REPEATED_NAME_XML))
        weekly_reviews = [task for task in model.tasks.values() if task.name == "Weekly Review"]
        assert len(weekly_reviews) == 2
        assert {task.id for task in weekly_reviews} == {
            "weekly-review-1",
            "weekly-review-2",
        }


class TestWriterParserRoundTrip:
    """End-to-end: a transaction produced by the writer parses back to the same intent."""

    def test_drop_round_trips_to_hidden_field(self) -> None:
        from omnifocus.writer import TaskWriter

        base = make_zip(_BASE_XML)
        task = build_model(base).tasks["t1"]
        _, drop_delta = TaskWriter(head_id="tailA", parent_tail_id="tail0").drop_task(task)
        model = build_model(base, [drop_delta])
        # OmniFocus 4 serializes a dropped task as a <hidden> timestamp.
        assert model.tasks["t1"].hidden is not None

    def test_recurrence_triad_round_trips(self) -> None:
        import dataclasses

        from omnifocus.writer import TaskWriter

        base = make_zip(_BASE_XML)
        task = dataclasses.replace(
            build_model(base).tasks["t1"],
            repetition_rule="FREQ=MONTHLY;INTERVAL=1",
            repetition_method="due-after-completion",
            repetition_schedule_type="from-completion",
            repetition_anchor_date="dateDue",
        )
        _, delta = TaskWriter(head_id="tailA", parent_tail_id="tail0").update_task(task)
        parsed = build_model(base, [delta]).tasks["t1"]
        assert parsed.repetition_schedule_type == "from-completion"
        assert parsed.repetition_anchor_date == "dateDue"
        assert parsed.repetition_method == "due-after-completion"


class TestTransactionMergeExtra:
    def test_update_without_name_merges_into_existing_task(self) -> None:
        base = make_zip(_BASE_XML)
        tx = make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="t1" op="update">
    <added>2026-01-01T00:00:00.000Z</added>
    <modified>2026-01-03T00:00:00.000Z</modified>
    <due>2026-01-05T19:00:00.000</due>
  </task>
</omnifocus>
""")
        model = build_model(base, [tx])
        assert model.tasks["t1"].name == "Original name"
        assert model.tasks["t1"].due is not None
        assert model.tasks["t1"].due.isoformat() == "2026-01-05T19:00:00"

    def test_app_style_task_creation_with_extra_update_remains_visible(self) -> None:
        base = make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2"/>
""")
        txs = [
            make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="new1">
    <project/>
    <inbox>true</inbox>
    <task/>
    <added>2026-01-01T00:00:00.000Z</added>
    <name/>
    <note/>
    <rank>1</rank>
    <hidden/>
    <context/>
    <start/>
    <planned/>
    <due/>
    <completed/>
    <estimated-minutes/>
    <order>sequential</order>
    <flagged>false</flagged>
    <completed-by-children>false</completed-by-children>
    <repetition-rule/>
    <repetition-method/>
    <repetition-schedule-type/>
    <repetition-anchor-date/>
    <catch-up-automatically>false</catch-up-automatically>
    <next-clone-identifier>0</next-clone-identifier>
    <due-date-alarm-policy/>
    <defer-date-alarm-policy/>
    <latest-time-to-start-alarm-policy/>
    <planned-date-alarm-policy/>
  </task>
</omnifocus>
"""),
            make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="new1" op="update">
    <added>2026-01-01T00:00:00.000Z</added>
    <modified>2026-01-02T00:00:00.000Z</modified>
    <name>Created via updates</name>
  </task>
</omnifocus>
"""),
            make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="new1" op="update">
    <added>2026-01-01T00:00:00.000Z</added>
    <modified>2026-01-03T00:00:00.000Z</modified>
    <due>2026-01-10T19:00:00.000</due>
  </task>
</omnifocus>
"""),
        ]
        model = build_model(base, txs)
        assert model.tasks["new1"].name == "Created via updates"
        assert model.tasks["new1"].due is not None
        assert model.tasks["new1"].due.isoformat() == "2026-01-10T19:00:00"

    def test_reference_snapshot_preserves_folder_entities(self) -> None:
        model = build_model(
            make_zip(_REFERENCE_FOLDER_BASE_XML),
            [make_zip(_REFERENCE_FOLDER_TX_XML)],
        )
        assert model.folders["eYAcOM9PHFo"].name == "💻 Work"
        assert model.folders["ptPMm-uIiPn"].name == "🔄 Routines"
        assert model.projects["jWkUhVYC7Qx"].folder_id == "ptPMm-uIiPn"

    def test_empty_project_update_turns_project_into_task(self) -> None:
        model = build_model(
            make_zip(_PROJECT_TO_TASK_BASE_XML),
            [make_zip(_PROJECT_TO_TASK_TX_XML)],
        )
        assert "JkUWGlVomMw" not in model.projects
        assert model.tasks["JkUWGlVomMw"].parent_task_id == "pI8a3Bp_6Wy"
        assert model.tasks["JkUWGlVomMw"].project_id == "pI8a3Bp_6Wy"
        assert model.tasks["JkUWGlVomMw"].name == "Migracja starej infry na nową"

    def test_reference_snapshot_can_seed_new_folder_without_baseline_entry(self) -> None:
        model = build_model(
            make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2"/>
"""),
            [
                make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <folder id="gi7RylwrvLG" op="reference">
    <reference-snapshot>
      <folder/>
      <added>2026-03-24T22:03:58.719Z</added>
      <modified>2026-03-24T22:04:23.544Z</modified>
      <name>dupa</name>
      <note/>
      <rank>2147396910</rank>
      <hidden/>
    </reference-snapshot>
  </folder>
</omnifocus>
"""),
            ],
        )
        assert model.folders["gi7RylwrvLG"].name == "dupa"

    def test_reference_without_snapshot_is_returned_unchanged(self) -> None:
        elem = ET.fromstring(  # noqa: S314
            f'<folder xmlns="{NS[1:-1]}" id="gi7RylwrvLG" op="reference" />'
        )
        assert _materialize_reference_snapshot(elem) is elem

    def test_update_without_existing_entry_is_indexed(self) -> None:
        base = make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2"/>
""")
        tx = make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="orphan" op="update">
    <added>2026-01-01T00:00:00.000Z</added>
    <modified>2026-01-01T00:00:00.000Z</modified>
    <name>Orphan update</name>
  </task>
</omnifocus>
""")
        model = build_model(base, [tx])
        assert model.tasks["orphan"].name == "Orphan update"

    def test_delete_op_removes_task(self) -> None:
        base = make_zip(_BASE_XML)
        tx = make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="t1" op="delete">
    <delete-snapshot/>
  </task>
</omnifocus>
""")
        model = build_model(base, [tx])
        assert "t1" not in model.tasks

    def test_empty_update_keeps_existing_task_intact(self) -> None:
        base = make_zip(_BASE_XML)
        tx = make_zip("""\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="t1" op="update"/>
</omnifocus>
""")
        model = build_model(base, [tx])
        assert model.tasks["t1"].name == "Original name"

    def test_folder_partial_update_without_name_keeps_folder(self) -> None:
        model = build_model(make_zip(_FOLDER_UPDATE_BASE_XML), [make_zip(_FOLDER_UPDATE_TX_XML)])
        assert model.folders["eYAcOM9PHFo"].name == "Work"
        assert model.folders["eYAcOM9PHFo"].rank == 5
        assert model.folders["ptPMm-uIiPn"].name == "Routines"
        assert model.folders["ptPMm-uIiPn"].rank == 6

    def test_project_partial_update_keeps_existing_folder_assignment(self) -> None:
        model = build_model(make_zip(_FOLDER_UPDATE_BASE_XML), [make_zip(_FOLDER_UPDATE_TX_XML)])
        assert model.projects["IPZIKgwH2FI"].folder_id == "eYAcOM9PHFo"

    def test_project_partial_update_keeps_existing_review_metadata(self) -> None:
        baseline = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="review-project">
    <project>
      <status>active</status>
      <singleton>false</singleton>
      <last-review>2026-03-01T09:00:00.000Z</last-review>
      <next-review>2026-04-01T09:00:00.000Z</next-review>
      <review-interval>@1m</review-interval>
    </project>
    <inbox>false</inbox>
    <added>2026-03-01T09:00:00.000Z</added>
    <name>Review me</name>
    <note/>
    <rank>1</rank>
    <modified>2026-03-01T09:00:00.000Z</modified>
  </task>
</omnifocus>
"""
        update = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="review-project" op="update">
    <project>
      <status>inactive</status>
    </project>
    <added>2026-03-01T09:00:00.000Z</added>
    <modified>2026-03-22T09:00:00.000Z</modified>
  </task>
</omnifocus>
"""
        model = build_model(make_zip(baseline), [make_zip(update)])
        project = model.projects["review-project"]
        assert project.status == "inactive"
        assert project.last_review is not None
        assert project.next_review is not None
        assert project.review_interval == "@1m"

    def test_project_can_fall_back_to_direct_folder_reference(self) -> None:
        model = build_model(make_zip(_PROJECT_DIRECT_FOLDER_XML))
        assert model.projects["direct-folder-project"].folder_id == "folder-direct"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_element_without_id_ignored(self) -> None:
        """Elements without an id attribute must be silently skipped."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task>
    <name>No ID task</name>
  </task>
  <folder>
    <name>No ID folder</name>
  </folder>
</omnifocus>
"""
        model = build_model(make_zip(xml))
        assert len(model.tasks) == 0
        assert len(model.folders) == 0

    def test_task_missing_name_ignored(self) -> None:
        """Tasks with no <name> child in the baseline are treated as deletions."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="ghost">
    <added>2026-01-01T00:00:00.000Z</added>
  </task>
</omnifocus>
"""
        model = build_model(make_zip(xml))
        assert "ghost" not in model.tasks

    def test_unicode_and_emoji_in_names(self) -> None:
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="u1">
    <project/>
    <inbox>true</inbox>
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Zrób zakupy 🛒 &amp; odpoczni 😴</name>
    <rank>1</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-01T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""
        model = build_model(make_zip(xml))
        assert model.tasks["u1"].name == "Zrób zakupy 🛒 & odpoczni 😴"

    def test_cycle_in_parent_chain_handled(self) -> None:
        """A circular parent reference must not cause infinite recursion."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <task id="a">
    <project/>
    <task idref="b"/>
    <inbox>false</inbox>
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Task A</name>
    <rank>1</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-01T00:00:00.000Z</modified>
  </task>
  <task id="b">
    <project/>
    <task idref="a"/>
    <inbox>false</inbox>
    <added>2026-01-01T00:00:00.000Z</added>
    <name>Task B</name>
    <rank>2</rank>
    <flagged>false</flagged>
    <completed/>
    <modified>2026-01-01T00:00:00.000Z</modified>
  </task>
</omnifocus>
"""
        # Should not raise, project_id will be None for both
        model = build_model(make_zip(xml))
        assert model.tasks["a"].project_id is None
        assert model.tasks["b"].project_id is None


# ---------------------------------------------------------------------------
# Perspectives
# ---------------------------------------------------------------------------

_PERSPECTIVE_XML = f"""<?xml version="1.0" encoding="utf-8"?>
<omnifocus xmlns="{NS[1:-1]}" app-id="com.omnigroup.OmniFocus4">
  <context id="tagP1"><name>P1</name></context>
  <perspective id="persp1">
    <added>2026-01-02T03:04:05.000Z</added>
    <modified>2026-02-03T04:05:06.000Z</modified>
    <plist version="1.0">
      <dict>
        <key>name</key><string>Today</string>
        <key>version</key><integer>3</integer>
        <key>iconScale</key><real>1.5</real>
        <key>filterRulesAggregation</key><string>all</string>
        <key>filterRules</key>
        <string>[{{"actionAvailability":"available"}},{{"aggregateType":"any","aggregateRules":[{{"actionHasAnyOfTags":["tagP1"]}},{{"actionStatus":"flagged"}}]}}]</string>
        <key>showsCompleted</key><false/>
        <key>sidebarVisible</key><true/>
        <key>createdAt</key><date>2026-01-02T03:04:05Z</date>
        <key>icon</key><data>
          AAEC
          AwQ=
        </data>
        <key>badInt</key><integer>x</integer>
        <key>badReal</key><real>y</real>
      </dict>
    </plist>
  </perspective>
  <perspective id="persp2">
    <plist xmlns="" version="1.0"><array><string>not a dict</string></array></plist>
  </perspective>
  <perspective id="persp5">
    <plist version="1.0"><dict><key>name</key><string>Odd</string>
    <key>filterRules</key><string>not json</string></dict></plist>
  </perspective>
  <perspective id="persp3"><plist xmlns="" version="1.0"/></perspective>
  <perspective id="persp4"/>
</omnifocus>
"""


class TestPerspectives:
    def test_decodes_plist_definition(self) -> None:
        model = build_model(make_zip(_PERSPECTIVE_XML))
        persp = model.perspectives["persp1"]
        assert persp.name == "Today"
        assert persp.added is not None and persp.added.year == 2026
        assert persp.modified is not None and persp.modified.month == 2
        d = persp.definition
        assert d["version"] == 3
        assert d["iconScale"] == 1.5
        assert d["filterRulesAggregation"] == "all"
        assert d["filterRules"][0] == {"actionAvailability": "available"}
        nested = d["filterRules"][1]
        assert nested["aggregateType"] == "any"
        assert nested["aggregateRules"][0] == {"actionHasAnyOfTags": ["tagP1"]}
        assert d["showsCompleted"] is False
        assert d["sidebarVisible"] is True
        assert d["createdAt"] == "2026-01-02T03:04:05Z"
        assert d["icon"] == "AAECAwQ="
        assert d["badInt"] == "x"
        assert d["badReal"] == "y"

    def test_keeps_undecodable_filter_rules_as_text(self) -> None:
        model = build_model(make_zip(_PERSPECTIVE_XML))
        assert model.perspectives["persp5"].definition["filterRules"] == "not json"

    def test_tolerates_missing_or_odd_plists(self) -> None:
        model = build_model(make_zip(_PERSPECTIVE_XML))
        for pid in ("persp2", "persp3", "persp4"):
            persp = model.perspectives[pid]
            assert persp.name == ""
            assert persp.definition == {}
            assert persp.added is None

    def test_transaction_updates_and_deletes(self) -> None:
        update = make_zip(
            f'<omnifocus xmlns="{NS[1:-1]}">'
            '<perspective id="persp1" op="update"><plist version="1.0"><dict>'
            "<key>name</key><string>Now</string></dict></plist></perspective>"
            '<perspective id="persp2" op="delete"/>'
            "</omnifocus>"
        )
        model = build_model(make_zip(_PERSPECTIVE_XML), [update])
        persp = model.perspectives["persp1"]
        assert persp.name == "Now"
        assert persp.definition == {"name": "Now"}
        assert persp.added is not None
        assert "persp2" not in model.perspectives
