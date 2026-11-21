"""OmniFocus .ofocus bundle parser.

Parses the ZIP-based ``.ofocus`` bundle format used by OmniFocus 4.  The bundle
consists of one *baseline* ZIP (``00000000000000=<id>.zip``) and zero or more
*transaction* ZIPs (``<ISO8601_timestamp>=<clientID>+<parentID>.zip``).

Each ZIP contains a single ``contents.xml`` file with an ``<omnifocus>``
root element.  The baseline contains the full snapshot; transactions contain
incremental changes using upsert / delete semantics.

Merge semantics
---------------
For every element with an ``id`` attribute in a transaction:

- ``op="update"`` elements merge only the changed fields into the prior snapshot.
- ``op="delete"`` elements remove the id from the model.
- Legacy name-less deletion markers are still accepted for task and context
  elements, but not for folders where partial updates may legitimately omit
  ``<name>``.

Transactions are applied in lexicographic filename order, which equals
chronological order because filenames are ISO 8601 UTC timestamps.

Usage::

    from omnifocus.parser import build_model

    with open("baseline.zip", "rb") as f:
        baseline_bytes = f.read()

    model = build_model(baseline_bytes, transaction_bytes_list=[])
"""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import io
import json
import xml.etree.ElementTree as ET
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from omnifocus.errors import OFParseError
from omnifocus.models import Folder, OFModel, Perspective, Project, Tag, Task

# OmniFocus v2 XML namespace
_NS = "{http://www.omnigroup.com/namespace/OmniFocus/v2}"

# Sentinel: a datetime that will never match a real modification time
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Low-level ZIP / XML helpers
# ---------------------------------------------------------------------------


def load_xml_from_zip(zip_bytes: bytes) -> ET.Element:
    """Parse ``contents.xml`` from a ZIP archive given as raw bytes.

    Args:
        zip_bytes: Raw bytes of a ``.zip`` file containing ``contents.xml``.

    Returns:
        The root ``ET.Element`` of the parsed XML document.

    Raises:
        OFParseError: If the bytes are not a valid ZIP, if ``contents.xml``
            is missing, or if the XML is malformed.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            try:
                data = zf.read("contents.xml")
            except KeyError as exc:
                raise OFParseError("contents.xml not found in ZIP archive") from exc
    except zipfile.BadZipFile as exc:
        raise OFParseError(f"Invalid ZIP archive: {exc}") from exc

    try:
        return ET.fromstring(data.decode("utf-8"))  # noqa: S314
    except ET.ParseError as exc:
        raise OFParseError(f"Malformed XML in contents.xml: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise OFParseError(f"contents.xml is not valid UTF-8: {exc}") from exc


def _tag(local: str) -> str:
    """Return the fully-qualified tag name with the OmniFocus namespace prefix."""
    return f"{_NS}{local}"


def _text(el: ET.Element, local: str, default: str = "") -> str:
    """Return the text content of a child element, or *default* if absent."""
    child = el.find(_tag(local))
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _bool(el: ET.Element, local: str, default: bool = False) -> bool:
    """Return the boolean text value of a child element."""
    raw = _text(el, local)
    if not raw:
        return default
    return raw.lower() == "true"


def _int(el: ET.Element, local: str, default: int | None = None) -> int | None:
    """Return the integer text value of a child element, or *default*."""
    raw = _text(el, local)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_dt_utc(value: str) -> datetime | None:
    """Parse an ISO 8601 UTC timestamp (with trailing ``Z``) to a UTC datetime.

    Args:
        value: A string like ``"2026-03-22T15:40:11.347Z"``.  Returns ``None``
            for empty strings.
    """
    if not value:
        return None
    # Normalise the Z suffix to +00:00 for fromisoformat (Python 3.11+)
    normalised = value.rstrip("Z") + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalised)
    except ValueError:
        return None


def _parse_dt_local(value: str) -> datetime | None:
    """Parse a local-time ISO 8601 datetime (no timezone suffix).

    Args:
        value: A string like ``"2026-03-23T19:00:00.000"``.  Returns ``None``
            for empty strings.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _idref(el: ET.Element, local: str) -> str | None:
    """Return the ``idref`` attribute of a single child element, or ``None``."""
    child = el.find(_tag(local))
    if child is None:
        return None
    return child.get("idref")


# ---------------------------------------------------------------------------
# Element indexing (merge step)
# ---------------------------------------------------------------------------

# Raw element index: tag-local-name → {id: ET.Element}
_RawIndex = dict[str, dict[str, ET.Element]]


def _merge_update_element(existing: ET.Element, update: ET.Element) -> ET.Element:
    """Merge an ``op="update"`` element into an existing indexed element.

    OmniFocus update transactions often contain only the changed fields. Replacing the
    whole indexed element would discard unchanged fields from earlier snapshots.
    """
    merged = deepcopy(existing)
    for attr_name, attr_value in update.attrib.items():
        if attr_name != "op":
            merged.set(attr_name, attr_value)

    merged_children: dict[str, ET.Element] = {child.tag: child for child in list(merged)}
    for child in update:
        existing_child = merged_children.get(child.tag)
        if existing_child is None:
            merged.append(deepcopy(child))
            continue

        # ``<project>`` acts as a nested container whose fields are often updated
        # partially. Replacing the whole child would drop the earlier folder link.
        # A self-closing ``<project/>`` is different: it changes a project back
        # into a plain task and must replace the prior nested payload entirely.
        if child.tag == _tag("project"):
            if len(list(child)) == 0:
                merged.remove(existing_child)
                cloned_child = deepcopy(child)
                merged.append(cloned_child)
                merged_children[child.tag] = cloned_child
                continue
            nested = _merge_update_element(existing_child, child)
            merged.remove(existing_child)
            merged.append(nested)
            merged_children[child.tag] = nested
            continue

        merged.remove(existing_child)
        cloned_child = deepcopy(child)
        merged.append(cloned_child)
        merged_children[child.tag] = cloned_child
    return merged


def _materialize_reference_snapshot(elem: ET.Element) -> ET.Element:
    """Return a concrete element built from a ``reference-snapshot`` wrapper.

    OmniFocus transactions sometimes ship ``op="reference"`` elements whose
    actual fields live under ``<reference-snapshot>``. Storing the wrapper
    directly would discard fields such as ``<name>`` or parent ``idref`` links.
    """
    snapshot = elem.find(_tag("reference-snapshot"))
    if snapshot is None:
        return elem

    attrs = {key: value for key, value in elem.attrib.items() if key != "op"}
    materialized = ET.Element(elem.tag, attrs)
    for child in snapshot:
        materialized.append(deepcopy(child))
    return materialized


def _index_elements(root: ET.Element, index: _RawIndex) -> None:
    """Index all direct children of *root* into *index* using upsert/delete semantics.

    Elements without an ``id`` attribute are silently ignored (e.g. setting
    elements that have no id).

    Args:
        root: An ``<omnifocus>`` element from baseline or a transaction ZIP.
        index: Mutable dict to update in-place.
    """
    for elem in root:
        eid = elem.get("id")
        if not eid:
            continue
        local = elem.tag.replace(_NS, "")
        bucket = index.setdefault(local, {})
        op = elem.get("op")

        if op == "update":
            existing = bucket.get(eid)
            if existing is None:
                bucket[eid] = elem
            else:
                bucket[eid] = _merge_update_element(existing, elem)
            continue

        if op == "reference":
            reference = _materialize_reference_snapshot(elem)
            existing = bucket.get(eid)
            if existing is None:
                bucket[eid] = reference
            else:
                bucket[eid] = _merge_update_element(existing, reference)
            continue

        if op == "delete":
            bucket.pop(eid, None)
            continue

        # Legacy deletion marker: element is present but has no <name> child.
        # Applies to task/context elements. Folder updates may legitimately
        # omit ``<name>`` and still carry useful metadata such as parent/rank.
        has_name = elem.find(_tag("name")) is not None
        if not has_name and local in ("task", "context"):
            bucket.pop(eid, None)
        else:
            bucket[eid] = elem


# ---------------------------------------------------------------------------
# Folder builder
# ---------------------------------------------------------------------------


def _build_folder(el: ET.Element) -> Folder | None:
    """Convert a raw ``<folder>`` XML element into a :class:`Folder`.

    Returns ``None`` if mandatory fields are missing or unparseable.
    """
    fid = el.get("id")
    name = _text(el, "name")
    if not fid or not name:
        return None

    added_raw = _text(el, "added")
    modified_raw = _text(el, "modified")

    return Folder(
        id=fid,
        name=name,
        parent_folder_id=_idref(el, "folder"),
        rank=_int(el, "rank", 0) or 0,
        added=_parse_dt_utc(added_raw) or _EPOCH,
        modified=_parse_dt_utc(modified_raw) or _EPOCH,
    )


# ---------------------------------------------------------------------------
# Tag / context builder
# ---------------------------------------------------------------------------


def _build_tag(el: ET.Element) -> Tag | None:
    """Convert a raw ``<context>`` XML element into a :class:`Tag`."""
    tid = el.get("id")
    name = _text(el, "name")
    if not tid or not name:
        return None
    return Tag(
        id=tid,
        name=name,
        parent_tag_id=_idref(el, "context"),
        rank=_int(el, "rank", 0) or 0,
        added=_parse_dt_utc(_text(el, "added")),
        modified=_parse_dt_utc(_text(el, "modified")),
        note=_text(el, "note") or "",
        hidden=_parse_dt_utc(_text(el, "hidden")),
    )


# ---------------------------------------------------------------------------
# Perspective builder
# ---------------------------------------------------------------------------


def _localname(tag: str) -> str:
    """Return *tag* without its ``{namespace}`` prefix."""
    return tag.rsplit("}", 1)[-1]


def _plist_value(el: ET.Element) -> Any:
    """Decode one Apple plist value element into JSON-safe Python data.

    Namespace-agnostic: inside ``contents.xml`` the plist inherits the
    OmniFocus default namespace. Dates stay ISO strings and ``<data>`` stays
    base64 text, so the result serialises as JSON unchanged.
    """
    kind = _localname(el.tag)
    text = (el.text or "").strip()
    if kind == "dict":
        out: dict[str, Any] = {}
        children = list(el)
        for key_el, value_el in zip(children[::2], children[1::2], strict=False):
            out[(key_el.text or "").strip()] = _plist_value(value_el)
        return out
    if kind == "array":
        return [_plist_value(child) for child in el]
    if kind == "true":
        return True
    if kind == "false":
        return False
    if kind == "integer":
        try:
            return int(text)
        except ValueError:
            return text
    if kind == "real":
        try:
            return float(text)
        except ValueError:
            return text
    if kind == "data":
        return "".join(text.split())
    return text


def _build_perspective(el: ET.Element) -> Perspective:
    """Convert a raw ``<perspective>`` XML element into a :class:`Perspective`."""
    definition: dict[str, Any] = {}
    plist = el.find(_tag("plist"))
    if plist is None:
        plist = el.find("plist")
    if plist is not None and len(plist):
        decoded = _plist_value(plist[0])
        if isinstance(decoded, dict):
            definition = decoded
    # OmniFocus stores the rule tree as a JSON string inside the plist; decode it so
    # callers (and id resolution) see the rules rather than one opaque string.
    rules = definition.get("filterRules")
    if isinstance(rules, str):
        try:
            definition["filterRules"] = json.loads(rules)
        except ValueError:
            pass
    name = definition.get("name")
    return Perspective(
        id=el.get("id", ""),
        name=name if isinstance(name, str) else "",
        definition=definition,
        added=_parse_dt_utc(_text(el, "added")),
        modified=_parse_dt_utc(_text(el, "modified")),
    )


# ---------------------------------------------------------------------------
# Project / task builder helpers
# ---------------------------------------------------------------------------


def _is_project(task_el: ET.Element) -> bool:
    """Return True if the ``<task>`` element represents a project container.

    A task element is a *project* when its ``<project>`` child element exists
    **and** has at least one child element (e.g. ``<folder>``, ``<status>``,
    ``<singleton>``).  A plain task has a self-closing ``<project/>`` with no
    children.
    """
    proj_child = task_el.find(_tag("project"))
    return proj_child is not None and len(list(proj_child)) > 0


def _build_project(el: ET.Element) -> Project | None:
    """Convert a project ``<task>`` element into a :class:`Project`."""
    pid = el.get("id")
    name = _text(el, "name")
    if not pid or not name:
        return None

    proj = el.find(_tag("project"))
    folder_id: str | None = None
    status = "active"
    singleton = False
    if proj is not None:
        folder_id = _idref(proj, "folder")
        raw_status = _text(proj, "status")
        if raw_status:
            status = raw_status
        singleton = _bool(proj, "singleton")
    if folder_id is None:
        folder_id = _idref(el, "folder")

    tag_ids = tuple(c.get("idref", "") for c in el.findall(_tag("context")) if c.get("idref"))

    return Project(
        id=pid,
        name=name,
        folder_id=folder_id,
        status=status,
        singleton=singleton,
        rank=_int(el, "rank", 0) or 0,
        added=_parse_dt_utc(_text(el, "added")) or _EPOCH,
        modified=_parse_dt_utc(_text(el, "modified")) or _EPOCH,
        flagged=_bool(el, "flagged"),
        due=_parse_dt_local(_text(el, "due")),
        start=_parse_dt_local(_text(el, "start")),
        note=_text(el, "note"),
        completed=_parse_dt_utc(_text(el, "completed")),
        last_review=(_parse_dt_utc(_text(proj, "last-review")) if proj is not None else None),
        next_review=(_parse_dt_utc(_text(proj, "next-review")) if proj is not None else None),
        review_interval=((_text(proj, "review-interval") or None) if proj is not None else None),
        tag_ids=tag_ids,
        repetition_rule=_text(el, "repetition-rule") or None,
        repetition_method=_text(el, "repetition-method") or None,
        repetition_schedule_type=_text(el, "repetition-schedule-type") or None,
        repetition_anchor_date=_text(el, "repetition-anchor-date") or None,
        catch_up_automatically=_bool(el, "catch-up-automatically"),
        next_clone_identifier=_int(el, "next-clone-identifier", 0) or 0,
        due_date_alarm_policy=_text(el, "due-date-alarm-policy") or None,
        defer_date_alarm_policy=_text(el, "defer-date-alarm-policy") or None,
        latest_time_to_start_alarm_policy=_text(el, "latest-time-to-start-alarm-policy") or None,
        planned_date_alarm_policy=_text(el, "planned-date-alarm-policy") or None,
    )


def _build_task(el: ET.Element, project_id: str | None) -> Task | None:
    """Convert a leaf/intermediate ``<task>`` XML element into a :class:`Task`.

    Args:
        el: The ``<task>`` XML element.
        project_id: Pre-resolved containing project id, or ``None`` for inbox tasks.
    """
    tid = el.get("id")
    name = _text(el, "name")
    if not tid or not name:
        return None

    # Hidden: an empty <hidden/> means not hidden; a timestamp means dropped.
    hidden_raw = _text(el, "hidden")
    hidden = _parse_dt_utc(hidden_raw) if hidden_raw else None

    tag_ids = tuple(c.get("idref", "") for c in el.findall(_tag("context")) if c.get("idref"))

    return Task(
        id=tid,
        name=name,
        parent_task_id=_idref(el, "task"),
        project_id=project_id,
        inbox=_bool(el, "inbox"),
        completed=_parse_dt_utc(_text(el, "completed")),
        flagged=_bool(el, "flagged"),
        due=_parse_dt_local(_text(el, "due")),
        start=_parse_dt_local(_text(el, "start")),
        hidden=hidden,
        note=_text(el, "note"),
        rank=_int(el, "rank", 0) or 0,
        repetition_rule=_text(el, "repetition-rule") or None,
        estimated_minutes=_int(el, "estimated-minutes"),
        tag_ids=tag_ids,
        added=_parse_dt_utc(_text(el, "added")) or _EPOCH,
        modified=_parse_dt_utc(_text(el, "modified")) or _EPOCH,
        order=_text(el, "order") or "parallel",
        repetition_method=_text(el, "repetition-method") or None,
        repetition_schedule_type=_text(el, "repetition-schedule-type") or None,
        repetition_anchor_date=_text(el, "repetition-anchor-date") or None,
        catch_up_automatically=_bool(el, "catch-up-automatically"),
        next_clone_identifier=_int(el, "next-clone-identifier", 0) or 0,
        due_date_alarm_policy=_text(el, "due-date-alarm-policy") or None,
        defer_date_alarm_policy=_text(el, "defer-date-alarm-policy") or None,
        latest_time_to_start_alarm_policy=_text(el, "latest-time-to-start-alarm-policy") or None,
        planned_date_alarm_policy=_text(el, "planned-date-alarm-policy") or None,
    )


# ---------------------------------------------------------------------------
# Parent chain resolution
# ---------------------------------------------------------------------------


def _resolve_project_ids(
    raw_tasks: dict[str, ET.Element],
    project_ids: set[str],
) -> dict[str, str | None]:
    """Build a mapping from every task id to its containing project id.

    Uses memoised traversal of the parent chain (``<task idref="..."/>``) to
    avoid O(n²) behaviour on deeply-nested tasks.

    Args:
        raw_tasks: The raw XML element dict for all ``<task>`` elements.
        project_ids: Set of task ids that are projects.

    Returns:
        Dict mapping each task id → project id (or ``None`` for inbox tasks
        that have no parent project).
    """
    memo: dict[str, str | None] = {}

    def resolve(tid: str, visited: set[str]) -> str | None:
        if tid in memo:
            return memo[tid]
        if tid in visited:
            # Cycle guard — should never happen in a valid OF database
            memo[tid] = None
            return None
        if tid in project_ids:
            memo[tid] = tid
            return tid
        visited.add(tid)
        el = raw_tasks.get(tid)
        if el is None:
            memo[tid] = None
            return None
        parent_idref = _idref(el, "task")
        if parent_idref is None:
            memo[tid] = None
            return None
        result = resolve(parent_idref, visited)
        memo[tid] = result
        return result

    for tid in raw_tasks:
        if tid not in memo:
            resolve(tid, set())

    return memo


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_model(
    baseline_bytes: bytes,
    transaction_bytes_list: list[bytes] | None = None,
) -> OFModel:
    """Parse a ``.ofocus`` bundle into an :class:`OFModel`.

    Applies the baseline first, then each transaction in the provided order
    (callers are responsible for sorting transactions chronologically by
    filename before passing them here).

    Args:
        baseline_bytes: Raw bytes of the baseline ``00000000000000=*.zip``.
        transaction_bytes_list: Optional list of raw ZIP bytes for incremental
            transactions, in chronological order.

    Returns:
        A fully populated :class:`OFModel`.

    Raises:
        OFParseError: If any ZIP or XML is malformed.
    """
    if transaction_bytes_list is None:
        transaction_bytes_list = []

    # Step 1: build raw element index from baseline
    index: _RawIndex = {}
    baseline_root = load_xml_from_zip(baseline_bytes)
    _index_elements(baseline_root, index)

    # Step 2: apply transactions (chronological order maintained by caller)
    for tx_bytes in transaction_bytes_list:
        tx_root = load_xml_from_zip(tx_bytes)
        _index_elements(tx_root, index)

    # Step 3: build folders
    folders: dict[str, Folder] = {}
    for el in index.get("folder", {}).values():
        folder = _build_folder(el)
        if folder is not None:
            folders[folder.id] = folder

    # Step 4: build tags from <context> elements
    tags: dict[str, Tag] = {}
    for el in index.get("context", {}).values():
        tag = _build_tag(el)
        if tag is not None:
            tags[tag.id] = tag

    # Step 5: separate project elements from task elements
    raw_task_els: dict[str, ET.Element] = index.get("task", {})
    project_els: dict[str, ET.Element] = {}
    leaf_els: dict[str, ET.Element] = {}

    for tid, el in raw_task_els.items():
        if _is_project(el):
            project_els[tid] = el
        else:
            leaf_els[tid] = el

    # Step 6: build projects
    projects: dict[str, Project] = {}
    for el in project_els.values():
        project = _build_project(el)
        if project is not None:
            projects[project.id] = project

    # Step 7: resolve parent project ids for all leaf tasks
    project_id_map = _resolve_project_ids(raw_task_els, set(project_els.keys()))

    # Step 8: build tasks
    tasks: dict[str, Task] = {}
    for tid, el in leaf_els.items():
        project_id = project_id_map.get(tid)
        task = _build_task(el, project_id)
        if task is not None:
            tasks[task.id] = task

    # Step 9: perspectives (filter rules, grouping, sorting live in their plist)
    perspectives = {pid: _build_perspective(el) for pid, el in index.get("perspective", {}).items()}

    return OFModel(
        folders=folders,
        projects=projects,
        tasks=tasks,
        tags=tags,
        perspectives=perspectives,
    )
