"""MCP server for OmniFocus CLI.

Exposes OmniFocus task management as MCP tools consumable by Claude.
Runs over stdio transport (suitable for ``podman run --rm -i``).

Tools
-----
``list_tasks``      Filter active tasks by inbox/today/flagged/project/due.
``search_tasks``    Fuzzy search tasks by name.
``get_task``        Retrieve a single task by id.
``add_task``        Create a new task.
``complete_task``   Mark a task as completed.
``update_task``     Update task fields and state.
``get_project``     Retrieve a single project by id.
``add_project``     Create a new project.
``update_project``  Update a project.
``complete_project`` Mark a project completed.
``list_projects``   List projects (optionally filtered by status).
``list_projects_for_review`` List projects that are due for review.
``mark_project_reviewed`` Stamp a project as reviewed.
``list_folders``    List all folders.
``get_folder``      Retrieve a single folder by id.
``get_folder_tree`` Return the nested folder/project tree.
``add_folder``      Create a new folder.
``update_folder``   Update a folder.
``drop_folder``     Drop a folder.
``list_tags``       List all tags.
``get_tag``         Retrieve a single tag by id.
``list_perspectives`` Read perspective definitions (filter rules, grouping, sorting).
``add_tag``         Create a new tag.
``update_tag``      Update a tag.
``drop_tag``        Drop a tag.
``sync_now``        Trigger a full WebDAV sync.

Usage::

    # Native Python entry point
    of-mcp

    # Container default: MCP server mode (stdin/stdout)
    podman run --rm -i of

    # Explicit container MCP mode
    podman run --rm -i of mcp

    # In Claude MCP config (settings.json):
    {
      "mcpServers": {
        "omnifocus": {
          "command": "podman",
          "args": ["run", "--rm", "-i",
                   "-e", "OF_WEBDAV_URL",
                   "-e", "OF_WEBDAV_USER",
                   "-e", "OF_WEBDAV_PASS",
                   "-e", "OF_ENCRYPTION_PASSPHRASE",
                   "of:latest"]
        }
      }
    }
"""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import date
from typing import Any, cast

import uvicorn
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from omnifocus.api_common import (
    folder_summary,
    parse_optional_date,
    parse_optional_utc_datetime,
    project_review_sort_key,
    project_summary,
    serialise_json,
    tag_summary,
    task_summary,
    validate_folder_parent_change,
    validate_tag_parent_change,
)
from omnifocus.api_service import StoreBackedApiService
from omnifocus.errors import OFError
from omnifocus.fuzzy import find_tasks
from omnifocus.models import OFModel
from omnifocus.store import OFocusStore
from omnifocus.write_contract import ContractError, WriteContract, load_contract

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

server: Server = Server("omnifocus")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialise(obj: Any) -> Any:
    """Recursively serialise an object to a JSON-safe form."""
    return serialise_json(obj)


def _text(data: Any) -> list[TextContent]:
    """Wrap any JSON-serialisable data as a list of MCP TextContent."""
    return [
        TextContent(
            type="text",
            text=json.dumps(_serialise(data), ensure_ascii=False, indent=2),
        )
    ]


async def _service_text(coro: Awaitable[Any]) -> list[TextContent]:
    """Convert service results and service-layer errors into MCP text payloads."""
    try:
        return _text(await coro)
    except OFError as exc:
        return _text({"error": str(exc)})


async def _load_model(force: bool = False) -> OFModel:
    """Load the current OFModel via the store."""
    async with OFocusStore.from_env() as store:
        return await store.load(force_refresh=force)


async def _warm_model() -> None:
    """Best-effort pre-load of the model so the first client request skips the
    cold sync (pull + decrypt + parse of the whole delta chain).

    Run once at HTTP startup. Failures (e.g. the sync server is unreachable) are
    swallowed so they never crash the server — the next read simply syncs lazily.
    """
    try:
        await _load_model()
    except Exception:
        logger.warning("startup model warm-up failed; first read will sync lazily", exc_info=True)


def _service() -> StoreBackedApiService:
    """Return a store-backed service using MCP-local loader/store hooks."""
    return StoreBackedApiService(load_model=_load_model, store_factory=OFocusStore.from_env)


# ---------------------------------------------------------------------------
# Tool: list_tasks
# ---------------------------------------------------------------------------


_READ_ONLY_TOOLS = frozenset(
    {
        "list_tasks",
        "search_tasks",
        "get_task",
        "get_project",
        "list_projects",
        "list_projects_for_review",
        "list_folders",
        "get_folder",
        "get_folder_tree",
        "list_tags",
        "get_tag",
        "list_perspectives",
        "sync_now",
    }
)
# Writes that remove/complete data; everything else not read-only is a
# non-destructive add/update.
_DESTRUCTIVE_TOOLS = frozenset({"complete_task", "complete_project", "drop_folder", "drop_tag"})


def _annotate(tools: list[Tool]) -> list[Tool]:
    """Attach read-only / destructive behaviour hints to each tool.

    This is the only tool grouping the MCP spec lets a server express (there is
    no category/group field): clients such as claude.ai/Notion split tools into
    read-only vs write/destructive from these hints. It also mirrors the
    harness-layer write policy — reads and writes become distinct surfaces.
    """
    annotated: list[Tool] = []
    for tool in tools:
        if tool.name in _READ_ONLY_TOOLS:
            hints = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
        else:
            hints = ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=tool.name in _DESTRUCTIVE_TOOLS,
                openWorldHint=False,
            )
        annotated.append(tool.model_copy(update={"annotations": hints}))
    return annotated


_SOURCE_SCHEMA = {
    "type": "string",
    "description": (
        "Who is writing: '<surface> · <channel or routine>', e.g. 'Claude Code · inbox "
        "sweep'. Becomes note line 1 as 'via <source> · YYYY-MM-DD'."
    ),
}
_USER_ASKED_SCHEMA = {
    "type": "string",
    "description": (
        "Only when the user explicitly asked for a project, tag, date or flag on this "
        "task: the user's own words. Lifts the write contract's field checks; logged."
    ),
}


def _apply_contract_schema(tools: list[Tool], contract: WriteContract | None) -> list[Tool]:
    """Advertise the write-contract parameters when the contract is on."""
    if contract is None:
        return tools
    out: list[Tool] = []
    for tool in tools:
        if tool.name in ("add_task", "update_task"):
            schema = json.loads(json.dumps(tool.inputSchema))
            schema["properties"]["user_asked"] = _USER_ASKED_SCHEMA
            if tool.name == "add_task":
                schema["properties"]["source"] = _SOURCE_SCHEMA
                schema["required"] = [*schema["required"], "source"]
            note = (
                " Follows the OmniFocus write contract: agent tasks go to the Inbox with "
                "no project, defer or flag (tags and due only where the contract allows), "
                "and note line 1 is the source line."
            )
            tool = tool.model_copy(
                update={"inputSchema": schema, "description": (tool.description or "") + note}
            )
        out.append(tool)
    return out


@server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
async def list_tools() -> list[Tool]:
    """Return the list of all MCP tools provided by this server."""
    try:
        contract = load_contract()
    except ContractError:
        # A bad contract must not take reads down with it: list_tools runs before every
        # call. Advertise the strict contract; the write handlers still fail closed.
        logger.exception("invalid write contract; task writes are refused until it is fixed")
        contract = WriteContract()
    return _apply_contract_schema(_all_tools(), contract)


def _all_tools() -> list[Tool]:
    return _annotate(
        [
            Tool(
                name="list_tasks",
                description=(
                    "List OmniFocus tasks. By default returns active (incomplete) tasks. "
                    "Use status=completed with completed_on/completed_since to see finished "
                    "work (e.g. what was completed today). Optionally filter by inbox, today "
                    "(due today or overdue), flagged, due date, project name, tag, or folder."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["active", "completed", "dropped", "all"],
                            "description": "Base set: active (default), completed, dropped, or all",
                        },
                        "completed_on": {
                            "type": "string",
                            "description": (
                                "Only tasks completed on this local date (ISO YYYY-MM-DD or "
                                "today/yesterday). Implies status=completed."
                            ),
                        },
                        "completed_since": {
                            "type": "string",
                            "description": (
                                "Only tasks completed on or after this local date (ISO or "
                                "today/yesterday). Implies status=completed."
                            ),
                        },
                        "inbox": {"type": "boolean", "description": "Inbox tasks only"},
                        "today": {"type": "boolean", "description": "Due today or overdue"},
                        "flagged": {"type": "boolean", "description": "Flagged tasks only"},
                        "due": {"type": "boolean", "description": "Tasks with any due date"},
                        "no_due": {"type": "boolean", "description": "Tasks with no due date"},
                        "no_defer": {"type": "boolean", "description": "Tasks with no defer date"},
                        "available": {
                            "type": "boolean",
                            "description": "Available now (no defer date or defer date reached)",
                        },
                        "overdue": {"type": "boolean", "description": "Due date in the past"},
                        "has_project": {
                            "type": "boolean",
                            "description": "Tasks assigned to a project (excludes inbox)",
                        },
                        "project": {"type": "string", "description": "Project name substring"},
                        "project_status": {
                            "type": "string",
                            "enum": ["active", "inactive", "all"],
                            "description": (
                                "Filter by containing project status (default all). "
                                "'active' also keeps inbox/loose tasks."
                            ),
                        },
                        "tag": {"type": "string", "description": "Tag name substring"},
                        "tag_id": {"type": "string", "description": "Exact tag ID"},
                        "folder": {
                            "type": "string",
                            "description": "Folder name substring (includes nested subfolders)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max tasks to return (default 50)",
                        },
                    },
                },
            ),
            Tool(
                name="search_tasks",
                description="Fuzzy search tasks by name or ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results (default 10)"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_task",
                description="Get a single task by its OmniFocus ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID"},
                    },
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="add_task",
                description="Create a new OmniFocus task.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Task name"},
                        "project": {"type": "string", "description": "Project name (substring)"},
                        "parent_task_id": {
                            "type": "string",
                            "description": "Nest under this task/project ID (excludes project)",
                        },
                        "due": {
                            "type": "string",
                            "description": "Due date ISO 8601 or natural (today/tomorrow/mon-sun)",
                        },
                        "defer": {
                            "type": "string",
                            "description": (
                                "Defer date ISO 8601 or natural (today/tomorrow/mon-sun)"
                            ),
                        },
                        "flagged": {"type": "boolean"},
                        "note": {"type": "string"},
                        "repeat_every": {
                            "type": "string",
                            "description": "Recurrence interval, e.g. '30d', '6w', '3m', '1y'",
                        },
                        "repeat_from": {
                            "type": "string",
                            "enum": [
                                "fixed",
                                "completion",
                                "start-after-completion",
                                "due-after-completion",
                            ],
                            "description": "Recurrence anchor (default fixed calendar schedule)",
                        },
                        "repetition_rule": {
                            "type": "string",
                            "description": "Raw OmniFocus RRULE, overrides repeat_every",
                        },
                        "repetition_method": {
                            "type": "string",
                            "description": "Raw OmniFocus repetition-method override",
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="complete_task",
                description="Mark a task as completed by ID or fuzzy name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Task ID or name fragment"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="update_task",
                description="Update a task's fields or mark it dropped.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "name": {"type": "string"},
                        "project_id": {
                            "type": "string",
                            "description": "Move task into this project",
                        },
                        "clear_project": {
                            "type": "boolean",
                            "description": "Remove project assignment and move task to inbox",
                        },
                        "inbox": {
                            "type": "boolean",
                            "description": "When true, move task to inbox",
                        },
                        "due": {
                            "type": "string",
                            "description": "ISO 8601 datetime or empty to clear",
                        },
                        "defer": {
                            "type": "string",
                            "description": "ISO 8601 datetime or empty to clear",
                        },
                        "flagged": {"type": "boolean"},
                        "note": {"type": "string"},
                        "estimate": {
                            "type": ["integer", "string"],
                            "description": "Estimated minutes or empty to clear",
                        },
                        "tag_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Replace tag IDs on the task",
                        },
                        "clear_tags": {"type": "boolean"},
                        "dropped": {"type": "boolean"},
                        "repeat_every": {
                            "type": "string",
                            "description": "Recurrence interval, e.g. '30d', '6w', '3m', '1y'",
                        },
                        "repeat_from": {
                            "type": "string",
                            "enum": [
                                "fixed",
                                "completion",
                                "start-after-completion",
                                "due-after-completion",
                            ],
                            "description": "Recurrence anchor (default fixed calendar schedule)",
                        },
                        "repetition_rule": {
                            "type": "string",
                            "description": "Raw OmniFocus RRULE, overrides repeat_every",
                        },
                        "repetition_method": {
                            "type": "string",
                            "description": "Raw OmniFocus repetition-method override",
                        },
                    },
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="get_project",
                description="Get a single project by its OmniFocus ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string", "description": "Project ID"},
                    },
                    "required": ["project_id"],
                },
            ),
            Tool(
                name="add_project",
                description="Create a new OmniFocus project.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "folder": {"type": "string", "description": "Folder name substring"},
                        "due": {"type": "string", "description": "Due date ISO 8601 or natural"},
                        "defer": {
                            "type": "string",
                            "description": "Defer date ISO 8601 or natural",
                        },
                        "flagged": {"type": "boolean"},
                        "note": {"type": "string"},
                        "status": {"type": "string", "enum": ["active", "inactive"]},
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="update_project",
                description="Update a project's fields, status, or folder assignment.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "name": {"type": "string"},
                        "folder_id": {
                            "type": "string",
                            "description": "Move project into this folder",
                        },
                        "clear_folder": {
                            "type": "boolean",
                            "description": "Remove folder assignment from the project",
                        },
                        "due": {
                            "type": "string",
                            "description": "ISO 8601 datetime or empty to clear",
                        },
                        "defer": {
                            "type": "string",
                            "description": "ISO 8601 datetime or empty to clear",
                        },
                        "flagged": {"type": "boolean"},
                        "note": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["active", "inactive", "done", "dropped"],
                        },
                        "tag_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Replace tag IDs on the project",
                        },
                        "clear_tags": {"type": "boolean"},
                    },
                    "required": ["project_id"],
                },
            ),
            Tool(
                name="complete_project",
                description="Mark a project as completed by ID or fuzzy name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Project ID or name fragment"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="list_projects",
                description="List OmniFocus projects.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["active", "all", "inactive", "done", "dropped"],
                            "description": "Filter by status (default: active)",
                        },
                        "tag": {"type": "string", "description": "Tag name substring"},
                        "tag_id": {"type": "string", "description": "Exact tag ID"},
                    },
                },
            ),
            Tool(
                name="list_projects_for_review",
                description="List active and inactive projects that are due for review.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "due_only": {
                            "type": "boolean",
                            "description": "When false, include non-due projects as well",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max projects to return (default 50)",
                        },
                    },
                },
            ),
            Tool(
                name="mark_project_reviewed",
                description=(
                    "Stamp a project as reviewed and recalculate next review when possible."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string", "description": "Project ID"},
                        "reviewed_at": {
                            "type": "string",
                            "description": "Optional ISO 8601 timestamp; defaults to now in UTC",
                        },
                    },
                    "required": ["project_id"],
                },
            ),
            Tool(
                name="list_folders",
                description="List all OmniFocus folders.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="get_folder",
                description="Get a single folder by its OmniFocus ID.",
                inputSchema={
                    "type": "object",
                    "properties": {"folder_id": {"type": "string", "description": "Folder ID"}},
                    "required": ["folder_id"],
                },
            ),
            Tool(
                name="get_folder_tree",
                description="Return the nested folder hierarchy with direct child projects.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="add_folder",
                description="Create a new OmniFocus folder.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "parent_folder_id": {"type": "string"},
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="update_folder",
                description="Rename or move a folder under another folder.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder_id": {"type": "string"},
                        "name": {"type": "string"},
                        "parent_folder_id": {"type": "string"},
                        "clear_parent": {"type": "boolean"},
                    },
                    "required": ["folder_id"],
                },
            ),
            Tool(
                name="drop_folder",
                description="Drop a folder by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {"folder_id": {"type": "string"}},
                    "required": ["folder_id"],
                },
            ),
            Tool(
                name="list_tags",
                description="List OmniFocus tags/contexts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "all": {
                            "type": "boolean",
                            "description": "Include dropped/hidden tags",
                        }
                    },
                },
            ),
            Tool(
                name="get_tag",
                description="Get a single tag by its OmniFocus ID.",
                inputSchema={
                    "type": "object",
                    "properties": {"tag_id": {"type": "string", "description": "Tag ID"}},
                    "required": ["tag_id"],
                },
            ),
            Tool(
                name="list_perspectives",
                description=(
                    "Read OmniFocus perspective definitions as synced with the database: "
                    "name plus the decoded plist (filter rules and their aggregation, "
                    "grouping, sorting, focus, icon). Tags, projects and folders the "
                    "definition names by id are resolved under 'references'. Read-only."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Case-insensitive name substring filter",
                        }
                    },
                },
            ),
            Tool(
                name="add_tag",
                description="Create a new OmniFocus tag.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "parent_tag_id": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="update_tag",
                description="Rename or move a tag under another tag.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tag_id": {"type": "string"},
                        "name": {"type": "string"},
                        "parent_tag_id": {"type": "string"},
                        "clear_parent": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                    "required": ["tag_id"],
                },
            ),
            Tool(
                name="drop_tag",
                description="Drop a tag by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {"tag_id": {"type": "string"}},
                    "required": ["tag_id"],
                },
            ),
            Tool(
                name="sync_now",
                description="Trigger a full sync from the WebDAV server.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


@server.call_tool()  # type: ignore[untyped-decorator]
async def call_tool_result(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """MCP entry point: run ``call_tool`` and mark error payloads with ``isError``.

    Every failure here (unknown tool, OFError, write-contract rejection, not found)
    is a ``{"error": ...}`` JSON payload. Without ``isError`` a client that only
    reads the flag would count those as successes.
    """
    content = await call_tool(name, arguments)
    return CallToolResult(content=list(content), isError=_is_error(content))


def _is_error(content: list[TextContent]) -> bool:
    try:
        data = json.loads(content[0].text)
    except IndexError, ValueError:
        return False
    return isinstance(data, dict) and "error" in data


async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch incoming tool calls to the appropriate handler."""
    handlers: dict[str, Any] = {
        "list_tasks": _handle_list_tasks,
        "search_tasks": _handle_search_tasks,
        "get_task": _handle_get_task,
        "add_task": _handle_add_task,
        "complete_task": _handle_complete_task,
        "update_task": _handle_update_task,
        "get_project": _handle_get_project,
        "add_project": _handle_add_project,
        "update_project": _handle_update_project,
        "complete_project": _handle_complete_project,
        "list_projects": _handle_list_projects,
        "list_projects_for_review": _handle_list_projects_for_review,
        "mark_project_reviewed": _handle_mark_project_reviewed,
        "list_folders": _handle_list_folders,
        "get_folder": _handle_get_folder,
        "get_folder_tree": _handle_get_folder_tree,
        "add_folder": _handle_add_folder,
        "update_folder": _handle_update_folder,
        "drop_folder": _handle_drop_folder,
        "list_tags": _handle_list_tags,
        "get_tag": _handle_get_tag,
        "list_perspectives": _handle_list_perspectives,
        "add_tag": _handle_add_tag,
        "update_tag": _handle_update_tag,
        "drop_tag": _handle_drop_tag,
        "sync_now": _handle_sync_now,
    }
    handler = handlers.get(name)
    if handler is None:
        return _text({"error": f"Unknown tool: {name}"})
    try:
        typed_handler = cast(Callable[[dict[str, Any]], Awaitable[list[TextContent]]], handler)
        return await typed_handler(arguments)
    except (OFError, ContractError) as exc:
        return _text({"error": str(exc)})


async def _handle_list_tasks(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(
        _service().list_tasks(
            status=args.get("status", "active"),
            completed_on=args.get("completed_on"),
            completed_since=args.get("completed_since"),
            inbox=bool(args.get("inbox")),
            today=bool(args.get("today")),
            flagged=bool(args.get("flagged")),
            due=bool(args.get("due")),
            no_due=bool(args.get("no_due")),
            no_defer=bool(args.get("no_defer")),
            available=bool(args.get("available")),
            overdue=bool(args.get("overdue")),
            has_project=bool(args.get("has_project")),
            project=args.get("project"),
            project_status=str(args.get("project_status", "all")),
            tag=args.get("tag"),
            tag_id=args.get("tag_id"),
            folder=args.get("folder"),
            limit=int(args.get("limit", 50)),
        )
    )


async def _handle_search_tasks(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(
        _service().search_tasks(
            query=str(args.get("query", "")),
            limit=int(args.get("limit", 10)),
        )
    )


async def _handle_get_task(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(_service().get_task(task_id=str(args.get("task_id", ""))))


async def _handle_add_task(args: dict[str, Any]) -> list[TextContent]:
    if args.get("tag_ids") or args.get("tags"):
        # add_task has no tag field; say so rather than create the task without them.
        return _text(
            {
                "error": "add_task cannot set tags. Add the task, then call update_task "
                "with its task_id and tag_ids (from list_tags)."
            }
        )
    model = await _load_model()
    project_id: str | None = None
    project_name: str | None = None
    if args.get("project"):
        needle = str(args["project"]).lower()
        matches = [
            project
            for project in model.projects.values()
            if needle in project.name.lower() and project.status not in {"done", "dropped"}
        ]
        if not matches:
            return _text({"error": f"No project matching {args['project']!r}"})
        project_id = matches[0].id
        project_name = matches[0].name
    note = str(args.get("note", ""))
    contract = load_contract()
    source = str(args.get("source", "")).strip()
    user_asked = str(args.get("user_asked", "")).strip()
    if contract is not None:
        if not source:
            return _text(
                {
                    "error": "source is required by the OmniFocus write contract: "
                    "'<surface> · <channel or routine>', written as note line 1."
                }
            )
        today = date.today()
        if not user_asked:
            rejected = contract.rejected_fields(
                source=source,
                project_name=project_name,
                due=str(args["due"]) if "due" in args else None,
                defer=str(args["defer"]) if "defer" in args else None,
                flagged=bool(args.get("flagged", False)),
                parent=bool(args.get("parent_task_id")),
                repeat=_sets_repeat(args),
                today=today,
            )
            if rejected:
                return _text({"error": contract.rejection(rejected)})
        note = WriteContract.stamp_note(note, source, today)
    result = await _service_text(
        _service().add_task(
            name=str(args.get("name", "")),
            project_id=project_id,
            parent_task_id=str(args["parent_task_id"]) if "parent_task_id" in args else None,
            validation_model=model,
            due=str(args["due"]) if "due" in args else None,
            defer=str(args["defer"]) if "defer" in args else None,
            flagged=bool(args.get("flagged", False)),
            note=note,
            **_repeat_kwargs(args),
        )
    )
    if contract is not None:
        _log_write(contract, "add_task", source, result, args, user_asked)
    return result


_REPEAT_KEYS = ("repeat_every", "repeat_from", "repetition_rule", "repetition_method")


def _sets_repeat(args: dict[str, Any]) -> bool:
    return any(args.get(key) for key in _REPEAT_KEYS)


def _repeat_kwargs(args: dict[str, Any]) -> dict[str, str | None]:
    return {key: str(args[key]) if key in args else None for key in _REPEAT_KEYS}


def _log_write(
    contract: WriteContract,
    tool: str,
    source: str | None,
    result: list[TextContent],
    args: dict[str, Any],
    user_asked: str,
) -> None:
    """Record a successful write in the contract's append-only log."""
    data = json.loads(result[0].text)
    if not isinstance(data, dict) or "error" in data:
        return
    skip = {"task_id", "source", "user_asked"}
    entry: dict[str, Any] = {
        "tool": tool,
        "source": source,
        "task_id": data.get("task_id") or args.get("task_id"),
        "fields": sorted(key for key in args if key not in skip),
    }
    if user_asked:
        entry["user_asked"] = user_asked
    contract.log(entry)


async def _handle_complete_task(args: dict[str, Any]) -> list[TextContent]:
    query = str(args.get("query", ""))
    model = await _load_model()
    results = find_tasks(query, model.active_tasks, limit=5)
    if not results:
        return _text({"error": f"No active task matching {query!r}"})
    task = results[0].task

    async with OFocusStore.from_env() as store:
        result = await store.complete_task(task)

    return _text(result)


async def _handle_update_task(args: dict[str, Any]) -> list[TextContent]:
    contract = load_contract()
    user_asked = str(args.get("user_asked", "")).strip()
    source: str | None = None
    note = str(args["note"]) if "note" in args else None
    if contract is not None:
        model = await _load_model()
        task = model.tasks.get(str(args.get("task_id", "")))
        if task is not None and WriteContract.is_agent_task(task.note):
            source = task.note.split("\n", 1)[0]
            if note is not None and not WriteContract.is_agent_task(note):
                # Keep the source line: it is how the task stays recognisable.
                note = f"{source}\n{note}" if note else source
            if task.inbox and not user_asked:
                project = model.projects.get(str(args.get("project_id", "")))
                rejected = contract.rejected_fields(
                    source=source,
                    project_name=(
                        (project.name if project else str(args["project_id"]))
                        if "project_id" in args
                        else None
                    ),
                    due=str(args["due"]) if "due" in args else None,
                    defer=str(args["defer"]) if "defer" in args else None,
                    flagged=bool(args.get("flagged", False)),
                    tags=bool(args.get("tag_ids")),
                    repeat=_sets_repeat(args),
                    today=date.today(),
                )
                if rejected:
                    return _text({"error": contract.rejection(rejected, update=True)})
    tag_ids = tuple(str(tag_id) for tag_id in args["tag_ids"]) if "tag_ids" in args else None
    result = await _service_text(
        _service().update_task(
            task_id=str(args.get("task_id", "")),
            name=str(args["name"]) if "name" in args else None,
            project_id=str(args["project_id"]) if "project_id" in args else None,
            clear_project=bool(args.get("clear_project", False)),
            inbox=bool(args["inbox"]) if "inbox" in args else None,
            due=str(args["due"]) if "due" in args else None,
            defer=str(args["defer"]) if "defer" in args else None,
            flagged=bool(args["flagged"]) if "flagged" in args else None,
            note=note,
            estimate=cast(int | str | None, args["estimate"]) if "estimate" in args else None,
            tag_ids=tag_ids,
            clear_tags=bool(args.get("clear_tags", False)),
            dropped=bool(args["dropped"]) if "dropped" in args else None,
            **_repeat_kwargs(args),
        )
    )
    if contract is not None:
        _log_write(contract, "update_task", source, result, args, user_asked)
    return result


async def _handle_add_project(args: dict[str, Any]) -> list[TextContent]:
    model = await _load_model()
    folder_id: str | None = None
    if args.get("folder"):
        needle = str(args["folder"]).lower()
        matches = [folder for folder in model.folders.values() if needle in folder.name.lower()]
        if not matches:
            return _text({"error": f"No folder matching {args['folder']!r}"})
        if len(matches) > 1:
            return _text({"error": f"Multiple folders match {args['folder']!r}"})
        folder_id = matches[0].id
    return await _service_text(
        _service().add_project(
            name=str(args.get("name", "")),
            folder_id=folder_id,
            validation_model=model,
            due=str(args["due"]) if "due" in args else None,
            defer=str(args["defer"]) if "defer" in args else None,
            flagged=bool(args.get("flagged", False)),
            note=str(args.get("note", "")),
            status=str(args.get("status", "active")),
        )
    )


async def _handle_get_project(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(_service().get_project(project_id=str(args.get("project_id", ""))))


async def _handle_update_project(args: dict[str, Any]) -> list[TextContent]:
    tag_ids = tuple(str(tag_id) for tag_id in args["tag_ids"]) if "tag_ids" in args else None
    return await _service_text(
        _service().update_project(
            project_id=str(args.get("project_id", "")),
            name=str(args["name"]) if "name" in args else None,
            folder_id=str(args["folder_id"]) if "folder_id" in args else None,
            clear_folder=bool(args.get("clear_folder", False)),
            due=str(args["due"]) if "due" in args else None,
            defer=str(args["defer"]) if "defer" in args else None,
            flagged=bool(args["flagged"]) if "flagged" in args else None,
            note=str(args["note"]) if "note" in args else None,
            status=str(args["status"]) if "status" in args else None,
            tag_ids=tag_ids,
            clear_tags=bool(args.get("clear_tags", False)),
        )
    )


async def _handle_complete_project(args: dict[str, Any]) -> list[TextContent]:
    query = str(args.get("query", ""))
    model = await _load_model()
    project = model.projects.get(query)
    if project is None:
        needle = query.lower()
        matches = [
            candidate for candidate in model.projects.values() if needle in candidate.name.lower()
        ]
        if not matches:
            return _text({"error": f"No project matching {query!r}"})
        if len(matches) > 1:
            return _text({"error": f"Multiple projects match {query!r}"})
        project = matches[0]

    return await _service_text(_service().complete_project(project_id=project.id))


async def _handle_list_projects(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(
        _service().list_projects(
            status=str(args.get("status", "active")),
            tag=str(args["tag"]) if "tag" in args else None,
            tag_id=str(args["tag_id"]) if "tag_id" in args else None,
        )
    )


async def _handle_list_projects_for_review(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(
        _service().list_projects_for_review(
            due_only=bool(args.get("due_only", True)),
            limit=int(args.get("limit", 50)),
        )
    )


async def _handle_mark_project_reviewed(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(
        _service().mark_project_reviewed(
            project_id=str(args.get("project_id", "")),
            reviewed_at=str(args["reviewed_at"]) if "reviewed_at" in args else None,
        )
    )


async def _handle_list_folders(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(_service().list_folders())


async def _handle_get_folder(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(_service().get_folder(folder_id=str(args.get("folder_id", ""))))


async def _handle_get_folder_tree(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(_service().get_folder_tree())


async def _handle_add_folder(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(
        _service().add_folder(
            name=str(args.get("name", "")),
            parent_folder_id=str(args["parent_folder_id"]) if "parent_folder_id" in args else None,
        )
    )


async def _handle_update_folder(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(
        _service().update_folder(
            folder_id=str(args.get("folder_id", "")),
            name=str(args["name"]) if "name" in args else None,
            parent_folder_id=str(args["parent_folder_id"]) if "parent_folder_id" in args else None,
            clear_parent=bool(args.get("clear_parent", False)),
        )
    )


async def _handle_drop_folder(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(_service().drop_folder(folder_id=str(args.get("folder_id", ""))))


async def _handle_list_tags(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(_service().list_tags(include_hidden=bool(args.get("all", False))))


async def _handle_get_tag(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(_service().get_tag(tag_id=str(args.get("tag_id", ""))))


async def _handle_list_perspectives(args: dict[str, Any]) -> list[TextContent]:
    name = args.get("name")
    return await _service_text(
        _service().list_perspectives(name=str(name) if name is not None else None)
    )


async def _handle_add_tag(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(
        _service().add_tag(
            name=str(args.get("name", "")),
            parent_tag_id=str(args["parent_tag_id"]) if "parent_tag_id" in args else None,
            note=str(args.get("note", "")),
        )
    )


async def _handle_update_tag(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(
        _service().update_tag(
            tag_id=str(args.get("tag_id", "")),
            name=str(args["name"]) if "name" in args else None,
            parent_tag_id=str(args["parent_tag_id"]) if "parent_tag_id" in args else None,
            clear_parent=bool(args.get("clear_parent", False)),
            note=str(args["note"]) if "note" in args else None,
        )
    )


async def _handle_drop_tag(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(_service().drop_tag(tag_id=str(args.get("tag_id", ""))))


async def _handle_sync_now(args: dict[str, Any]) -> list[TextContent]:
    return await _service_text(_service().sync_now())


def _task_summary(task: Any, model: OFModel) -> dict[str, Any]:
    """Return a concise dict representation of a task."""
    return task_summary(task, model)


def _project_summary(project: Any, model: OFModel, *, now: Any = None) -> dict[str, Any]:
    """Return a concise dict representation of a project."""
    return project_summary(project, model, now=now)


def _project_review_sort_key(summary: dict[str, Any], review_state: Any) -> tuple[Any, ...]:
    """Return a stable sort key for review queues."""
    return project_review_sort_key(summary, review_state)


def _folder_summary(folder: Any, model: OFModel) -> dict[str, Any]:
    """Return a concise dict representation of a folder."""
    return folder_summary(folder, model)


def _tag_summary(tag: Any, model: OFModel) -> dict[str, Any]:
    """Return a concise dict representation of a tag."""
    return tag_summary(tag, model)


def _parse_optional_date(value: str | None) -> Any:
    """Parse an optional ISO 8601 date/datetime string."""
    return parse_optional_date(value)


def _parse_optional_utc_datetime(value: str | None) -> Any:
    """Parse an optional ISO 8601 UTC timestamp string."""
    return parse_optional_utc_datetime(value)


def _validate_folder_parent_change(
    *,
    model: OFModel,
    folder_id: str,
    parent_folder_id: str | None,
    clear_parent: bool,
) -> str | None:
    """Validate requested folder reparenting."""
    return validate_folder_parent_change(
        model=model,
        folder_id=folder_id,
        parent_folder_id=parent_folder_id,
        clear_parent=clear_parent,
    )


def _validate_tag_parent_change(
    *,
    model: OFModel,
    tag_id: str,
    parent_tag_id: str | None,
    clear_parent: bool,
) -> str | None:
    """Validate requested tag reparenting."""
    return validate_tag_parent_change(
        model=model,
        tag_id=tag_id,
        parent_tag_id=parent_tag_id,
        clear_parent=clear_parent,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the MCP server over stdio.

    This is the entry point registered as ``of-mcp`` in ``pyproject.toml``.
    """

    async def _serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_serve())


def build_http_app(
    *,
    warm: bool = False,
    health_path: str = "/healthz",
    json_response: bool = False,
    stateless: bool = False,
    session_idle_timeout: float | None = 600.0,
) -> Starlette:
    """Build the Starlette app that serves this MCP server over Streamable HTTP.

    ``StreamableHTTPSessionManager`` wraps the same in-process ``server`` — there
    is NO per-session child process. The decrypted OmniFocus bundle stays warm
    across requests, and client cancellations are handled by the SDK's anyio task
    group instead of crashing the transport. This replaces the supergateway proxy
    that fronted the stdio MCP and bit us three times: the 2026-06-14 memory leak
    (orphaned children → 14 GiB), the 2026-06-15 zombie pileup, and the 2026-06-16
    cancel-race that took the whole bridge down on a slow cold-cache query.

    The endpoint speaks plaintext HTTP — TLS terminates upstream at the Cloudflare
    tunnel edge, and local agents reach it over loopback. Health lives at
    ``health_path`` for the container/orchestrator probe.

    The MCP transport is mounted at the app root rather than under a ``/mcp``
    prefix on purpose. Starlette's ``Mount("/mcp", ...)`` 307-redirects the bare
    ``/mcp`` to ``/mcp/``; behind the CF tunnel that ``Location`` is rewritten to
    the internal origin (``http://of-bridge:8096/mcp/`` — unreachable, and scheme
    downgraded to http), so a remote MCP client that POSTs to ``/mcp`` without a
    trailing slash never completes ``initialize`` even after OAuth succeeds
    (the Notion "Login" loop, 2026-06-18). The session manager routes on method /
    headers / session id, not on the URL path, so a root mount serves ``/mcp`` and
    ``/mcp/`` identically with no redirect — matching the retired supergateway's
    behaviour. ``health_path`` is registered first so the probe still wins.

    In production the session manager runs STATELESS (``run_http`` and the
    ``mcp --http`` container entrypoint pass ``stateless=True``): every request is
    self-contained, no per-session state is held, and no ``Mcp-Session-Id`` is
    issued. This is deliberate. A stateful in-process session idle-expires
    (``session_idle_timeout``) and does not survive the mcp-remote bridge's
    "initialize at startup, call the tool minutes later" pattern through the
    Cloudflare edge: the keepalive SSE stream is cut, the session is evicted, and
    every later request fails with ``-32600 Session not found`` (POST) or ``404``
    (GET SSE reconnect) -- the 2026-07-08 Claude Desktop hang. First-party clients
    (Claude Code, claude.ai, Notion) never tripped it because they fire
    ``initialize`` and the call back-to-back. Stateless removes the whole failure
    class; OmniFocus exposes no server->client notifications, so nothing is lost.
    Pass ``--stateful`` (``stateless=False``) to restore per-session state.

    When ``warm`` is set the model is pre-loaded once during lifespan startup
    (best-effort) so the first client request skips the cold sync; ``run_http``
    enables it for the long-running server.
    """
    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=json_response,
        stateless=stateless,
        # The SDK forbids session_idle_timeout in stateless mode (there are no
        # sessions to expire), so only pass it through when running stateful.
        session_idle_timeout=None if stateless else session_idle_timeout,
    )

    async def handle_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            if warm:
                await _warm_model()
            yield

    return Starlette(
        routes=[
            Route(health_path, health, methods=["GET"]),
            Mount("", app=handle_mcp),
        ],
        lifespan=lifespan,
    )


def run_http(
    *,
    host: str = "0.0.0.0",  # noqa: S104 — container must listen on all interfaces (cloudflared reaches of-bridge:8096 on the proxy net)
    port: int = 8096,
    health_path: str = "/healthz",
    json_response: bool = False,
    stateless: bool = True,
    session_idle_timeout: float | None = 600.0,
) -> None:
    """Serve the OmniFocus MCP server over Streamable HTTP via uvicorn.

    Single long-running process; no supergateway, no per-session fork. See
    :func:`build_http_app` for why this transport replaces the old proxy.
    """
    app = build_http_app(
        warm=True,
        health_path=health_path,
        json_response=json_response,
        stateless=stateless,
        session_idle_timeout=session_idle_timeout,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
        server_header=False,
        date_header=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
