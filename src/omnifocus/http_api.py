"""FastAPI-based HTTPS REST transport for ``omnifocus-cli``.

This module exposes the store-backed OmniFocus service layer over a private,
TLS-only HTTPS API intended for machine-to-machine use. It enforces Bearer
authentication on every endpoint, including the OpenAPI schema endpoint.
"""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import argparse
import asyncio
import hmac
import json
import logging
import os
import ssl
import time
import uuid
from collections import deque
from collections.abc import Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import uvicorn
from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from omnifocus import __version__
from omnifocus.api_service import StoreBackedApiService, default_api_service
from omnifocus.env import read_env_or_file
from omnifocus.errors import OFError, OFHTTPError
from omnifocus.mcp_http import MCPHTTPApp

_MAX_JSON_BODY_BYTES = 1024 * 1024
_DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost")
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
_AUTH_FAILURE_LIMIT = 5
_AUTH_FAILURE_WINDOW_SECONDS = 60.0
_HSTS_HEADER_VALUE = "max-age=63072000; includeSubDomains"
_LOGGER = logging.getLogger("omnifocus.http_api")


@dataclass(frozen=True)
class HTTPServerConfig:
    """Immutable runtime configuration for the HTTPS API."""

    host: str
    port: int
    api_keys: tuple[str, ...]
    tls_cert_file: Path
    tls_key_file: Path
    allowed_hosts: tuple[str, ...] = _DEFAULT_ALLOWED_HOSTS

    @property
    def api_key(self) -> str:
        """Return the primary API key.

        This property keeps the single-key v1 behavior while leaving room for
        future key rotation through ``api_keys``.
        """

        return self.api_keys[0]

    @classmethod
    def from_env(cls) -> HTTPServerConfig:
        """Load and validate HTTPS API configuration from environment variables."""

        api_key = read_env_or_file("OF_HTTP_API_KEY", error_type=OFError)
        cert_path = os.getenv("OF_HTTP_TLS_CERT_FILE")
        key_path = os.getenv("OF_HTTP_TLS_KEY_FILE")
        missing = [
            name
            for name, value in (
                ("OF_HTTP_API_KEY", api_key),
                ("OF_HTTP_TLS_CERT_FILE", cert_path),
                ("OF_HTTP_TLS_KEY_FILE", key_path),
            )
            if not value
        ]
        if missing:
            _emit_log(
                logging.ERROR,
                "security.startup.misconfiguration",
                {"missing_env_vars": tuple(missing)},
            )
            raise OFError(f"Missing required HTTP environment variables: {', '.join(missing)}")
        assert api_key is not None
        assert cert_path is not None
        assert key_path is not None

        host = os.getenv("OF_HTTP_HOST", "127.0.0.1")
        port_raw = os.getenv("OF_HTTP_PORT", "8443")
        try:
            port = int(port_raw)
        except ValueError as exc:
            _emit_log(
                logging.ERROR,
                "security.startup.invalid_port",
                {"configured_port": port_raw},
            )
            raise OFError(f"Invalid OF_HTTP_PORT: {port_raw!r}") from exc
        if not (1 <= port <= 65535):
            _emit_log(
                logging.ERROR,
                "security.startup.invalid_port",
                {"configured_port": port_raw},
            )
            raise OFError(f"Invalid OF_HTTP_PORT: {port_raw!r}")

        cert_file = Path(cert_path)
        key_file = Path(key_path)
        if not cert_file.is_file():
            _emit_log(
                logging.ERROR,
                "security.startup.missing_tls_file",
                {"file_kind": "certificate", "path": str(cert_file)},
            )
            raise OFError(f"TLS certificate file not found: {cert_file}")
        if not key_file.is_file():
            _emit_log(
                logging.ERROR,
                "security.startup.missing_tls_file",
                {"file_kind": "key", "path": str(key_file)},
            )
            raise OFError(f"TLS key file not found: {key_file}")

        allowed_hosts_raw = os.getenv("OF_HTTP_ALLOWED_HOSTS", ",".join(_DEFAULT_ALLOWED_HOSTS))
        allowed_hosts = tuple(
            host_item.strip() for host_item in allowed_hosts_raw.split(",") if host_item.strip()
        )
        if not allowed_hosts:
            _emit_log(
                logging.ERROR,
                "security.startup.invalid_allowed_hosts",
                {"configured_allowed_hosts": allowed_hosts_raw},
            )
            raise OFError("OF_HTTP_ALLOWED_HOSTS must contain at least one host")

        return cls(
            host=host,
            port=port,
            api_keys=(api_key,),
            tls_cert_file=cert_file,
            tls_key_file=key_file,
            allowed_hosts=allowed_hosts,
        )


@dataclass
class _AuthFailureBucket:
    """Sliding-window authentication failure bucket."""

    failures: deque[float] = field(default_factory=deque)


class _AuthFailureLimiter:
    """In-memory brute-force limiter for invalid Bearer authentication attempts."""

    def __init__(self, *, limit: int, window_seconds: float) -> None:
        """Initialise the limiter."""

        self._limit = limit
        self._window_seconds = window_seconds
        self._buckets: dict[str, _AuthFailureBucket] = {}

    def _prune(self, client_key: str, now_monotonic: float) -> _AuthFailureBucket:
        """Return a pruned bucket for the client key."""

        bucket = self._buckets.setdefault(client_key, _AuthFailureBucket())
        threshold = now_monotonic - self._window_seconds
        while bucket.failures and bucket.failures[0] < threshold:
            bucket.failures.popleft()
        if not bucket.failures:
            self._buckets.pop(client_key, None)
            bucket = self._buckets.setdefault(client_key, _AuthFailureBucket())
        return bucket

    def is_limited(self, client_key: str, *, now_monotonic: float) -> bool:
        """Return whether the client is currently rate-limited."""

        bucket = self._prune(client_key, now_monotonic)
        return len(bucket.failures) >= self._limit

    def record_failure(self, client_key: str, *, now_monotonic: float) -> int:
        """Record a failed attempt and return the current count in the window."""

        bucket = self._prune(client_key, now_monotonic)
        bucket.failures.append(now_monotonic)
        return len(bucket.failures)

    def reset(self, client_key: str) -> None:
        """Reset the failure bucket for a successful client."""

        self._buckets.pop(client_key, None)


class EnvelopeModel(BaseModel):
    """Base class for public HTTP API schemas.

    All request and response models inherit the same strict `extra="forbid"` policy so the
    OpenAPI contract, runtime validation, and tests agree on accepted fields.
    """

    model_config = ConfigDict(extra="forbid")


class ErrorDetail(EnvelopeModel):
    """Machine-readable error detail payload returned inside error envelopes."""

    code: str
    message: str


class ErrorEnvelope(EnvelopeModel):
    """Standard top-level error envelope for all HTTP failures."""

    ok: Literal[False]
    error: ErrorDetail


class SuccessEnvelope[T](EnvelopeModel):
    """Standard top-level success envelope wrapping typed response data."""

    ok: Literal[True]
    data: T


class HealthData(EnvelopeModel):
    """Minimal authenticated liveness payload for the HTTPS transport."""

    status: Literal["ok"]
    transport: Literal["https"]
    version: str


class SyncResultModel(EnvelopeModel):
    """Forced-sync summary containing top-level object counts."""

    status: str
    tasks: int
    projects: int
    folders: int
    tags: int


class TaskSummaryModel(EnvelopeModel):
    """Canonical HTTP task summary returned by list and get endpoints.

    This model mirrors the transport-facing task projection: stable IDs, task state, dates,
    notes, and both raw `tag_ids` plus resolved `tag_names`.
    """

    id: str
    name: str
    project: str | None
    project_id: str | None
    project_status: str | None = None
    project_folder_name: str | None = None
    inbox: bool
    flagged: bool
    due: datetime | None
    start: datetime | None
    defer: datetime | None = None
    completed: datetime | None
    note: str
    tag_ids: list[str]
    tag_names: list[str]


class TaskSearchResultModel(TaskSummaryModel):
    """Task summary extended with a fuzzy-match score."""

    score: float


class TaskMutationResult(EnvelopeModel):
    """Minimal mutation result returned by task write endpoints."""

    status: str
    task_id: str
    name: str | None = None


class ProjectSummaryModel(EnvelopeModel):
    """Canonical HTTP project summary returned by list, get, and review endpoints.

    The schema intentionally includes both core OmniFocus fields and transport-added convenience
    fields such as `folder_name`, `tag_names`, `review_due`, and `review_basis`.
    """

    id: str
    name: str
    folder_id: str | None
    status: str
    singleton: bool
    rank: int
    added: datetime
    modified: datetime
    flagged: bool
    due: datetime | None
    start: datetime | None
    note: str
    completed: datetime | None
    last_review: datetime | None = None
    next_review: datetime | None = None
    review_interval: str | None = None
    tag_ids: tuple[str, ...]
    repetition_rule: str | None = None
    repetition_method: str | None = None
    repetition_schedule_type: str | None = None
    repetition_anchor_date: str | None = None
    catch_up_automatically: bool = False
    next_clone_identifier: int = 0
    due_date_alarm_policy: str | None = None
    defer_date_alarm_policy: str | None = None
    latest_time_to_start_alarm_policy: str | None = None
    planned_date_alarm_policy: str | None = None
    folder_name: str | None
    tag_names: list[str]
    review_due: bool
    review_basis: Literal["next_review", "interval", "unknown"]


class ProjectMutationResult(EnvelopeModel):
    """Minimal mutation result returned by project write endpoints."""

    status: str
    project_id: str
    name: str | None = None


class ProjectReviewMutationResult(ProjectSummaryModel):
    """Project summary returned after a review stamp operation."""

    next_review_recalculated: bool


class FolderSummaryModel(EnvelopeModel):
    """Folder summary returned by flat folder endpoints.

    Besides the folder's own fields, the payload includes direct child folder IDs and direct child
    project IDs so non-tree clients can navigate relationships cheaply.
    """

    id: str
    name: str
    parent_folder_id: str | None
    rank: int
    added: datetime
    modified: datetime
    child_folder_ids: list[str] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)


class FolderTreeFolderModel(EnvelopeModel):
    """Folder projection embedded inside recursive tree responses."""

    id: str
    name: str
    parent_folder_id: str | None
    rank: int
    added: datetime
    modified: datetime


class FolderTreeProjectModel(EnvelopeModel):
    """Project projection embedded inside folder-tree responses."""

    id: str
    name: str
    folder_id: str | None
    status: str
    singleton: bool
    rank: int
    added: datetime
    modified: datetime
    flagged: bool
    due: datetime | None
    start: datetime | None
    note: str
    completed: datetime | None
    last_review: datetime | None = None
    next_review: datetime | None = None
    review_interval: str | None = None
    tag_ids: tuple[str, ...]
    repetition_rule: str | None = None
    repetition_method: str | None = None
    repetition_schedule_type: str | None = None
    repetition_anchor_date: str | None = None
    catch_up_automatically: bool = False
    next_clone_identifier: int = 0
    due_date_alarm_policy: str | None = None
    defer_date_alarm_policy: str | None = None
    latest_time_to_start_alarm_policy: str | None = None
    planned_date_alarm_policy: str | None = None


class FolderTreeNodeModel(EnvelopeModel):
    """Recursive folder tree node with nested folders and direct child projects."""

    folder: FolderTreeFolderModel
    children: list[FolderTreeNodeModel] = Field(default_factory=list)
    projects: list[FolderTreeProjectModel] = Field(default_factory=list)


class FolderTreeDataModel(EnvelopeModel):
    """Top-level folder tree response, including dangling and no-folder projects."""

    folders: list[FolderTreeNodeModel] = Field(default_factory=list)
    no_folder_projects: list[FolderTreeProjectModel] = Field(default_factory=list)
    dangling_folder_projects: list[FolderTreeProjectModel] = Field(default_factory=list)


class FolderMutationResult(EnvelopeModel):
    """Minimal mutation result returned by folder write endpoints."""

    status: str
    folder_id: str
    name: str | None = None


class TagSummaryModel(EnvelopeModel):
    """Canonical HTTP tag summary, including parent and child references."""

    id: str
    name: str
    parent_tag_id: str | None
    rank: int
    added: datetime | None = None
    modified: datetime | None = None
    note: str = ""
    hidden: datetime | None = None
    parent_name: str | None
    child_tag_ids: list[str] = Field(default_factory=list)


class TagMutationResult(EnvelopeModel):
    """Minimal mutation result returned by tag write endpoints."""

    status: str
    tag_id: str
    name: str | None = None


class AddTaskRequest(EnvelopeModel):
    """Request body for creating tasks.

    The transport accepts human-friendly due strings here because creation still allows the
    higher-level CLI/MCP date vocabulary at the HTTP edge.
    """

    name: str
    project_id: str | None = None
    parent_task_id: str | None = None
    due: str | None = None
    defer: str | None = None
    flagged: bool = False
    note: str = ""
    repeat_every: str | None = None
    repeat_from: str | None = None
    repetition_rule: str | None = None
    repetition_method: str | None = None


class UpdateTaskRequest(EnvelopeModel):
    """Request body for updating tasks by stable ID.

    This schema models conflicting move and tag operations explicitly so FastAPI can reject
    malformed shapes before they reach the service layer.
    """

    name: str | None = None
    project_id: str | None = None
    clear_project: bool = False
    inbox: bool | None = None
    due: str | None = None
    defer: str | None = None
    flagged: bool | None = None
    note: str | None = None
    estimate: int | Literal[""] | None = None
    tag_ids: list[str] | None = None
    clear_tags: bool = False
    dropped: bool | None = None
    repeat_every: str | None = None
    repeat_from: str | None = None
    repetition_rule: str | None = None
    repetition_method: str | None = None


class AddProjectRequest(EnvelopeModel):
    """Request body for creating projects assigned to a folder by stable ID."""

    name: str
    folder_id: str | None = None
    due: str | None = None
    defer: str | None = None
    flagged: bool = False
    note: str = ""
    status: Literal["active", "inactive"] = "active"


class UpdateProjectRequest(EnvelopeModel):
    """Request body for updating projects by stable ID."""

    name: str | None = None
    folder_id: str | None = None
    clear_folder: bool = False
    due: str | None = None
    defer: str | None = None
    flagged: bool | None = None
    note: str | None = None
    status: Literal["active", "inactive", "done", "dropped"] | None = None
    tag_ids: list[str] | None = None
    clear_tags: bool = False


class ReviewProjectRequest(EnvelopeModel):
    """Optional review-marking request body with an explicit UTC timestamp override."""

    reviewed_at: datetime | None = None


class AddFolderRequest(EnvelopeModel):
    """Request body for creating folders, optionally under a parent folder ID."""

    name: str
    parent_folder_id: str | None = None


class UpdateFolderRequest(EnvelopeModel):
    """Request body for renaming or reparenting folders."""

    name: str | None = None
    parent_folder_id: str | None = None
    clear_parent: bool = False


class AddTagRequest(EnvelopeModel):
    """Request body for creating tags, optionally under a parent tag ID."""

    name: str
    parent_tag_id: str | None = None
    note: str = ""


class UpdateTagRequest(EnvelopeModel):
    """Request body for renaming or reparenting tags."""

    name: str | None = None
    parent_tag_id: str | None = None
    clear_parent: bool = False
    note: str | None = None


FolderTreeNodeModel.model_rebuild()


_ERROR_RESPONSES: dict[int, dict[str, object]] = {
    400: {"model": ErrorEnvelope, "description": "Malformed request"},
    401: {"model": ErrorEnvelope, "description": "Missing or invalid bearer token"},
    404: {"model": ErrorEnvelope, "description": "Resource not found"},
    409: {"model": ErrorEnvelope, "description": "Semantic conflict"},
    422: {"model": ErrorEnvelope, "description": "Validation failure"},
    429: {"model": ErrorEnvelope, "description": "Too many authentication failures"},
    500: {"model": ErrorEnvelope, "description": "Internal server error"},
    504: {"model": ErrorEnvelope, "description": "Request timed out"},
}

_FASTAPI_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = cast(
    dict[int | str, dict[str, Any]],
    _ERROR_RESPONSES,
)
_REQUIRED_BODY = Body(...)
_OPTIONAL_BODY = Body(default=None)


def _emit_log(
    level: int,
    event_name: str,
    attributes: dict[str, object],
) -> None:
    """Emit a structured JSON log in an OTel-aligned shape."""

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "severity_text": logging.getLevelName(level),
        "body": event_name,
        "attributes": attributes,
    }
    _LOGGER.log(level, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _request_id_from_headers(request: Request) -> str:
    """Return the incoming request ID or create a new one."""

    header = request.headers.get("x-request-id")
    if header:
        return header
    return uuid.uuid4().hex


def _trace_id_from_header(traceparent: str | None) -> str | None:
    """Extract the W3C trace id from a ``traceparent`` header when present."""

    if traceparent is None:
        return None
    parts = traceparent.split("-")
    if len(parts) != 4:
        return None
    version, trace_id, _parent_id, _flags = parts
    if version != "00":
        return None
    if len(trace_id) != 32:
        return None
    return trace_id


def _client_address(request: Request) -> str:
    """Return the client address string for logging and rate limiting."""

    client = request.client
    return "unknown" if client is None else client.host


def _error_envelope(status_code: int, *, code: str, message: str) -> JSONResponse:
    """Return a standard JSON error envelope."""

    envelope = ErrorEnvelope(ok=False, error=ErrorDetail(code=code, message=message))
    return JSONResponse(envelope.model_dump(mode="json"), status_code=status_code)


def _success_envelope(
    data: BaseModel | list[BaseModel],
    *,
    status_code: int = 200,
) -> JSONResponse:
    """Return a standard JSON success envelope."""

    if isinstance(data, BaseModel):
        payload: object = data.model_dump(mode="json")
    else:
        payload = [item.model_dump(mode="json") for item in data]
    return JSONResponse({"ok": True, "data": payload}, status_code=status_code)


def _pydantic_message(exc: RequestValidationError) -> str:
    """Return a stable, human-readable validation error message."""

    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
    message = str(error.get("msg", "Validation failed"))
    return message if not location else f"{location}: {message}"


def build_ssl_context(config: HTTPServerConfig) -> ssl.SSLContext:
    """Build and validate the HTTPS server SSL context with TLS 1.3 minimum."""

    if not hasattr(ssl.TLSVersion, "TLSv1_3"):
        _emit_log(logging.ERROR, "security.startup.tls_unavailable", {})
        raise OFError("TLS 1.3 support is required by the HTTPS API runtime")
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(
            certfile=str(config.tls_cert_file),
            keyfile=str(config.tls_key_file),
        )
    except ssl.SSLError as exc:
        _emit_log(
            logging.ERROR,
            "security.startup.tls_configuration_failure",
            {"reason": str(exc)},
        )
        raise OFError("Failed to initialise TLS 1.3 HTTPS listener") from exc
    return context


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request IDs and emit structured request lifecycle logs."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Add request context and log the completed request."""

        request_id = _request_id_from_headers(request)
        traceparent = request.headers.get("traceparent")
        trace_id = _trace_id_from_header(traceparent)
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        response.headers["X-Request-ID"] = request_id
        _emit_log(
            logging.INFO,
            "http.request",
            {
                "request.id": request_id,
                "http.request.method": request.method,
                "http.route": route_path,
                "url.path": request.url.path,
                "http.response.status_code": response.status_code,
                "event.duration_ms": duration_ms,
                "network.client.address": _client_address(request),
                "trace.id": trace_id,
            },
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply strict response headers to every HTTP response."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Append security headers after request processing."""

        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = _HSTS_HEADER_VALUE
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        with suppress(KeyError):
            del response.headers["server"]
        return response


class JSONBodyGuardMiddleware(BaseHTTPMiddleware):
    """Reject non-JSON, malformed, or oversized request bodies before parsing."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Validate mutating request bodies."""

        if request.method in {"POST", "PATCH"}:
            body = await request.body()
            if len(body) > _MAX_JSON_BODY_BYTES:
                return _error_envelope(
                    400,
                    code="bad_request",
                    message="Request body too large",
                )
            if body:
                content_type = request.headers.get("content-type", "")
                if not content_type.lower().startswith("application/json"):
                    return _error_envelope(
                        400,
                        code="bad_request",
                        message="Request body must be application/json",
                    )
                try:
                    json.loads(body)
                except ValueError:
                    return _error_envelope(
                        400,
                        code="bad_request",
                        message="Malformed JSON body",
                    )
        return await call_next(request)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Apply a wall-clock timeout to every request."""

    def __init__(self, app: ASGIApp, *, timeout_seconds: float) -> None:
        """Initialise the middleware."""

        super().__init__(app)
        self._timeout_seconds = timeout_seconds

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Abort requests that exceed the configured timeout."""

        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await call_next(request)
        except TimeoutError:
            _emit_log(
                logging.WARNING,
                "security.request.timeout",
                {
                    "request.id": getattr(request.state, "request_id", ""),
                    "http.request.method": request.method,
                    "url.path": request.url.path,
                    "network.client.address": _client_address(request),
                },
            )
            return _error_envelope(
                504,
                code="timeout",
                message="Request processing timed out",
            )


class RateLimitedBearerAuthMiddleware(BaseHTTPMiddleware):
    """Require a valid Bearer token and apply brute-force protection."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_keys: Sequence[str],
        failure_limit: int,
        failure_window_seconds: float,
    ) -> None:
        """Initialise the middleware with accepted keys and limiter settings."""

        super().__init__(app)
        self._api_keys = tuple(api_keys)
        self._limiter = _AuthFailureLimiter(
            limit=failure_limit,
            window_seconds=failure_window_seconds,
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Reject unauthenticated requests before routing."""

        client_key = _client_address(request)
        now_monotonic = time.monotonic()
        if self._limiter.is_limited(client_key, now_monotonic=now_monotonic):
            _emit_log(
                logging.WARNING,
                "security.auth.rate_limited",
                {
                    "request.id": getattr(request.state, "request_id", ""),
                    "network.client.address": client_key,
                    "url.path": request.url.path,
                },
            )
            return _error_envelope(
                429,
                code="rate_limited",
                message="Too many authentication failures",
            )

        auth_header = request.headers.get("authorization")
        if auth_header is None:
            self._limiter.record_failure(client_key, now_monotonic=now_monotonic)
            _emit_log(
                logging.WARNING,
                "security.auth.failure",
                {
                    "request.id": getattr(request.state, "request_id", ""),
                    "network.client.address": client_key,
                    "failure_reason": "missing_header",
                    "url.path": request.url.path,
                },
            )
            return _error_envelope(
                401,
                code="unauthorized",
                message="Missing Authorization header",
            )
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            self._limiter.record_failure(client_key, now_monotonic=now_monotonic)
            _emit_log(
                logging.WARNING,
                "security.auth.failure",
                {
                    "request.id": getattr(request.state, "request_id", ""),
                    "network.client.address": client_key,
                    "failure_reason": "malformed_header",
                    "url.path": request.url.path,
                },
            )
            return _error_envelope(
                401,
                code="unauthorized",
                message="Authorization header must use Bearer token authentication",
            )
        if not any(hmac.compare_digest(token, api_key) for api_key in self._api_keys):
            attempts = self._limiter.record_failure(client_key, now_monotonic=now_monotonic)
            _emit_log(
                logging.WARNING,
                "security.auth.failure",
                {
                    "request.id": getattr(request.state, "request_id", ""),
                    "network.client.address": client_key,
                    "failure_reason": "invalid_token",
                    "attempts_in_window": attempts,
                    "url.path": request.url.path,
                },
            )
            return _error_envelope(
                401,
                code="unauthorized",
                message="Invalid API key",
            )
        self._limiter.reset(client_key)
        return await call_next(request)


class JSONTrustedHostMiddleware(TrustedHostMiddleware):
    """Trusted host validation with JSON error responses and audit logging."""

    def __init__(self, app: ASGIApp, *, allowed_hosts: Sequence[str]) -> None:
        """Initialise the middleware."""

        super().__init__(app, allowed_hosts=list(allowed_hosts))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject invalid ``Host`` headers before request routing."""

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            bytes(key).decode("latin1").lower(): bytes(value).decode("latin1")
            for key, value in cast(list[tuple[bytes, bytes]], scope["headers"])
        }
        host_header = headers.get("host", "").split(":", 1)[0]
        if self.allow_any or self._host_is_allowed(host_header):
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        _emit_log(
            logging.WARNING,
            "security.host.forbidden",
            {
                "request.id": request_id,
                "configured_allowed_hosts": tuple(self.allowed_hosts),
                "provided_host": host_header,
                "url.path": cast(str, scope.get("path", "")),
            },
        )
        response = _error_envelope(400, code="bad_request", message="Invalid Host header")
        response.headers["X-Request-ID"] = request_id
        await response(scope, receive, send)

    def _host_is_allowed(self, host_header: str) -> bool:
        """Return whether the incoming host header matches the allowlist."""

        for pattern in self.allowed_hosts:
            if pattern == host_header:
                return True
            if pattern.startswith("*") and host_header.endswith(pattern[1:]):
                return True
        return False


def _configure_openapi(app: FastAPI) -> None:
    """Install the authenticated OpenAPI generator."""

    def custom_openapi() -> dict[str, object]:
        if app.openapi_schema is not None:
            return cast(dict[str, object], app.openapi_schema)
        schema = get_openapi(
            title="OmniFocus HTTPS API",
            version=__version__,
            description=(
                "Private HTTPS JSON API for omnifocus-cli. "
                "All endpoints require Bearer authentication over TLS 1.3+."
            ),
            routes=app.routes,
        )
        components = cast(dict[str, object], schema.setdefault("components", {}))
        security_schemes = cast(dict[str, object], components.setdefault("securitySchemes", {}))
        security_schemes["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API key",
        }
        paths = cast(dict[str, object], schema.get("paths", {}))
        for path_item in paths.values():
            operations = cast(dict[str, object], path_item)
            for operation in operations.values():
                operation_dict = cast(dict[str, object], operation)
                operation_dict.setdefault("security", [{"BearerAuth": []}])
        app.openapi_schema = schema
        return cast(dict[str, object], schema)

    app.openapi = custom_openapi  # type: ignore[method-assign]  # FastAPI supports overriding this hook.


def create_app(
    *,
    api_key: str | None = None,
    api_keys: Sequence[str] | None = None,
    allowed_hosts: Sequence[str] | None = None,
    service: StoreBackedApiService | None = None,
    auth_failure_limit: int = _AUTH_FAILURE_LIMIT,
    auth_failure_window_seconds: float = _AUTH_FAILURE_WINDOW_SECONDS,
    request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
    enable_mcp: bool = False,
) -> FastAPI:
    """Create the FastAPI HTTPS application."""

    resolved_api_keys = tuple(api_keys or ((api_key,) if api_key is not None else ()))
    if not resolved_api_keys:
        raise OFError("At least one HTTP API key is required")
    resolved_service = default_api_service() if service is None else service
    resolved_allowed_hosts = tuple(allowed_hosts or _DEFAULT_ALLOWED_HOSTS)
    mcp_app = MCPHTTPApp() if enable_mcp else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        if mcp_app is None:
            yield
            return
        async with mcp_app.lifespan():
            yield

    app = FastAPI(
        title="OmniFocus HTTPS API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url="/v1/openapi.json",
        lifespan=lifespan,
    )
    _configure_openapi(app)
    if mcp_app is not None:
        app.mount("/mcp", mcp_app)

    @app.exception_handler(OFError)
    async def of_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        """Map known ``OFError`` exceptions to JSON envelopes."""

        if isinstance(exc, OFHTTPError):
            return _error_envelope(exc.status_code, code=exc.code, message=str(exc))
        return _error_envelope(500, code="internal_error", message="Internal server error")

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Convert FastAPI validation failures into the standard envelope."""

        return _error_envelope(
            422,
            code="validation_error",
            message=_pydantic_message(exc),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
        """Return a generic JSON envelope for unexpected server failures."""

        return _error_envelope(500, code="internal_error", message="Internal server error")

    @app.get(
        "/v1/health",
        response_model=SuccessEnvelope[HealthData],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def health() -> JSONResponse:
        """Return a minimal health payload."""

        return _success_envelope(
            HealthData(status="ok", version=__version__, transport="https"),
        )

    @app.post(
        "/v1/sync",
        response_model=SuccessEnvelope[SyncResultModel],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def sync_now() -> JSONResponse:
        """Synchronise the local cache from WebDAV and return counts."""

        return _success_envelope(SyncResultModel.model_validate(await resolved_service.sync_now()))

    @app.get(
        "/v1/tasks",
        response_model=SuccessEnvelope[list[TaskSummaryModel]],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def list_tasks(
        inbox: bool = Query(False),
        today: bool = Query(False),
        flagged: bool = Query(False),
        due: bool = Query(False),
        no_due: bool = Query(False),
        no_defer: bool = Query(False),
        available: bool = Query(False),
        overdue: bool = Query(False),
        has_project: bool = Query(False),
        project: str | None = Query(None),
        project_status: str = Query("all"),
        tag: str | None = Query(None),
        tag_id: str | None = Query(None),
        limit: int = Query(50),
    ) -> JSONResponse:
        """List tasks using the current filter model."""

        data = await resolved_service.list_tasks(
            inbox=inbox,
            today=today,
            flagged=flagged,
            due=due,
            no_due=no_due,
            no_defer=no_defer,
            available=available,
            overdue=overdue,
            has_project=has_project,
            project=project,
            project_status=project_status,
            tag=tag,
            tag_id=tag_id,
            limit=limit,
        )
        return _success_envelope([TaskSummaryModel.model_validate(item) for item in data])

    @app.get(
        "/v1/tasks/search",
        response_model=SuccessEnvelope[list[TaskSearchResultModel]],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def search_tasks(
        query: str = Query(..., min_length=1),
        limit: int = Query(10),
    ) -> JSONResponse:
        """Search tasks by fuzzy name or id."""

        data = await resolved_service.search_tasks(query=query, limit=limit)
        return _success_envelope([TaskSearchResultModel.model_validate(item) for item in data])

    @app.get(
        "/v1/tasks/{task_id}",
        response_model=SuccessEnvelope[TaskSummaryModel],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def get_task(task_id: str) -> JSONResponse:
        """Return a single task summary by id."""

        return _success_envelope(
            TaskSummaryModel.model_validate(await resolved_service.get_task(task_id=task_id))
        )

    @app.post(
        "/v1/tasks",
        response_model=SuccessEnvelope[TaskMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
        status_code=201,
    )
    async def add_task(body: AddTaskRequest = _REQUIRED_BODY) -> JSONResponse:
        """Create a task by project ID or in the inbox."""

        result = await resolved_service.add_task(
            name=body.name,
            project_id=body.project_id,
            parent_task_id=body.parent_task_id,
            due=body.due,
            defer=body.defer,
            flagged=body.flagged,
            note=body.note,
            repeat_every=body.repeat_every,
            repeat_from=body.repeat_from,
            repetition_rule=body.repetition_rule,
            repetition_method=body.repetition_method,
        )
        return _success_envelope(TaskMutationResult.model_validate(result), status_code=201)

    @app.patch(
        "/v1/tasks/{task_id}",
        response_model=SuccessEnvelope[TaskMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def patch_task(task_id: str, body: UpdateTaskRequest = _REQUIRED_BODY) -> JSONResponse:
        """Update a task by stable ID."""

        result = await resolved_service.update_task(
            task_id=task_id,
            name=body.name,
            project_id=body.project_id,
            clear_project=body.clear_project,
            inbox=body.inbox,
            due=body.due,
            defer=body.defer,
            flagged=body.flagged,
            note=body.note,
            estimate=body.estimate,
            tag_ids=None if body.tag_ids is None else tuple(body.tag_ids),
            clear_tags=body.clear_tags,
            dropped=body.dropped,
            repeat_every=body.repeat_every,
            repeat_from=body.repeat_from,
            repetition_rule=body.repetition_rule,
            repetition_method=body.repetition_method,
        )
        return _success_envelope(TaskMutationResult.model_validate(result))

    @app.post(
        "/v1/tasks/{task_id}/complete",
        response_model=SuccessEnvelope[TaskMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def complete_task(task_id: str) -> JSONResponse:
        """Mark a task complete by stable ID."""

        return _success_envelope(
            TaskMutationResult.model_validate(await resolved_service.complete_task(task_id=task_id))
        )

    @app.post(
        "/v1/tasks/{task_id}/drop",
        response_model=SuccessEnvelope[TaskMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def drop_task(task_id: str) -> JSONResponse:
        """Drop a task by stable ID."""

        return _success_envelope(
            TaskMutationResult.model_validate(await resolved_service.drop_task(task_id=task_id))
        )

    @app.get(
        "/v1/projects",
        response_model=SuccessEnvelope[list[ProjectSummaryModel]],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def list_projects(
        status: str = Query("active"),
        tag: str | None = Query(None),
        tag_id: str | None = Query(None),
        limit: int | None = Query(None),
    ) -> JSONResponse:
        """List projects using the current filter model."""

        data = await resolved_service.list_projects(
            status=status,
            tag=tag,
            tag_id=tag_id,
            limit=limit,
        )
        return _success_envelope([ProjectSummaryModel.model_validate(item) for item in data])

    @app.get(
        "/v1/projects/review",
        response_model=SuccessEnvelope[list[ProjectSummaryModel]],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def list_project_reviews(
        due_only: bool = Query(True),
        limit: int = Query(50),
    ) -> JSONResponse:
        """List projects in review order."""

        data = await resolved_service.list_projects_for_review(due_only=due_only, limit=limit)
        return _success_envelope([ProjectSummaryModel.model_validate(item) for item in data])

    @app.get(
        "/v1/projects/{project_id}",
        response_model=SuccessEnvelope[ProjectSummaryModel],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def get_project(project_id: str) -> JSONResponse:
        """Return a single project summary by id."""

        return _success_envelope(
            ProjectSummaryModel.model_validate(
                await resolved_service.get_project(project_id=project_id)
            )
        )

    @app.post(
        "/v1/projects",
        response_model=SuccessEnvelope[ProjectMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
        status_code=201,
    )
    async def add_project(body: AddProjectRequest = _REQUIRED_BODY) -> JSONResponse:
        """Create a project by folder ID."""

        result = await resolved_service.add_project(
            name=body.name,
            folder_id=body.folder_id,
            due=body.due,
            defer=body.defer,
            flagged=body.flagged,
            note=body.note,
            status=body.status,
        )
        return _success_envelope(ProjectMutationResult.model_validate(result), status_code=201)

    @app.patch(
        "/v1/projects/{project_id}",
        response_model=SuccessEnvelope[ProjectMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def patch_project(
        project_id: str,
        body: UpdateProjectRequest = _REQUIRED_BODY,
    ) -> JSONResponse:
        """Update a project by stable ID."""

        result = await resolved_service.update_project(
            project_id=project_id,
            name=body.name,
            folder_id=body.folder_id,
            clear_folder=body.clear_folder,
            due=body.due,
            defer=body.defer,
            flagged=body.flagged,
            note=body.note,
            status=body.status,
            tag_ids=None if body.tag_ids is None else tuple(body.tag_ids),
            clear_tags=body.clear_tags,
        )
        return _success_envelope(ProjectMutationResult.model_validate(result))

    @app.post(
        "/v1/projects/{project_id}/complete",
        response_model=SuccessEnvelope[ProjectMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def complete_project(project_id: str) -> JSONResponse:
        """Mark a project complete by stable ID."""

        return _success_envelope(
            ProjectMutationResult.model_validate(
                await resolved_service.complete_project(project_id=project_id)
            )
        )

    @app.post(
        "/v1/projects/{project_id}/review",
        response_model=SuccessEnvelope[ProjectReviewMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def mark_reviewed(
        project_id: str,
        body: ReviewProjectRequest | None = _OPTIONAL_BODY,
    ) -> JSONResponse:
        """Mark a project reviewed and recalculate the next review when possible."""

        reviewed_at = (
            None if body is None or body.reviewed_at is None else body.reviewed_at.isoformat()
        )
        result = await resolved_service.mark_project_reviewed(
            project_id=project_id,
            reviewed_at=reviewed_at,
        )
        return _success_envelope(ProjectReviewMutationResult.model_validate(result))

    @app.get(
        "/v1/folders",
        response_model=SuccessEnvelope[list[FolderSummaryModel]],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def list_folders() -> JSONResponse:
        """List folders."""

        data = await resolved_service.list_folders()
        return _success_envelope([FolderSummaryModel.model_validate(item) for item in data])

    @app.get(
        "/v1/folders/tree",
        response_model=SuccessEnvelope[FolderTreeDataModel],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def get_folder_tree() -> JSONResponse:
        """Return the nested folder tree structure."""

        return _success_envelope(
            FolderTreeDataModel.model_validate(await resolved_service.get_folder_tree())
        )

    @app.get(
        "/v1/folders/{folder_id}",
        response_model=SuccessEnvelope[FolderSummaryModel],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def get_folder(folder_id: str) -> JSONResponse:
        """Return a single folder summary by id."""

        return _success_envelope(
            FolderSummaryModel.model_validate(
                await resolved_service.get_folder(folder_id=folder_id)
            )
        )

    @app.post(
        "/v1/folders",
        response_model=SuccessEnvelope[FolderMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
        status_code=201,
    )
    async def add_folder(body: AddFolderRequest = _REQUIRED_BODY) -> JSONResponse:
        """Create a folder by parent folder ID."""

        result = await resolved_service.add_folder(
            name=body.name,
            parent_folder_id=body.parent_folder_id,
        )
        return _success_envelope(FolderMutationResult.model_validate(result), status_code=201)

    @app.patch(
        "/v1/folders/{folder_id}",
        response_model=SuccessEnvelope[FolderMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def patch_folder(
        folder_id: str,
        body: UpdateFolderRequest = _REQUIRED_BODY,
    ) -> JSONResponse:
        """Update a folder by stable ID."""

        result = await resolved_service.update_folder(
            folder_id=folder_id,
            name=body.name,
            parent_folder_id=body.parent_folder_id,
            clear_parent=body.clear_parent,
        )
        return _success_envelope(FolderMutationResult.model_validate(result))

    @app.post(
        "/v1/folders/{folder_id}/drop",
        response_model=SuccessEnvelope[FolderMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def drop_folder(folder_id: str) -> JSONResponse:
        """Drop a folder by stable ID."""

        return _success_envelope(
            FolderMutationResult.model_validate(
                await resolved_service.drop_folder(folder_id=folder_id)
            )
        )

    @app.get(
        "/v1/tags",
        response_model=SuccessEnvelope[list[TagSummaryModel]],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def list_tags(include_hidden: bool = Query(False, alias="all")) -> JSONResponse:
        """List tags."""

        data = await resolved_service.list_tags(include_hidden=include_hidden)
        return _success_envelope([TagSummaryModel.model_validate(item) for item in data])

    @app.get(
        "/v1/tags/{tag_id}",
        response_model=SuccessEnvelope[TagSummaryModel],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def get_tag(tag_id: str) -> JSONResponse:
        """Return a single tag summary by id."""

        return _success_envelope(
            TagSummaryModel.model_validate(await resolved_service.get_tag(tag_id=tag_id))
        )

    @app.post(
        "/v1/tags",
        response_model=SuccessEnvelope[TagMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
        status_code=201,
    )
    async def add_tag(body: AddTagRequest = _REQUIRED_BODY) -> JSONResponse:
        """Create a tag by parent tag ID."""

        result = await resolved_service.add_tag(
            name=body.name,
            parent_tag_id=body.parent_tag_id,
            note=body.note,
        )
        return _success_envelope(TagMutationResult.model_validate(result), status_code=201)

    @app.patch(
        "/v1/tags/{tag_id}",
        response_model=SuccessEnvelope[TagMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def patch_tag(tag_id: str, body: UpdateTagRequest = _REQUIRED_BODY) -> JSONResponse:
        """Update a tag by stable ID."""

        result = await resolved_service.update_tag(
            tag_id=tag_id,
            name=body.name,
            parent_tag_id=body.parent_tag_id,
            clear_parent=body.clear_parent,
            note=body.note,
        )
        return _success_envelope(TagMutationResult.model_validate(result))

    @app.post(
        "/v1/tags/{tag_id}/drop",
        response_model=SuccessEnvelope[TagMutationResult],
        responses=_FASTAPI_ERROR_RESPONSES,
    )
    async def drop_tag(tag_id: str) -> JSONResponse:
        """Drop a tag by stable ID."""

        return _success_envelope(
            TagMutationResult.model_validate(await resolved_service.drop_tag(tag_id=tag_id))
        )

    app.add_middleware(
        RequestTimeoutMiddleware,
        timeout_seconds=request_timeout_seconds,
    )
    app.add_middleware(
        RateLimitedBearerAuthMiddleware,
        api_keys=resolved_api_keys,
        failure_limit=auth_failure_limit,
        failure_window_seconds=auth_failure_window_seconds,
    )
    app.add_middleware(JSONBodyGuardMiddleware)
    app.add_middleware(
        JSONTrustedHostMiddleware,
        allowed_hosts=resolved_allowed_hosts,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    return app


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the HTTPS API entrypoint."""

    return argparse.ArgumentParser(
        prog="of-http",
        description=(
            "Run the OmniFocus HTTPS API with TLS 1.3+, authenticated OpenAPI, "
            "and mandatory Bearer token authentication."
        ),
    )


async def _serve_uvicorn(
    config: HTTPServerConfig,
    *,
    service: StoreBackedApiService | None = None,
) -> None:
    """Serve the HTTPS API using a prebuilt TLS context."""

    app = create_app(
        api_keys=config.api_keys,
        allowed_hosts=config.allowed_hosts,
        service=service,
        enable_mcp=True,
    )
    uvicorn_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        access_log=False,
        log_level="warning",
        server_header=False,
        date_header=False,
        ssl_context_factory=lambda _uvicorn_config, _default_factory: build_ssl_context(config),
    )
    uvicorn_config.load()
    server = uvicorn.Server(uvicorn_config)
    await server.serve()


def run_server(config: HTTPServerConfig, *, service: StoreBackedApiService | None = None) -> None:
    """Run the HTTPS API server."""

    _emit_log(
        logging.INFO,
        "http.startup",
        {
            "host": config.host,
            "port": config.port,
            "tls_enabled": True,
            "tls_minimum_version": "TLSv1.3",
            "allowed_hosts": config.allowed_hosts,
            "api_keys_configured": len(config.api_keys),
        },
    )
    asyncio.run(_serve_uvicorn(config, service=service))


def main(argv: list[str] | None = None) -> None:
    """Run the HTTPS API entrypoint."""

    parser = build_arg_parser()
    parser.parse_args(argv)
    try:
        config = HTTPServerConfig.from_env()
        run_server(config)
    except OFError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
