"""Tests for :mod:`omnifocus.http_api`."""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import asyncio
import json
import logging
import ssl
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from starlette.testclient import TestClient

from omnifocus import __version__
from omnifocus.errors import OFError, OFHTTPError
from omnifocus.http_api import (
    _MAX_JSON_BODY_BYTES,
    HTTPServerConfig,
    JSONTrustedHostMiddleware,
    _AuthFailureLimiter,
    _serve_uvicorn,
    build_arg_parser,
    build_ssl_context,
    create_app,
    main,
    run_server,
)
from omnifocus.models import OFModel

_TEST_API_KEY = "test-api-key"
_TEST_ALLOWED_HOSTS = ("testserver", "127.0.0.1", "localhost")


class _FakeService:
    """Minimal async service double for HTTP transport tests."""

    def __init__(self) -> None:
        """Initialise canned responses and call tracking."""

        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def sync_now(self) -> dict[str, Any]:
        """Return a fake sync result."""

        self.calls.append(("sync_now", {}))
        return {"status": "synced", "tasks": 1, "projects": 2, "folders": 3, "tags": 4}

    async def list_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return fake task summaries."""

        self.calls.append(("list_tasks", kwargs))
        return [
            {
                "id": "t1",
                "name": "Task 1",
                "project": "Work",
                "project_id": "p1",
                "inbox": False,
                "flagged": True,
                "due": "2026-04-10T09:00:00",
                "start": "2026-04-09T09:00:00",
                "completed": None,
                "note": "Task note",
                "tag_ids": ["tag1"],
                "tag_names": ["@home"],
            }
        ]

    async def search_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return fake task search results."""

        self.calls.append(("search_tasks", kwargs))
        result = (await self.list_tasks())[0] | {"score": 1.0}
        return [result]

    async def get_task(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake task summary."""

        self.calls.append(("get_task", kwargs))
        return (await self.list_tasks())[0] | {"id": kwargs["task_id"]}

    async def add_task(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake task creation result."""

        self.calls.append(("add_task", kwargs))
        return {"status": "created", "task_id": "t1", "name": kwargs["name"]}

    async def update_task(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake task update result."""

        self.calls.append(("update_task", kwargs))
        return {"status": "updated", "task_id": kwargs["task_id"], "name": "Task 1"}

    async def complete_task(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake task completion result."""

        self.calls.append(("complete_task", kwargs))
        return {"status": "completed", "task_id": kwargs["task_id"]}

    async def drop_task(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake task drop result."""

        self.calls.append(("drop_task", kwargs))
        return {"status": "dropped", "task_id": kwargs["task_id"]}

    async def list_projects(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return fake project summaries."""

        self.calls.append(("list_projects", kwargs))
        return [
            {
                "id": "p1",
                "name": "Project 1",
                "folder_id": "f1",
                "status": "active",
                "singleton": False,
                "rank": 1,
                "added": "2026-04-01T09:00:00+00:00",
                "modified": "2026-04-02T09:00:00+00:00",
                "flagged": False,
                "due": None,
                "start": None,
                "note": "",
                "completed": None,
                "last_review": "2026-04-03T09:00:00+00:00",
                "next_review": "2026-04-10T09:00:00+00:00",
                "review_interval": "@1w",
                "tag_ids": ["tag1"],
                "repetition_rule": None,
                "repetition_method": None,
                "repetition_schedule_type": None,
                "repetition_anchor_date": None,
                "catch_up_automatically": False,
                "next_clone_identifier": 0,
                "due_date_alarm_policy": None,
                "defer_date_alarm_policy": None,
                "latest_time_to_start_alarm_policy": None,
                "planned_date_alarm_policy": None,
                "folder_name": "Work",
                "tag_names": ["@home"],
                "review_due": False,
                "review_basis": "next_review",
            }
        ]

    async def get_project(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake project summary."""

        self.calls.append(("get_project", kwargs))
        return (await self.list_projects())[0] | {"id": kwargs["project_id"]}

    async def add_project(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake project creation result."""

        self.calls.append(("add_project", kwargs))
        return {"status": "created", "project_id": "p1", "name": kwargs["name"]}

    async def update_project(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake project update result."""

        self.calls.append(("update_project", kwargs))
        return {"status": "updated", "project_id": kwargs["project_id"], "name": "Project 1"}

    async def complete_project(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake project completion result."""

        self.calls.append(("complete_project", kwargs))
        return {"status": "completed", "project_id": kwargs["project_id"]}

    async def list_projects_for_review(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return fake project review queue items."""

        self.calls.append(("list_projects_for_review", kwargs))
        return await self.list_projects()

    async def mark_project_reviewed(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake project review update result."""

        self.calls.append(("mark_project_reviewed", kwargs))
        return (await self.list_projects())[0] | {
            "id": kwargs["project_id"],
            "next_review_recalculated": True,
        }

    async def list_folders(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return fake folder summaries."""

        self.calls.append(("list_folders", kwargs))
        return [
            {
                "id": "f1",
                "name": "Folder 1",
                "parent_folder_id": None,
                "rank": 1,
                "added": "2026-04-01T09:00:00+00:00",
                "modified": "2026-04-02T09:00:00+00:00",
                "child_folder_ids": [],
                "project_ids": ["p1"],
            }
        ]

    async def get_folder(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake folder summary."""

        self.calls.append(("get_folder", kwargs))
        return (await self.list_folders())[0] | {"id": kwargs["folder_id"]}

    async def get_folder_tree(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake folder tree."""

        self.calls.append(("get_folder_tree", kwargs))
        folder = {
            "id": "f1",
            "name": "Folder 1",
            "parent_folder_id": None,
            "rank": 1,
            "added": "2026-04-01T09:00:00+00:00",
            "modified": "2026-04-02T09:00:00+00:00",
        }
        project = {
            "id": "p1",
            "name": "Project 1",
            "folder_id": "f1",
            "status": "active",
            "singleton": False,
            "rank": 1,
            "added": "2026-04-01T09:00:00+00:00",
            "modified": "2026-04-02T09:00:00+00:00",
            "flagged": False,
            "due": None,
            "start": None,
            "note": "",
            "completed": None,
            "last_review": None,
            "next_review": None,
            "review_interval": None,
            "tag_ids": [],
            "repetition_rule": None,
            "repetition_method": None,
            "repetition_schedule_type": None,
            "repetition_anchor_date": None,
            "catch_up_automatically": False,
            "next_clone_identifier": 0,
            "due_date_alarm_policy": None,
            "defer_date_alarm_policy": None,
            "latest_time_to_start_alarm_policy": None,
            "planned_date_alarm_policy": None,
        }
        return {
            "folders": [{"folder": folder, "children": [], "projects": [project]}],
            "no_folder_projects": [],
            "dangling_folder_projects": [],
        }

    async def add_folder(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake folder creation result."""

        self.calls.append(("add_folder", kwargs))
        return {"status": "created", "folder_id": "f1", "name": kwargs["name"]}

    async def update_folder(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake folder update result."""

        self.calls.append(("update_folder", kwargs))
        return {"status": "updated", "folder_id": kwargs["folder_id"], "name": "Folder 1"}

    async def drop_folder(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake folder drop result."""

        self.calls.append(("drop_folder", kwargs))
        return {"status": "dropped", "folder_id": kwargs["folder_id"]}

    async def list_tags(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return fake tag summaries."""

        self.calls.append(("list_tags", kwargs))
        return [
            {
                "id": "tag1",
                "name": "@home",
                "parent_tag_id": None,
                "rank": 1,
                "added": "2026-04-01T09:00:00+00:00",
                "modified": "2026-04-02T09:00:00+00:00",
                "note": "",
                "hidden": None,
                "parent_name": None,
                "child_tag_ids": [],
            }
        ]

    async def get_tag(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake tag summary."""

        self.calls.append(("get_tag", kwargs))
        return (await self.list_tags())[0] | {"id": kwargs["tag_id"]}

    async def add_tag(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake tag creation result."""

        self.calls.append(("add_tag", kwargs))
        return {"status": "created", "tag_id": "tag1", "name": kwargs["name"]}

    async def update_tag(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake tag update result."""

        self.calls.append(("update_tag", kwargs))
        return {"status": "updated", "tag_id": kwargs["tag_id"], "name": "@home"}

    async def drop_tag(self, **kwargs: Any) -> dict[str, Any]:
        """Return a fake tag drop result."""

        self.calls.append(("drop_tag", kwargs))
        return {"status": "dropped", "tag_id": kwargs["tag_id"]}


def _auth_headers(token: str | None = None, **extra: str) -> dict[str, str]:
    """Return a bearer-auth header set."""

    return {"Authorization": f"Bearer {token or _TEST_API_KEY}", **extra}


def _create_client(
    service: _FakeService,
    *,
    raise_server_exceptions: bool = True,
    **kwargs: Any,
) -> TestClient:
    """Return a test client with trusted hosts aligned to the test server."""

    return TestClient(
        create_app(
            api_key=_TEST_API_KEY,
            allowed_hosts=_TEST_ALLOWED_HOSTS,
            service=service,
            **kwargs,
        ),
        raise_server_exceptions=raise_server_exceptions,
    )


class TestMCP:
    def test_default_app_has_no_mcp_lifespan(self) -> None:
        client = _create_client(_FakeService())

        with client:
            response = client.get("/v1/openapi.json", headers=_auth_headers())

        assert response.status_code == 200

    def test_requires_bearer_authentication(self) -> None:
        client = _create_client(_FakeService(), enable_mcp=True)

        with client:
            response = client.post("/mcp/", json={})

        assert response.status_code == 401

    def test_initializes_lists_and_calls_mutation_tools(self) -> None:
        service = _FakeService()
        client = _create_client(service, enable_mcp=True)
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        }
        list_tools = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        add_task = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "add_task", "arguments": {"name": "Plan release"}},
        }
        headers = _auth_headers(Accept="application/json, text/event-stream")

        with (
            patch("omnifocus.mcp_server._service", return_value=service),
            patch("omnifocus.mcp_server._load_model", new=AsyncMock(return_value=OFModel())),
        ):
            with client:
                initialized = client.post("/mcp/", json=initialize, headers=headers)
                tools = client.post("/mcp/", json=list_tools, headers=headers)
                added = client.post("/mcp/", json=add_task, headers=headers)

        assert initialized.status_code == 200
        assert tools.status_code == 200
        assert added.status_code == 200
        names = {tool["name"] for tool in tools.json()["result"]["tools"]}
        assert "list_tasks" in names
        assert "add_task" in names
        assert service.calls[-1][0] == "add_task"
        assert service.calls[-1][1]["name"] == "Plan release"


def _write_self_signed_cert(tmp_path: Path) -> tuple[Path, Path]:
    """Create a temporary self-signed certificate and private key."""

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "PL"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "omnifocus-cli"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(days=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=30))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.DNSName("127.0.0.1")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


class TestHTTPServerConfig:
    """Tests for HTTP environment parsing."""

    def test_from_env_uses_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        monkeypatch.setenv("OF_HTTP_API_KEY", "secret-token")
        monkeypatch.setenv("OF_HTTP_TLS_CERT_FILE", str(cert_path))
        monkeypatch.setenv("OF_HTTP_TLS_KEY_FILE", str(key_path))

        config = HTTPServerConfig.from_env()

        assert config.host == "127.0.0.1"
        assert config.port == 8443
        assert config.api_key == "secret-token"
        assert config.api_keys == ("secret-token",)
        assert config.allowed_hosts == ("127.0.0.1", "localhost")

    def test_from_env_parses_allowed_hosts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        monkeypatch.setenv("OF_HTTP_API_KEY", "secret-token")
        monkeypatch.setenv("OF_HTTP_TLS_CERT_FILE", str(cert_path))
        monkeypatch.setenv("OF_HTTP_TLS_KEY_FILE", str(key_path))
        monkeypatch.setenv("OF_HTTP_ALLOWED_HOSTS", "127.0.0.1,localhost,api.internal")

        config = HTTPServerConfig.from_env()

        assert config.allowed_hosts == ("127.0.0.1", "localhost", "api.internal")

    def test_from_env_reads_api_key_from_secret_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        api_key_file = tmp_path / "api-key"
        api_key_file.write_text("secret-token\n", encoding="utf-8")
        monkeypatch.setenv("OF_HTTP_API_KEY_FILE", str(api_key_file))
        monkeypatch.setenv("OF_HTTP_TLS_CERT_FILE", str(cert_path))
        monkeypatch.setenv("OF_HTTP_TLS_KEY_FILE", str(key_path))

        config = HTTPServerConfig.from_env()

        assert config.api_key == "secret-token"

    def test_from_env_rejects_api_key_and_secret_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        api_key_file = tmp_path / "api-key"
        api_key_file.write_text("secret-token", encoding="utf-8")
        monkeypatch.setenv("OF_HTTP_API_KEY", "secret-token")
        monkeypatch.setenv("OF_HTTP_API_KEY_FILE", str(api_key_file))
        monkeypatch.setenv("OF_HTTP_TLS_CERT_FILE", str(cert_path))
        monkeypatch.setenv("OF_HTTP_TLS_KEY_FILE", str(key_path))

        with pytest.raises(OFError, match="OF_HTTP_API_KEY and OF_HTTP_API_KEY_FILE"):
            HTTPServerConfig.from_env()

    def test_from_env_rejects_empty_allowed_hosts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        monkeypatch.setenv("OF_HTTP_API_KEY", "secret-token")
        monkeypatch.setenv("OF_HTTP_TLS_CERT_FILE", str(cert_path))
        monkeypatch.setenv("OF_HTTP_TLS_KEY_FILE", str(key_path))
        monkeypatch.setenv("OF_HTTP_ALLOWED_HOSTS", ", ,")

        with pytest.raises(OFError, match="OF_HTTP_ALLOWED_HOSTS must contain at least one host"):
            HTTPServerConfig.from_env()

    def test_from_env_requires_mandatory_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OF_HTTP_API_KEY", raising=False)
        monkeypatch.delenv("OF_HTTP_TLS_CERT_FILE", raising=False)
        monkeypatch.delenv("OF_HTTP_TLS_KEY_FILE", raising=False)

        with pytest.raises(OFError, match="Missing required HTTP environment variables"):
            HTTPServerConfig.from_env()

    @pytest.mark.parametrize("port_value", ["99999", "nope"])
    def test_from_env_rejects_invalid_port(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        port_value: str,
    ) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        monkeypatch.setenv("OF_HTTP_API_KEY", "secret-token")
        monkeypatch.setenv("OF_HTTP_TLS_CERT_FILE", str(cert_path))
        monkeypatch.setenv("OF_HTTP_TLS_KEY_FILE", str(key_path))
        monkeypatch.setenv("OF_HTTP_PORT", port_value)

        with pytest.raises(OFError, match="Invalid OF_HTTP_PORT"):
            HTTPServerConfig.from_env()

    def test_from_env_rejects_missing_certificate_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _cert_path, key_path = _write_self_signed_cert(tmp_path)
        monkeypatch.setenv("OF_HTTP_API_KEY", "secret-token")
        monkeypatch.setenv("OF_HTTP_TLS_CERT_FILE", str(tmp_path / "missing-cert.pem"))
        monkeypatch.setenv("OF_HTTP_TLS_KEY_FILE", str(key_path))

        with pytest.raises(OFError, match="TLS certificate file not found"):
            HTTPServerConfig.from_env()

    def test_from_env_rejects_missing_key_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cert_path, _key_path = _write_self_signed_cert(tmp_path)
        monkeypatch.setenv("OF_HTTP_API_KEY", "secret-token")
        monkeypatch.setenv("OF_HTTP_TLS_CERT_FILE", str(cert_path))
        monkeypatch.setenv("OF_HTTP_TLS_KEY_FILE", str(tmp_path / "missing-key.pem"))

        with pytest.raises(OFError, match="TLS key file not found"):
            HTTPServerConfig.from_env()


class TestTLSHelpers:
    """Tests for TLS context creation."""

    def test_build_ssl_context_sets_minimum_tls_version(self, tmp_path: Path) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        config = HTTPServerConfig(
            host="127.0.0.1",
            port=8443,
            api_keys=("secret-token",),
            tls_cert_file=cert_path,
            tls_key_file=key_path,
        )

        context = build_ssl_context(config)

        assert isinstance(context, ssl.SSLContext)
        assert context.minimum_version == ssl.TLSVersion.TLSv1_3

    def test_build_ssl_context_requires_tls_1_3_support(self, tmp_path: Path) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        config = HTTPServerConfig(
            host="127.0.0.1",
            port=8443,
            api_keys=("secret-token",),
            tls_cert_file=cert_path,
            tls_key_file=key_path,
        )

        class _DummyTLSVersion:
            """Dummy TLSVersion type without TLSv1_3."""

            pass

        with patch("omnifocus.http_api.ssl.TLSVersion", _DummyTLSVersion):
            with pytest.raises(OFError, match="TLS 1.3 support is required"):
                build_ssl_context(config)

    def test_build_ssl_context_surfaces_tls_initialisation_failures(self, tmp_path: Path) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        config = HTTPServerConfig(
            host="127.0.0.1",
            port=8443,
            api_keys=("secret-token",),
            tls_cert_file=cert_path,
            tls_key_file=key_path,
        )
        fake_context = MagicMock()
        fake_context.load_cert_chain.side_effect = ssl.SSLError("bad cert")

        with patch("omnifocus.http_api.ssl.SSLContext", return_value=fake_context):
            with pytest.raises(OFError, match="Failed to initialise TLS 1.3 HTTPS listener"):
                build_ssl_context(config)


class TestHTTPAPIAuthAndDocs:
    """Tests for auth, docs, and response hardening."""

    def test_create_app_requires_at_least_one_api_key(self) -> None:
        with pytest.raises(OFError, match="At least one HTTP API key is required"):
            create_app(service=_FakeService())

    def test_health_requires_bearer_auth(self) -> None:
        client = _create_client(_FakeService())

        response = client.get("/v1/health")

        assert response.status_code == 401
        assert response.json()["ok"] is False
        assert response.headers["Strict-Transport-Security"].startswith("max-age=")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Pragma"] == "no-cache"

    def test_health_rejects_malformed_authorization_header(self) -> None:
        client = _create_client(_FakeService())

        response = client.get("/v1/health", headers={"Authorization": f"Token {_TEST_API_KEY}"})

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    def test_health_rejects_wrong_token_and_rate_limits_repeated_failures(self) -> None:
        client = _create_client(_FakeService(), auth_failure_limit=2)

        first = client.get("/v1/health", headers=_auth_headers("wrong-one"))
        second = client.get("/v1/health", headers=_auth_headers("wrong-two"))
        third = client.get("/v1/health", headers=_auth_headers("wrong-three"))

        assert first.status_code == 401
        assert second.status_code == 401
        assert third.status_code == 429
        assert third.json()["error"]["code"] == "rate_limited"

    def test_successful_auth_resets_failure_bucket(self) -> None:
        client = _create_client(_FakeService(), auth_failure_limit=2)

        assert client.get("/v1/health", headers=_auth_headers("wrong")).status_code == 401
        assert client.get("/v1/health", headers=_auth_headers()).status_code == 200
        assert client.get("/v1/health", headers=_auth_headers("wrong-again")).status_code == 401

    def test_health_returns_success_envelope_and_request_id(self) -> None:
        client = _create_client(_FakeService())

        response = client.get("/v1/health", headers=_auth_headers(**{"X-Request-ID": "req-123"}))

        assert response.status_code == 200
        assert response.json() == {
            "ok": True,
            "data": {"status": "ok", "version": __version__, "transport": "https"},
        }
        assert response.headers["X-Request-ID"] == "req-123"

    def test_openapi_requires_auth_and_describes_bearer_security(self) -> None:
        client = _create_client(_FakeService())

        unauthenticated = client.get("/v1/openapi.json")
        authenticated = client.get("/v1/openapi.json", headers=_auth_headers())
        cached = client.get("/v1/openapi.json", headers=_auth_headers())

        assert unauthenticated.status_code == 401
        assert authenticated.status_code == 200
        assert cached.status_code == 200
        spec = authenticated.json()
        assert spec["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
        assert spec["paths"]["/v1/health"]["get"]["security"] == [{"BearerAuth": []}]

    def test_invalid_host_is_rejected(self) -> None:
        client = _create_client(_FakeService())

        response = client.get(
            "/v1/health",
            headers={**_auth_headers(), "Host": "evil.example"},
        )

        assert response.status_code == 400
        assert response.json()["error"]["message"] == "Invalid Host header"
        assert response.headers["Strict-Transport-Security"].startswith("max-age=")

    def test_wildcard_host_rule_is_supported(self) -> None:
        client = TestClient(
            create_app(
                api_key=_TEST_API_KEY,
                allowed_hosts=("testserver", "*.example.internal"),
                service=_FakeService(),
            )
        )

        response = client.get(
            "/v1/health",
            headers={**_auth_headers(), "Host": "api.example.internal"},
        )

        assert response.status_code == 200


class TestHTTPAPIRoutes:
    """Tests for route wiring and request parsing."""

    def test_task_endpoints_pass_expected_arguments(self) -> None:
        service = _FakeService()
        client = _create_client(service)
        headers = {**_auth_headers(), "Content-Type": "application/json"}

        list_response = client.get(
            "/v1/tasks?inbox=true&today=false&flagged=true&due=true&project=Work&tag=@home&tag_id=tag1&limit=7",
            headers=_auth_headers(),
        )
        search_response = client.get("/v1/tasks/search?query=milk&limit=3", headers=_auth_headers())
        get_response = client.get("/v1/tasks/t1", headers=_auth_headers())
        add_response = client.post(
            "/v1/tasks",
            headers=headers,
            json={
                "name": "New task",
                "project_id": "p1",
                "due": "today",
                "flagged": True,
                "note": "Use pytest",
            },
        )
        patch_response = client.patch(
            "/v1/tasks/t1",
            headers=headers,
            json={
                "name": "Renamed task",
                "project_id": "p1",
                "clear_project": False,
                "inbox": False,
                "due": "2026-04-08",
                "defer": "2026-04-07",
                "flagged": True,
                "note": "Updated",
                "estimate": 30,
                "tag_ids": ["tag1", "tag2"],
                "clear_tags": False,
                "dropped": False,
            },
        )
        complete_response = client.post("/v1/tasks/t1/complete", headers=_auth_headers())
        drop_response = client.post("/v1/tasks/t1/drop", headers=_auth_headers())

        assert list_response.status_code == 200
        assert search_response.status_code == 200
        assert get_response.status_code == 200
        assert add_response.status_code == 201
        assert patch_response.status_code == 200
        assert complete_response.status_code == 200
        assert drop_response.status_code == 200
        assert service.calls[0] == (
            "list_tasks",
            {
                "inbox": True,
                "today": False,
                "flagged": True,
                "due": True,
                "no_due": False,
                "no_defer": False,
                "available": False,
                "overdue": False,
                "has_project": False,
                "project": "Work",
                "project_status": "all",
                "tag": "@home",
                "tag_id": "tag1",
                "limit": 7,
            },
        )
        called_methods = [name for name, _kwargs in service.calls]
        assert "add_task" in called_methods
        assert "update_task" in called_methods

    def test_project_folder_and_tag_endpoints_are_reachable(self) -> None:
        service = _FakeService()
        client = _create_client(service)
        headers = {**_auth_headers(), "Content-Type": "application/json"}

        responses = [
            client.get(
                "/v1/projects?status=active&tag=@home&tag_id=tag1&limit=5",
                headers=_auth_headers(),
            ),
            client.get("/v1/projects/p1", headers=_auth_headers()),
            client.post(
                "/v1/projects",
                headers=headers,
                json={"name": "Project 1", "folder_id": "f1", "status": "active"},
            ),
            client.patch(
                "/v1/projects/p1",
                headers=headers,
                json={
                    "name": "Project 2",
                    "folder_id": "f1",
                    "clear_folder": False,
                    "status": "inactive",
                    "tag_ids": ["tag1"],
                    "clear_tags": False,
                },
            ),
            client.post("/v1/projects/p1/complete", headers=_auth_headers()),
            client.get("/v1/projects/review?due_only=false&limit=6", headers=_auth_headers()),
            client.post(
                "/v1/projects/p1/review",
                headers=headers,
                json={"reviewed_at": "2026-04-06T12:00:00+00:00"},
            ),
            client.get("/v1/folders", headers=_auth_headers()),
            client.get("/v1/folders/tree", headers=_auth_headers()),
            client.get("/v1/folders/f1", headers=_auth_headers()),
            client.post(
                "/v1/folders",
                headers=headers,
                json={"name": "Folder 1", "parent_folder_id": "root"},
            ),
            client.patch(
                "/v1/folders/f1",
                headers=headers,
                json={"name": "Renamed folder", "parent_folder_id": "root", "clear_parent": False},
            ),
            client.post("/v1/folders/f1/drop", headers=_auth_headers()),
            client.get("/v1/tags?all=true", headers=_auth_headers()),
            client.get("/v1/tags/tag1", headers=_auth_headers()),
            client.post(
                "/v1/tags",
                headers=headers,
                json={"name": "@home", "parent_tag_id": "root"},
            ),
            client.patch(
                "/v1/tags/tag1",
                headers=headers,
                json={
                    "name": "@desk",
                    "parent_tag_id": "root",
                    "clear_parent": False,
                    "note": "Updated",
                },
            ),
            client.post("/v1/tags/tag1/drop", headers=_auth_headers()),
        ]

        assert all(response.status_code in {200, 201} for response in responses)
        called_methods = [name for name, _kwargs in service.calls]
        assert "list_projects" in called_methods
        assert "list_projects_for_review" in called_methods
        assert "list_folders" in called_methods
        assert "list_tags" in called_methods


class TestHTTPAPIErrors:
    """Tests for validation and exception mapping."""

    def test_rejects_missing_query_for_task_search(self) -> None:
        client = _create_client(_FakeService())

        response = client.get("/v1/tasks/search", headers=_auth_headers())

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_rejects_invalid_query_boolean(self) -> None:
        client = _create_client(_FakeService())

        response = client.get("/v1/tasks?inbox=maybe", headers=_auth_headers())

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_rejects_invalid_query_integer(self) -> None:
        client = _create_client(_FakeService())

        response = client.get("/v1/tasks?limit=nope", headers=_auth_headers())

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_rejects_non_json_request_body(self) -> None:
        client = _create_client(_FakeService())

        response = client.post("/v1/tags", headers=_auth_headers(), content="name=@home")

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "bad_request"

    def test_rejects_malformed_json_request_body(self) -> None:
        client = _create_client(_FakeService())

        response = client.post(
            "/v1/tags",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            content="{",
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "bad_request"

    def test_rejects_non_object_json_body_with_validation_error(self) -> None:
        client = _create_client(_FakeService())

        response = client.post(
            "/v1/tags",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json=["not", "an", "object"],
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_rejects_missing_required_name(self) -> None:
        client = _create_client(_FakeService())

        response = client.post(
            "/v1/tags",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json={},
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_rejects_oversized_json_body(self) -> None:
        client = _create_client(_FakeService())

        response = client.post(
            "/v1/tags",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            content="x" * (_MAX_JSON_BODY_BYTES + 1),
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "bad_request"

    def test_rejects_invalid_array_shape(self) -> None:
        client = _create_client(_FakeService())

        response = client.patch(
            "/v1/projects/p1",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json={"tag_ids": "tag1"},
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_allows_null_optional_string_field(self) -> None:
        service = _FakeService()
        client = _create_client(service)

        response = client.patch(
            "/v1/tags/tag1",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json={"note": None},
        )

        assert response.status_code == 200
        assert service.calls[-1] == (
            "update_tag",
            {
                "tag_id": "tag1",
                "name": None,
                "parent_tag_id": None,
                "clear_parent": False,
                "note": None,
            },
        )

    def test_allows_patch_project_without_tag_ids(self) -> None:
        service = _FakeService()
        client = _create_client(service)

        response = client.patch(
            "/v1/projects/p1",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json={"name": "Renamed"},
        )

        assert response.status_code == 200
        assert service.calls[-1] == (
            "update_project",
            {
                "project_id": "p1",
                "name": "Renamed",
                "folder_id": None,
                "clear_folder": False,
                "due": None,
                "defer": None,
                "flagged": None,
                "note": None,
                "status": None,
                "tag_ids": None,
                "clear_tags": False,
            },
        )

    def test_allows_patch_task_with_empty_estimate(self) -> None:
        service = _FakeService()
        client = _create_client(service)

        response = client.patch(
            "/v1/tasks/t1",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json={"estimate": ""},
        )

        assert response.status_code == 200
        assert service.calls[-1][1]["estimate"] == ""

    def test_rejects_invalid_review_timestamp(self) -> None:
        client = _create_client(_FakeService())

        response = client.post(
            "/v1/projects/p1/review",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json={"reviewed_at": "not-a-timestamp"},
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_maps_service_of_http_error(self) -> None:
        service = _FakeService()
        service.get_task = AsyncMock(
            side_effect=OFHTTPError("Task not found: missing", status_code=404, code="not_found")
        )
        client = _create_client(service)

        response = client.get("/v1/tasks/missing", headers=_auth_headers())

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_maps_generic_of_error_without_leaking_details(self) -> None:
        service = _FakeService()
        service.sync_now = AsyncMock(side_effect=OFError("sensitive internal failure"))
        client = _create_client(service)

        response = client.post("/v1/sync", headers=_auth_headers())

        assert response.status_code == 500
        assert response.json()["error"]["message"] == "Internal server error"

    def test_maps_unexpected_exception_without_trace(self) -> None:
        service = _FakeService()
        service.list_tags = AsyncMock(side_effect=RuntimeError("boom"))
        client = _create_client(service, raise_server_exceptions=False)

        response = client.get("/v1/tags", headers=_auth_headers())

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"

    def test_request_timeout_returns_504(self) -> None:
        service = _FakeService()

        async def delayed_list_tags(**kwargs: Any) -> list[dict[str, Any]]:
            service.calls.append(("list_tags", kwargs))
            await asyncio.sleep(0.05)
            return []

        service.list_tags = delayed_list_tags  # type: ignore[assignment]  # Inject timeout stub.
        client = _create_client(service, request_timeout_seconds=0.001)

        response = client.get("/v1/tags", headers=_auth_headers())

        assert response.status_code == 504
        assert response.json()["error"]["code"] == "timeout"


class TestHTTPLogging:
    """Tests for structured request and audit logging."""

    def test_request_logs_are_structured_and_trace_aware(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="omnifocus.http_api")
        client = _create_client(_FakeService())

        response = client.get(
            "/v1/health",
            headers=_auth_headers(
                traceparent="00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
            ),
        )

        assert response.status_code == 200
        payloads = [json.loads(record.message) for record in caplog.records]
        request_log = next(payload for payload in payloads if payload["body"] == "http.request")
        assert request_log["attributes"]["http.route"] == "/v1/health"
        assert request_log["attributes"]["trace.id"] == "0123456789abcdef0123456789abcdef"
        assert "Authorization" not in json.dumps(payloads)

    def test_invalid_traceparent_shapes_do_not_emit_trace_ids(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="omnifocus.http_api")
        client = _create_client(_FakeService())

        invalid_headers = [
            "00-too-few",
            "00-only-three-parts",
            "01-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
            "00-short-0123456789abcdef-01",
        ]
        for traceparent in invalid_headers:
            response = client.get("/v1/health", headers=_auth_headers(traceparent=traceparent))
            assert response.status_code == 200

        payloads = [json.loads(record.message) for record in caplog.records]
        request_logs = [payload for payload in payloads if payload["body"] == "http.request"]
        assert all(log["attributes"]["trace.id"] is None for log in request_logs)

    def test_auth_failure_emits_security_log_without_token(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="omnifocus.http_api")
        client = _create_client(_FakeService())

        response = client.get("/v1/health", headers=_auth_headers("bad-token"))

        assert response.status_code == 401
        payloads = [json.loads(record.message) for record in caplog.records]
        audit_log = next(
            payload for payload in payloads if payload["body"] == "security.auth.failure"
        )
        assert audit_log["attributes"]["failure_reason"] == "invalid_token"
        assert "bad-token" not in json.dumps(payloads)

    def test_stale_auth_failures_age_out(self) -> None:
        limiter = _AuthFailureLimiter(limit=1, window_seconds=1.0)

        assert limiter.record_failure("127.0.0.1", now_monotonic=1.0) == 1
        assert limiter.is_limited("127.0.0.1", now_monotonic=1.0) is True
        assert limiter.is_limited("127.0.0.1", now_monotonic=3.0) is False
        assert limiter.record_failure("127.0.0.1", now_monotonic=3.0) == 1

    @pytest.mark.asyncio
    async def test_trusted_host_middleware_passthroughs_non_http_scope(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def passthrough_app(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": f"{scope['type']}.startup.complete"})

        passthrough = JSONTrustedHostMiddleware(
            passthrough_app,
            allowed_hosts=_TEST_ALLOWED_HOSTS,
        )

        async def receive() -> dict[str, Any]:
            return {"type": "lifespan.startup"}

        async def send(message: dict[str, Any]) -> None:
            sent_messages.append(message)

        await passthrough({"type": "lifespan"}, receive, send)

        assert sent_messages == [{"type": "lifespan.startup.complete"}]


class TestHTTPServerRuntime:
    """Tests for runtime launch integration."""

    @pytest.mark.asyncio
    async def test_serve_uvicorn_uses_tls_1_3_and_hides_server_headers(
        self, tmp_path: Path
    ) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        config = HTTPServerConfig(
            host="127.0.0.1",
            port=8443,
            api_keys=("secret-token",),
            tls_cert_file=cert_path,
            tls_key_file=key_path,
            allowed_hosts=("localhost",),
        )
        fake_uvicorn_config = MagicMock()
        fake_server = MagicMock()
        fake_server.serve = AsyncMock(return_value=None)

        with patch(
            "omnifocus.http_api.uvicorn.Config",
            return_value=fake_uvicorn_config,
        ) as mock_config:
            with patch(
                "omnifocus.http_api.uvicorn.Server",
                return_value=fake_server,
            ) as mock_server:
                await _serve_uvicorn(config, service=_FakeService())

        mock_config.assert_called_once()
        config_kwargs = mock_config.call_args.kwargs
        assert config_kwargs["server_header"] is False
        assert config_kwargs["date_header"] is False
        ssl_context = config_kwargs["ssl_context_factory"](MagicMock(), MagicMock())
        assert isinstance(ssl_context, ssl.SSLContext)
        assert ssl_context.minimum_version == ssl.TLSVersion.TLSv1_3
        mock_server.assert_called_once_with(fake_uvicorn_config)
        fake_server.serve.assert_awaited_once_with()

    def test_run_server_invokes_asyncio_run(self, tmp_path: Path) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        config = HTTPServerConfig(
            host="127.0.0.1",
            port=8443,
            api_keys=("secret-token",),
            tls_cert_file=cert_path,
            tls_key_file=key_path,
        )

        with patch("omnifocus.http_api.asyncio.run") as mock_run:
            run_server(config)

        mock_run.assert_called_once()
        coroutine = mock_run.call_args.args[0]
        assert asyncio.iscoroutine(coroutine)
        coroutine.close()

    def test_main_loads_config_and_runs_server(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cert_path, key_path = _write_self_signed_cert(tmp_path)
        monkeypatch.setenv("OF_HTTP_API_KEY", "secret-token")
        monkeypatch.setenv("OF_HTTP_TLS_CERT_FILE", str(cert_path))
        monkeypatch.setenv("OF_HTTP_TLS_KEY_FILE", str(key_path))

        with patch("omnifocus.http_api.run_server") as mock_run_server:
            main([])

        mock_run_server.assert_called_once()

    def test_main_exits_cleanly_on_configuration_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.delenv("OF_HTTP_API_KEY", raising=False)
        monkeypatch.delenv("OF_HTTP_TLS_CERT_FILE", raising=False)
        monkeypatch.delenv("OF_HTTP_TLS_KEY_FILE", raising=False)

        with pytest.raises(SystemExit, match="1"):
            main([])

        captured = capsys.readouterr()
        assert "Missing required HTTP environment variables" in captured.err

    def test_build_arg_parser_accepts_help_only_surface(self) -> None:
        parser = build_arg_parser()

        assert parser.prog == "of-http"
        with pytest.raises(SystemExit, match="0"):
            parser.parse_args(["--help"])
