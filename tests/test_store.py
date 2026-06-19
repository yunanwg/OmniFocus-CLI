"""Tests for :mod:`omnifocus.store`."""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import dataclasses
import json
import pickle
import zipfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omnifocus.errors import OFEncryptionError, OFError, OFWebDAVError
from omnifocus.models import Folder, OFModel, Project, Tag, Task
from omnifocus.store import (
    OFocusStore,
    _atomic_write_bytes,
    _default_cache_dir,
    _WriterState,
)
from omnifocus.sync.graph import (
    current_frontier_tail_ids,
    current_tail_id,
    maximal_tail_ids,
    reachable_delta_tail_ids,
    tail_depends_on,
    tail_reachable_from_baseline,
    transaction_filenames_for_frontier,
)
from omnifocus.sync.webdav import WebDAVClient
from omnifocus.writer import WritePlan
from tests.conftest import make_zip

_EMPTY_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2"/>'
)
_PERSONAL_BRANCH_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <folder id="personal-folder">
    <added>2026-03-24T10:00:00.000Z</added>
    <name>Personal</name>
    <rank>1</rank>
    <modified>2026-03-24T10:00:00.000Z</modified>
  </folder>
</omnifocus>
"""
_WORK_BRANCH_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<omnifocus xmlns="http://www.omnigroup.com/namespace/OmniFocus/v2">
  <folder id="work-folder">
    <added>2026-03-24T10:00:00.000Z</added>
    <name>Work</name>
    <rank>1</rank>
    <modified>2026-03-24T10:00:00.000Z</modified>
  </folder>
</omnifocus>
"""
NOW = datetime(2026, 3, 22, 12, 0, 0, tzinfo=UTC)


def _make_store(
    tmp_path: Path,
    filenames: list[str] | None = None,
    baseline_bytes: bytes | None = None,
    passphrase: str | None = None,
    encrypted_plist: bytes | None = None,
) -> tuple[OFocusStore, AsyncMock]:
    """Build an ``OFocusStore`` with a mocked ``WebDAVClient``."""
    client = AsyncMock(spec=WebDAVClient)
    entries = filenames or ["00000000000000=base+tail.zip"]
    client.list_entries = AsyncMock(return_value=entries)
    client.list_bundle = AsyncMock(return_value=[name for name in entries if name.endswith(".zip")])

    baseline = baseline_bytes or make_zip(_EMPTY_XML)

    async def _get_file(name: str) -> bytes:
        if name == "encrypted":
            if encrypted_plist is not None:
                return encrypted_plist
            raise OFWebDAVError("Not found", status_code=404)
        return baseline

    client.get_file = AsyncMock(side_effect=_get_file)
    store = OFocusStore(client=client, passphrase=passphrase, cache_dir=tmp_path)
    return store, client


def _read_contents_xml(data: bytes) -> str:
    """Return the XML payload from a transaction ZIP."""
    with zipfile.ZipFile(BytesIO(data)) as archive:
        return archive.read("contents.xml").decode("utf-8")


def _make_task() -> Task:
    """Build a stable test task."""
    return Task(
        id="t1",
        name="Write tests",
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
        estimated_minutes=None,
        added=NOW,
        modified=NOW,
    )


def _make_project() -> Project:
    """Build a stable test project."""
    return Project(
        id="p1",
        name="Engineering",
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
        last_review=datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC),
        next_review=datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC),
        review_interval="@1m",
    )


def _make_folder() -> Folder:
    """Build a stable test folder."""
    return Folder(
        id="f1",
        name="Engineering",
        parent_folder_id=None,
        rank=100,
        added=NOW,
        modified=NOW,
    )


def _make_writer_state() -> _WriterState:
    """Build a stable writer state."""
    return _WriterState(
        client_id="client123",
        host_id="ED325E58-F612-4653-BD34-7006A7D6DD52",
        device_name="air.local",
        registration_date=NOW,
        tail_identifiers=("tail123",),
        hardware_cpu_count="10",
        hardware_cpu_type="16777228,0",
        hardware_cpu_type_name="arm64",
        hardware_model="Mac16,12",
        os_version="25D2128",
        os_version_number="26.3.1",
        encrypted=False,
        bundle_fingerprint=None,
    )


class TestFromEnv:
    def test_missing_webdav_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("OF_WEBDAV_URL", "OF_WEBDAV_USER", "OF_WEBDAV_PASS"):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(OFWebDAVError):
            OFocusStore.from_env()

    def test_passphrase_falls_back_to_webdav_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OF_WEBDAV_URL", "https://dav.example.com/of/")
        monkeypatch.setenv("OF_WEBDAV_USER", "u")
        monkeypatch.setenv("OF_WEBDAV_PASS", "linked_pass")
        monkeypatch.delenv("OF_ENCRYPTION_PASSPHRASE", raising=False)
        store = OFocusStore.from_env()
        assert store._passphrase == "linked_pass"  # noqa: S105

    def test_passphrase_falls_back_to_url_embedded_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OF_WEBDAV_URL", "https://u:url_pass@dav.example.com/of/")
        monkeypatch.delenv("OF_WEBDAV_USER", raising=False)
        monkeypatch.delenv("OF_WEBDAV_PASS", raising=False)
        monkeypatch.delenv("OF_ENCRYPTION_PASSPHRASE", raising=False)
        store = OFocusStore.from_env()
        assert store._passphrase == "url_pass"  # noqa: S105

    def test_passphrase_set_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OF_WEBDAV_URL", "https://dav.example.com/of/")
        monkeypatch.setenv("OF_WEBDAV_USER", "u")
        monkeypatch.setenv("OF_WEBDAV_PASS", "p")
        monkeypatch.setenv("OF_ENCRYPTION_PASSPHRASE", "secret")
        store = OFocusStore.from_env()
        assert store._passphrase == "secret"  # noqa: S105

    def test_cache_dir_defaults_to_repo_local_dot_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OF_WEBDAV_URL", "https://dav.example.com/of/")
        monkeypatch.setenv("OF_WEBDAV_USER", "u")
        monkeypatch.setenv("OF_WEBDAV_PASS", "p")
        monkeypatch.delenv("OF_CACHE_DIR", raising=False)
        store = OFocusStore.from_env()
        assert store._cache_dir.name == ".of-cache"

    def test_cache_dir_respects_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OF_WEBDAV_URL", "https://dav.example.com/of/")
        monkeypatch.setenv("OF_WEBDAV_USER", "u")
        monkeypatch.setenv("OF_WEBDAV_PASS", "p")
        monkeypatch.setenv("OF_CACHE_DIR", "/custom-cache")
        store = OFocusStore.from_env()
        assert store._cache_dir == Path("/custom-cache")


class TestDefaultCacheDir:
    def test_prefers_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OF_CACHE_DIR", "/custom-cache")
        assert _default_cache_dir() == Path("/custom-cache")

    def test_defaults_to_repo_local_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OF_CACHE_DIR", raising=False)
        assert _default_cache_dir().name == ".of-cache"

    def test_falls_back_to_tmp_when_repo_root_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OF_CACHE_DIR", raising=False)
        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert _default_cache_dir() == Path("/tmp/of-cache")  # noqa: S108


class TestLoad:
    @pytest.mark.asyncio
    async def test_load_returns_model(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        model = await store.load()
        assert isinstance(model, OFModel)

    @pytest.mark.asyncio
    async def test_load_calls_list_and_get(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.load()
        client.list_entries.assert_called_once()
        client.get_file.assert_called_once_with("00000000000000=base+tail.zip")

    @pytest.mark.asyncio
    async def test_load_with_transactions(self, tmp_path: Path) -> None:
        store, client = _make_store(
            tmp_path,
            filenames=["00000000000000=base+tail0.zip", "20260322154011=tail0+tail1.zip"],
        )
        client.get_file = AsyncMock(return_value=make_zip(_EMPTY_XML))
        model = await store.load()
        assert isinstance(model, OFModel)
        assert client.get_file.call_count == 2

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.load()
        assert client.list_entries.call_count == 1
        await store.load(force_refresh=True)
        assert client.list_entries.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_cache_on_second_load(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.load()
        await store.load()
        assert client.list_entries.call_count == 2
        assert client.get_file.call_count == 1

    @pytest.mark.asyncio
    async def test_changed_transaction_listing_bypasses_cache(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.load()
        client.list_entries.return_value = [
            "00000000000000=base+tail0.zip",
            "20260322154011=tail0+tail1.zip",
        ]
        await store.load()
        assert client.list_entries.call_count == 2
        assert client.get_file.call_count == 3

    @pytest.mark.asyncio
    async def test_changed_baseline_listing_bypasses_cache(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.load()
        client.list_entries.return_value = ["00000000000000=base-v2+tail1.zip"]
        await store.load()
        assert client.list_entries.call_count == 2
        assert client.get_file.call_count == 2

    @pytest.mark.asyncio
    async def test_load_applies_all_reachable_sink_tails(self, tmp_path: Path) -> None:
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260324100000=tail0+head-personal.zip",
                "20260324100001=tail0+head-work.zip",
                "20260324100002=reader.client",
            ],
        )

        async def get_file(name: str) -> bytes:
            if name == "00000000000000=base+tail0.zip":
                return make_zip(_EMPTY_XML)
            if name == "20260324100000=tail0+head-personal.zip":
                return make_zip(_PERSONAL_BRANCH_XML)
            if name == "20260324100001=tail0+head-work.zip":
                return make_zip(_WORK_BRANCH_XML)
            if name == "20260324100002=reader.client":
                return (
                    b'<?xml version="1.0" encoding="UTF-8"?>'
                    b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                    b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
                    b'<plist version="1.0"><dict>'
                    b"<key>clientIdentifier</key><string>reader</string>"
                    b"<key>hostID</key><string>host-a</string>"
                    b"<key>name</key><string>a.local</string>"
                    b"<key>registrationDate</key><date>2026-03-24T10:00:02Z</date>"
                    b"<key>lastSyncDate</key><date>2026-03-24T10:00:02Z</date>"
                    b"<key>tailIdentifiers</key><array><string>head-work</string></array>"
                    b"</dict></plist>"
                )
            raise AssertionError(f"Unexpected fetch: {name}")

        client.get_file = AsyncMock(side_effect=get_file)

        model = await store.load()

        assert "work-folder" in model.folders
        assert "personal-folder" in model.folders
        fetched = [call.args[0] for call in client.get_file.await_args_list]
        assert fetched == [
            "20260324100002=reader.client",
            "00000000000000=base+tail0.zip",
            "20260324100000=tail0+head-personal.zip",
            "20260324100001=tail0+head-work.zip",
        ]

    @pytest.mark.asyncio
    async def test_load_raises_when_current_tail_chain_cannot_be_resolved(
        self, tmp_path: Path
    ) -> None:
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260324100002=reader.client",
            ],
        )

        async def get_file(name: str) -> bytes:
            if name == "20260324100002=reader.client":
                return (
                    b'<?xml version="1.0" encoding="UTF-8"?>'
                    b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                    b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
                    b'<plist version="1.0"><dict>'
                    b"<key>clientIdentifier</key><string>reader</string>"
                    b"<key>hostID</key><string>host-a</string>"
                    b"<key>name</key><string>a.local</string>"
                    b"<key>registrationDate</key><date>2026-03-24T10:00:02Z</date>"
                    b"<key>lastSyncDate</key><date>2026-03-24T10:00:02Z</date>"
                    b"<key>tailIdentifiers</key><array><string>missing-head</string></array>"
                    b"</dict></plist>"
                )
            return make_zip(_EMPTY_XML)

        client.get_file = AsyncMock(side_effect=get_file)

        with pytest.raises(OFError, match="Could not resolve the current OmniFocus delta DAG"):
            await store.load()

    def test_transaction_chain_resolution_rejects_cycles(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=base+tail0.zip",
                "20260324100000=head-a+head-b.zip",
                "20260324100001=head-b+head-a.zip",
            ]
        )

        with pytest.raises(OFError, match="Detected a cycle while resolving delta DAG"):
            reachable_delta_tail_ids(state, ("head-a",))

    def test_transaction_filenames_for_empty_frontier_returns_empty(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(["00000000000000=base+tail0.zip"])
        assert transaction_filenames_for_frontier(state, ()) == []


class TestDeepDeltaChain:
    """A long, near-linear delta chain must resolve without blowing the Python
    recursion stack.

    Regression (2026-06-18): a 526-deep live OmniFocus chain raised
    ``RecursionError: maximum recursion depth exceeded`` on every read. The DAG
    walks in ``sync/graph.py`` recursed once per delta, and the
    ``all(visit(p) for p in parents)`` generator expression added a second frame
    per level, so each delta cost two frames — ~500 deltas exceeded the default
    1000-frame limit. ``depth`` here is well past that threshold.
    """

    def test_deep_linear_chain_resolves_without_recursion_error(self) -> None:
        from datetime import UTC, datetime, timedelta

        from omnifocus.sync.protocol import build_bundle_state

        depth = 2000
        base_ts = datetime(2026, 1, 1, tzinfo=UTC)
        filenames = ["00000000000000=base+t0.zip"]
        for i in range(depth):
            stamp = (base_ts + timedelta(seconds=i)).strftime("%Y%m%d%H%M%S")
            filenames.append(f"{stamp}=t{i}+t{i + 1}.zip")
        state = build_bundle_state(filenames)
        head = f"t{depth}"

        # Frontier discovery: tail_reachable_from_baseline + maximal_tail_ids -> tail_depends_on
        assert current_frontier_tail_ids(state, {}) == (head,)
        assert current_tail_id(state, {}) == head

        # Direct reachability / dependency walks down the whole chain.
        assert tail_reachable_from_baseline(state, head) is True
        assert tail_depends_on(state, head, "t0") is True

        # Delta selection + topological ordering (parent-before-child).
        ordered = transaction_filenames_for_frontier(state, (head,))
        assert len(ordered) == depth
        assert ordered[0].endswith("=t0+t1.zip")
        assert ordered[-1].endswith(f"=t{depth - 1}+t{depth}.zip")


class TestBundleFingerprint:
    """The cache fingerprint must ignore ``.client`` churn (every device writes a
    new timestamped client-state file on each sync) when the model is a pure
    function of the baseline + deltas, yet still track client state in the
    degenerate case where the frontier is derived from client-advertised tails.
    """

    def test_ignores_client_churn_when_a_delta_frontier_exists(self) -> None:
        from omnifocus.store import _bundle_fingerprint

        deltas = ["00000000000000=base+t0.zip", "20260324100000=t0+t1.zip"]
        # Same data, only a client re-synced (new timestamped .client file).
        fp_a = _bundle_fingerprint([*deltas, "20260324100002=mac.client"])
        fp_b = _bundle_fingerprint([*deltas, "20260324100099=mac.client"])
        assert fp_a == fp_b

    def test_tracks_clients_without_a_delta_frontier(self) -> None:
        from omnifocus.store import _bundle_fingerprint

        # Baseline only → no delta-derived frontier, so the frontier falls back to
        # client-advertised tails and the model genuinely depends on client state.
        fp_a = _bundle_fingerprint(["00000000000000=base+t0.zip", "20260324100002=a.client"])
        fp_b = _bundle_fingerprint(["00000000000000=base+t0.zip", "20260324100099=b.client"])
        assert fp_a != fp_b

    def test_changes_when_a_delta_is_added(self) -> None:
        from omnifocus.store import _bundle_fingerprint

        deltas = ["00000000000000=base+t0.zip", "20260324100000=t0+t1.zip"]
        assert _bundle_fingerprint(deltas) != _bundle_fingerprint(
            [*deltas, "20260324100001=t1+t2.zip"]
        )


class TestCache:
    @pytest.mark.asyncio
    async def test_cache_file_created(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        await store.load()
        assert (tmp_path / "of_model.pkl").exists()

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.load()
        store.invalidate_cache()
        assert not (tmp_path / "of_model.pkl").exists()
        await store.load()
        assert client.list_entries.call_count == 2

    @pytest.mark.asyncio
    async def test_invalidate_cache_noop_when_no_cache(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        store.invalidate_cache()

    @pytest.mark.asyncio
    async def test_cache_contains_valid_pickle(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        model = await store.load()
        cached = pickle.loads((tmp_path / "of_model.pkl").read_bytes())  # noqa: S301
        assert cached.model.parsed_at == model.parsed_at
        assert cached.bundle_fingerprint == ("00000000000000=base+tail.zip", (), ())

    @pytest.mark.asyncio
    async def test_corrupt_cache_is_ignored(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        (tmp_path / "of_model.pkl").write_bytes(b"not a pickle")
        model = await store.load()
        assert isinstance(model, OFModel)
        assert client.get_file.call_count == 1

    @pytest.mark.asyncio
    async def test_legacy_model_only_cache_is_treated_as_stale(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        (tmp_path / "of_model.pkl").write_bytes(pickle.dumps(OFModel()))
        model = await store.load()
        assert isinstance(model, OFModel)
        assert client.list_entries.call_count == 1
        assert client.get_file.call_count == 1

    @pytest.mark.asyncio
    async def test_unexpected_cache_payload_is_treated_as_stale(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        (tmp_path / "of_model.pkl").write_bytes(pickle.dumps("unexpected"))
        model = await store.load()
        assert isinstance(model, OFModel)
        assert client.list_entries.call_count == 1
        assert client.get_file.call_count == 1


class TestEncryption:
    @pytest.mark.asyncio
    async def test_encrypted_data_decrypted_before_parse(self, tmp_path: Path) -> None:
        from omnifocus.crypto.encryption import create_encrypted_bundle

        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, encrypted_file = create_encrypted_bundle(plaintext, "passphrase123")
        store, _ = _make_store(
            tmp_path,
            baseline_bytes=encrypted_file,
            passphrase="passphrase123",  # noqa: S106
            encrypted_plist=encrypted_plist,
        )
        model = await store.load()
        assert isinstance(model, OFModel)

    @pytest.mark.asyncio
    async def test_encrypted_without_passphrase_raises(self, tmp_path: Path) -> None:
        from omnifocus.crypto.encryption import create_encrypted_bundle

        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, encrypted_file = create_encrypted_bundle(plaintext, "passphrase123")
        store, _ = _make_store(
            tmp_path,
            baseline_bytes=encrypted_file,
            passphrase=None,
            encrypted_plist=encrypted_plist,
        )
        with pytest.raises(OFEncryptionError, match="no passphrase is available"):
            await store.load()

    @pytest.mark.asyncio
    async def test_wrong_passphrase_raises(self, tmp_path: Path) -> None:
        from omnifocus.crypto.encryption import create_encrypted_bundle

        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, encrypted_file = create_encrypted_bundle(plaintext, "correct-passphrase")
        store, _ = _make_store(
            tmp_path,
            baseline_bytes=encrypted_file,
            passphrase="wrong-passphrase",  # noqa: S106
            encrypted_plist=encrypted_plist,
        )
        with pytest.raises(OFEncryptionError, match="HMAC verification failed"):
            await store.load()

    @pytest.mark.asyncio
    async def test_unencrypted_data_passes_through(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path, passphrase=None)
        model = await store.load()
        assert isinstance(model, OFModel)

    @pytest.mark.asyncio
    async def test_plaintext_transaction_in_encrypted_bundle(self, tmp_path: Path) -> None:
        from omnifocus.crypto.encryption import create_encrypted_bundle

        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, encrypted_baseline = create_encrypted_bundle(plaintext, "pw")
        plain_tx = make_zip(_EMPTY_XML)
        store, client = _make_store(
            tmp_path,
            filenames=["00000000000000=base+tail0.zip", "20260322154011=tail1+tail0.zip"],
            passphrase="pw",  # noqa: S106
            encrypted_plist=encrypted_plist,
        )

        async def get_file(name: str) -> bytes:
            if name == "encrypted":
                return encrypted_plist
            if name == "00000000000000=base+tail0.zip":
                return encrypted_baseline
            return plain_tx

        client.get_file = AsyncMock(side_effect=get_file)
        model = await store.load()
        assert isinstance(model, OFModel)

    @pytest.mark.asyncio
    async def test_unknown_key_slot_raises(self, tmp_path: Path) -> None:
        from omnifocus.crypto.encryption import create_encrypted_bundle, encrypt_file

        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, _ = create_encrypted_bundle(plaintext, "pw", slot_id=1)
        bad_file = encrypt_file(plaintext, b"A" * 16, b"B" * 16, key_id=99)
        store, _ = _make_store(
            tmp_path,
            baseline_bytes=bad_file,
            passphrase="pw",  # noqa: S106
            encrypted_plist=encrypted_plist,
        )
        with pytest.raises(OFEncryptionError, match="Key slot 99 not found"):
            await store.load()


class TestSyncStatus:
    @pytest.mark.asyncio
    async def test_status_no_cache(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        status = await store.sync_status()
        assert status == {
            "last_synced": None,
            "cached": False,
            "cache_age_seconds": None,
            "cache_valid": False,
            "bundle_state_version": 2,
            "registered_client": False,
            "tail_identifiers": [],
            "advertised_tail_identifiers": [],
            "client_id": None,
            "host_id": None,
            "current_tail_identifier": None,
        }

    @pytest.mark.asyncio
    async def test_status_after_load(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.load()
        status = await store.sync_status()
        assert status["cached"] is True
        assert status["last_synced"] is not None
        assert isinstance(status["cache_age_seconds"], float)
        assert status["cache_valid"] is True
        assert status["bundle_state_version"] == 2
        assert status["registered_client"] is False
        assert status["tail_identifiers"] == []
        assert status["advertised_tail_identifiers"] == []
        assert status["client_id"] is None
        assert status["host_id"] is None
        assert status["current_tail_identifier"] == "tail"
        assert client.list_entries.call_count == 2

    @pytest.mark.asyncio
    async def test_status_marks_cache_invalid_when_remote_listing_changes(
        self, tmp_path: Path
    ) -> None:
        store, client = _make_store(tmp_path)
        await store.load()
        client.list_entries.return_value = [
            "00000000000000=base+tail0.zip",
            "20260322154011=tail1+tail0.zip",
        ]
        status = await store.sync_status()
        assert status["cached"] is True
        assert status["cache_valid"] is False

    @pytest.mark.asyncio
    async def test_status_marks_legacy_cache_invalid_without_hitting_remote(
        self, tmp_path: Path
    ) -> None:
        store, client = _make_store(tmp_path)
        (tmp_path / "of_model.pkl").write_bytes(pickle.dumps(OFModel()))
        status = await store.sync_status()
        assert status["cached"] is True
        assert status["cache_valid"] is False
        assert client.list_entries.call_count == 0

    @pytest.mark.asyncio
    async def test_status_exposes_registered_client_tail_identifiers(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        store._save_writer_state(_make_writer_state())  # noqa: SLF001
        status = await store.sync_status()
        assert status["registered_client"] is True
        assert status["tail_identifiers"] == ["tail123"]
        assert status["client_id"] == "client123"
        assert status["host_id"] == "ED325E58-F612-4653-BD34-7006A7D6DD52"

    @pytest.mark.asyncio
    async def test_status_current_tail_uses_remote_client_documents(self, tmp_path: Path) -> None:
        client_doc = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>clientIdentifier</key><string>client123</string>
<key>hostID</key><string>host-123</string>
<key>name</key><string>air.local</string>
<key>registrationDate</key><date>2026-03-22T12:00:00Z</date>
<key>lastSyncDate</key><date>2026-03-22T12:00:00Z</date>
<key>tailIdentifiers</key><array><string>tail123</string></array>
</dict></plist>"""
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail123.zip",
                "20260322154011=client123.client",
            ],
        )

        async def get_file(name: str) -> bytes:
            if name.endswith(".client"):
                return client_doc
            return make_zip(_EMPTY_XML)

        client.get_file = AsyncMock(side_effect=get_file)
        await store.load()
        status = await store.sync_status()
        assert status["current_tail_identifier"] == "tail123"

    @pytest.mark.asyncio
    async def test_status_current_tail_swallow_remote_client_parse_errors(
        self, tmp_path: Path
    ) -> None:
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail123.zip",
                "20260322154011=client123.client",
            ],
        )

        valid_client_doc = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>clientIdentifier</key><string>client123</string>
<key>hostID</key><string>host-123</string>
<key>name</key><string>air.local</string>
<key>registrationDate</key><date>2026-03-22T12:00:00Z</date>
<key>lastSyncDate</key><date>2026-03-22T12:00:00Z</date>
<key>tailIdentifiers</key><array><string>tail123</string></array>
</dict></plist>"""

        async def initial_get_file(name: str) -> bytes:
            if name.endswith(".client"):
                return valid_client_doc
            return make_zip(_EMPTY_XML)

        client.get_file = AsyncMock(side_effect=initial_get_file)
        await store.load()

        async def get_file(name: str) -> bytes:
            if name.endswith(".client"):
                return b"not plist"
            return make_zip(_EMPTY_XML)

        client.get_file = AsyncMock(side_effect=get_file)
        status = await store.sync_status()
        assert status["current_tail_identifier"] is None


class TestWritePath:
    @pytest.mark.asyncio
    async def test_add_task_creates_writer_state_and_identity_files(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        result = await store.add_task(name="New task")
        assert result["status"] == "created"
        saved = json.loads((tmp_path / "writer_state.json").read_text())
        identity = json.loads((tmp_path / "writer_identity.json").read_text())
        assert isinstance(saved["client_id"], str)
        assert isinstance(saved["host_id"], str)
        assert len(saved["host_id"]) == 36
        assert saved["device_name"]
        assert isinstance(saved["registration_date"], str)
        assert isinstance(saved["tail_identifiers"], list)
        assert len(saved["tail_identifiers"]) == 1
        assert saved["hardware_model"] is None
        assert saved["encrypted"] is False
        assert identity["client_id"] == saved["client_id"]
        assert identity["host_id"] == saved["host_id"]
        assert identity["device_name"] == saved["device_name"]
        assert identity["registration_date"] == saved["registration_date"]

    @pytest.mark.asyncio
    async def test_reuses_same_client_id_across_writes(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        await store.add_task(name="First task")
        first_state = json.loads((tmp_path / "writer_state.json").read_text())
        await store.add_task(name="Second task")
        second_state = json.loads((tmp_path / "writer_state.json").read_text())
        assert second_state["client_id"] == first_state["client_id"]

    @pytest.mark.asyncio
    async def test_uses_explicit_env_identity_without_saved_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OF_CLIENT_ID", "STATICCLIENT01")
        monkeypatch.setenv("OF_DEVICE_NAME", "air.local")
        monkeypatch.setenv("OF_DEVICE_HOST_ID", "ED325E58-F612-4653-BD34-7006A7D6DD52")

        store, _ = _make_store(tmp_path)
        await store.add_task(name="First task")
        first_state = json.loads((tmp_path / "writer_state.json").read_text())

        fresh_tmp = tmp_path / "fresh"
        fresh_tmp.mkdir()
        fresh_store, _ = _make_store(fresh_tmp)
        await fresh_store.add_task(name="Second task")
        second_state = json.loads((fresh_tmp / "writer_state.json").read_text())

        assert first_state["client_id"] == "STATICCLIENT01"
        assert second_state["client_id"] == "STATICCLIENT01"
        assert first_state["device_name"] == "air.local"
        assert second_state["device_name"] == "air.local"
        assert first_state["host_id"] == "ED325E58-F612-4653-BD34-7006A7D6DD52"
        assert second_state["host_id"] == "ED325E58-F612-4653-BD34-7006A7D6DD52"

    @pytest.mark.asyncio
    async def test_reuses_identity_when_writer_state_is_missing(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        await store.add_task(name="First task")
        first_state = json.loads((tmp_path / "writer_state.json").read_text())

        (tmp_path / "writer_state.json").unlink()

        fresh_store, _ = _make_store(tmp_path)
        await fresh_store.add_task(name="Second task")
        second_state = json.loads((tmp_path / "writer_state.json").read_text())

        assert second_state["client_id"] == first_state["client_id"]
        assert second_state["host_id"] == first_state["host_id"]
        assert second_state["device_name"] == first_state["device_name"]
        assert second_state["registration_date"] == first_state["registration_date"]

    @pytest.mark.asyncio
    async def test_refreshes_tail_when_remote_listing_changes(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.add_task(name="First task")
        client.list_entries.return_value = [
            "00000000000000=base+tail0.zip",
            "20260322154011=tail0+remote123.zip",
        ]
        await store.add_task(name="Second task")
        zip_uploads = [
            call.args[0]
            for call in client.put_file.await_args_list
            if call.args[0].endswith(".zip")
        ]
        delta_upload = zip_uploads[-2]
        assert delta_upload.startswith("202")
        assert "=remote123+" in delta_upload
        payload = json.loads((tmp_path / "writer_state.json").read_text())
        assert payload["tail_identifiers"] == ["remote123"]

    @pytest.mark.asyncio
    async def test_no_remote_deltas_uses_baseline_tail_as_parent(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.add_task(name="First task")
        zip_uploads = [
            call.args[0]
            for call in client.put_file.await_args_list
            if call.args[0].endswith(".zip")
        ]
        delta_upload = zip_uploads[0]
        assert "=tail+" in delta_upload
        payload = json.loads((tmp_path / "writer_state.json").read_text())
        assert payload["tail_identifiers"] == ["tail"]

    @pytest.mark.asyncio
    async def test_plaintext_bundle_uploads_plain_zip(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.add_task(name="Plain task")
        uploaded = client.put_file.await_args_list[0].args[1]
        assert uploaded[:2] == b"PK"
        assert "<name/>" in _read_contents_xml(uploaded)

    @pytest.mark.asyncio
    async def test_write_also_uploads_client_state(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.add_task(name="Track state")
        zip_uploads = [
            call for call in client.put_file.await_args_list if call.args[0].endswith(".zip")
        ]
        client_uploads = [
            call for call in client.put_file.await_args_list if call.args[0].endswith(".client")
        ]
        delta_name = zip_uploads[-1].args[0]
        client_name = client_uploads[-1].args[0]
        client_payload = client_uploads[-1].args[1]
        assert delta_name.endswith(".zip")
        assert client_name.endswith(".client")
        assert b"<plist" in client_payload

    @pytest.mark.asyncio
    async def test_add_task_uploads_multiple_deltas(self, tmp_path: Path) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("OF_CHAIN_SHAPE", "linear")
        store, client = _make_store(tmp_path)
        await store.add_task(name="Track state")
        monkeypatch.undo()
        zip_uploads = [
            call for call in client.put_file.await_args_list if call.args[0].endswith(".zip")
        ]
        client_uploads = [
            call for call in client.put_file.await_args_list if call.args[0].endswith(".client")
        ]
        assert len(zip_uploads) >= 2
        assert len(client_uploads) == len(zip_uploads)
        first_name = zip_uploads[0].args[0]
        second_name = zip_uploads[1].args[0]
        first_tail = first_name.split("+", 1)[1].removesuffix(".zip")
        second_parent = second_name.split("=", 1)[1].split("+", 1)[0]
        assert second_parent == first_tail

    @pytest.mark.asyncio
    async def test_add_task_second_delta_updates_name(self, tmp_path: Path) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("OF_CHAIN_SHAPE", "linear")
        store, client = _make_store(tmp_path)
        await store.add_task(name="Track state")
        monkeypatch.undo()
        zip_uploads = [
            call for call in client.put_file.await_args_list if call.args[0].endswith(".zip")
        ]
        second_xml = _read_contents_xml(zip_uploads[1].args[1])
        assert 'op="update"' in second_xml
        assert "<name>Track state</name>" in second_xml

    @pytest.mark.asyncio
    async def test_add_task_with_due_is_visible_after_force_refresh(self, tmp_path: Path) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("OF_CHAIN_SHAPE", "linear")
        store, client = _make_store(tmp_path)
        due_dt = datetime(2026, 4, 10, 19, 0, 0)
        result = await store.add_task(name="Visible after refresh", due_dt=due_dt)
        monkeypatch.undo()

        zip_uploads = [
            call for call in client.put_file.await_args_list if call.args[0].endswith(".zip")
        ]
        uploaded_payloads = {call.args[0]: call.args[1] for call in zip_uploads}
        client.list_entries.return_value = [
            "00000000000000=base+tail.zip",
            *uploaded_payloads.keys(),
        ]

        async def _get_uploaded_or_baseline(name: str) -> bytes:
            if name in uploaded_payloads:
                return uploaded_payloads[name]
            return make_zip(_EMPTY_XML)

        client.get_file = AsyncMock(side_effect=_get_uploaded_or_baseline)
        model = await store.load(force_refresh=True)
        task = model.tasks[result["task_id"]]
        assert task.name == "Visible after refresh"
        assert task.due == due_dt

    @pytest.mark.asyncio
    async def test_add_task_chain_then_client_uploads_client_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OF_WRITE_STRATEGY", "chain_then_client")
        store, client = _make_store(tmp_path)
        await store.add_task(name="Track state")
        client_uploads = [
            call for call in client.put_file.await_args_list if call.args[0].endswith(".client")
        ]
        assert len(client_uploads) == 1

    @pytest.mark.asyncio
    async def test_add_task_client_after_each_delta_uploads_client_each_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OF_WRITE_STRATEGY", "client_after_each_delta")
        store, client = _make_store(tmp_path)
        await store.add_task(name="Track state")
        zip_uploads = [
            call for call in client.put_file.await_args_list if call.args[0].endswith(".zip")
        ]
        client_uploads = [
            call for call in client.put_file.await_args_list if call.args[0].endswith(".client")
        ]
        assert len(client_uploads) == len(zip_uploads)

    @pytest.mark.asyncio
    async def test_default_chain_shape_is_app_rebase(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.add_task(name="Track state")
        zip_uploads = [
            call.args[0]
            for call in client.put_file.await_args_list
            if call.args[0].endswith(".zip")
        ]
        first = zip_uploads[0].split("=", 1)[1].removesuffix(".zip")
        second = zip_uploads[1].split("=", 1)[1].removesuffix(".zip")
        first_head, first_parent = first.split("+", 1)
        second_head, second_parent = second.split("+", 1)
        assert first_head == "tail"
        assert second_head == first_parent
        assert second_parent != first_head

    @pytest.mark.asyncio
    async def test_invalid_write_strategy_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OF_WRITE_STRATEGY", "bogus")
        store, _ = _make_store(tmp_path)
        with pytest.raises(OFError, match="Invalid OF_WRITE_STRATEGY"):
            await store.add_task(name="Track state")

    @pytest.mark.asyncio
    async def test_invalid_chain_shape_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OF_CHAIN_SHAPE", "bogus")
        store, _ = _make_store(tmp_path)
        with pytest.raises(OFError, match="Invalid OF_CHAIN_SHAPE"):
            await store.add_task(name="Track state")

    @pytest.mark.asyncio
    async def test_app_rebase_chain_shape_changes_parent_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OF_CHAIN_SHAPE", "app_rebase")
        store, client = _make_store(tmp_path)
        await store.add_task(name="Track state")
        zip_uploads = [
            call.args[0]
            for call in client.put_file.await_args_list
            if call.args[0].endswith(".zip")
        ]
        first = zip_uploads[0].split("=", 1)[1].removesuffix(".zip")
        second = zip_uploads[1].split("=", 1)[1].removesuffix(".zip")
        first_head, first_parent = first.split("+", 1)
        second_head, second_parent = second.split("+", 1)
        assert first_head == "tail"
        assert second_head == first_parent
        assert second_parent != first_head

    @pytest.mark.asyncio
    async def test_initial_writer_state_metadata_comes_from_env_not_remote_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OF_DEVICE_HARDWARE_MODEL", "Mac16,12")
        monkeypatch.setenv("OF_DEVICE_OS_VERSION", "25D2128")
        monkeypatch.setenv("OF_DEVICE_OS_VERSION_NUMBER", "26.3.1")
        store, _ = _make_store(tmp_path)
        await store.add_task(name="Track state")
        payload = json.loads((tmp_path / "writer_state.json").read_text())
        assert payload["hardware_model"] == "Mac16,12"
        assert payload["os_version"] == "25D2128"
        assert payload["os_version_number"] == "26.3.1"

    @pytest.mark.asyncio
    async def test_encrypted_bundle_uploads_ciphertext(self, tmp_path: Path) -> None:
        from omnifocus.crypto.discovery import MAGIC
        from omnifocus.crypto.encryption import create_encrypted_bundle

        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, encrypted_baseline = create_encrypted_bundle(plaintext, "secret")
        store, client = _make_store(
            tmp_path,
            baseline_bytes=encrypted_baseline,
            passphrase="secret",  # noqa: S106
            encrypted_plist=encrypted_plist,
        )
        await store.add_task(name="Encrypted task")
        uploaded = client.put_file.await_args_list[0].args[1]
        assert uploaded.startswith(MAGIC)

    @pytest.mark.asyncio
    async def test_encrypted_bundle_without_passphrase_raises_on_write(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.crypto.encryption import create_encrypted_bundle

        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, encrypted_baseline = create_encrypted_bundle(plaintext, "secret")
        store, _ = _make_store(
            tmp_path,
            baseline_bytes=encrypted_baseline,
            passphrase=None,
            encrypted_plist=encrypted_plist,
        )
        with pytest.raises(OFEncryptionError, match="no passphrase is available"):
            await store.add_task(name="Encrypted task")

    @pytest.mark.asyncio
    async def test_missing_writable_key_slot_raises(self, tmp_path: Path) -> None:
        from omnifocus.crypto.encryption import create_encrypted_bundle

        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, encrypted_baseline = create_encrypted_bundle(plaintext, "secret")
        store, _ = _make_store(
            tmp_path,
            baseline_bytes=encrypted_baseline,
            passphrase="secret",  # noqa: S106
            encrypted_plist=encrypted_plist,
        )
        store._load_writable_key_slot = lambda _: (_ for _ in ()).throw(  # type: ignore[method-assign]
            OFEncryptionError("No active writable encryption key slot found in bundle")
        )
        with pytest.raises(OFEncryptionError, match="No active writable encryption key slot"):
            await store.add_task(name="Encrypted task")

    @pytest.mark.asyncio
    async def test_successful_upload_invalidates_cache(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        await store.load()
        assert (tmp_path / "of_model.pkl").exists()
        await store.add_task(name="Invalidate cache")
        assert not (tmp_path / "of_model.pkl").exists()

    @pytest.mark.asyncio
    async def test_writer_state_updates_after_upload(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store.add_task(name="Track state")
        payload = json.loads((tmp_path / "writer_state.json").read_text())
        assert payload["tail_identifiers"] == ["tail"]

    @pytest.mark.asyncio
    async def test_unreachable_remote_client_tail_falls_back_to_latest_delta(
        self, tmp_path: Path
    ) -> None:
        store, _ = _make_store(tmp_path)
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        state = build_bundle_state(
            [
                "00000000000000=base+tail0.zip",
                "20260322154011=tail0+delta999.zip",
                "20260322154012=appclient.client",
            ]
        )
        remote_clients = {
            "appclient": ClientStateDocument(
                client_identifier="appclient",
                tail_identifiers=("client-tail",),
                registration_date=NOW,
                last_sync_date=NOW,
                name="air.local",
                host_id="ED325E58-F612-4653-BD34-7006A7D6DD52",
            )
        }
        assert current_tail_id(state, remote_clients) == "delta999"

    def test_current_tail_prefers_latest_reachable_delta_over_stale_client_tail(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=base+tail0.zip",
                "20260322154011=tail0+head1.zip",
                "20260322154012=head1+head2.zip",
                "20260322154013=appclient.client",
            ]
        )
        remote_clients = {
            "appclient": ClientStateDocument(
                client_identifier="appclient",
                tail_identifiers=("head1",),
                registration_date=NOW,
                last_sync_date=NOW,
                name="air.local",
                host_id="ED325E58-F612-4653-BD34-7006A7D6DD52",
            )
        }
        assert current_tail_id(state, remote_clients) == "head2"

    def test_current_tail_skips_client_without_tail_and_uses_next_latest(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+baseline-tail.zip",
                "20260322154011=clientA.client",
                "20260322154012=clientB.client",
            ]
        )
        remote_clients = {
            "clientA": ClientStateDocument(
                client_identifier="clientA",
                host_id="hostA",
                name="a.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("tail-a",),
            ),
            "clientB": ClientStateDocument(
                client_identifier="clientB",
                host_id="hostB",
                name="b.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=(),
            ),
        }
        assert current_tail_id(state, remote_clients) == "tail-a"

    def test_current_frontier_skips_missing_remote_client_document(self, tmp_path: Path) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+baseline-tail.zip",
                "20260322154011=missing.client",
                "20260322154012=present.client",
            ]
        )
        remote_clients = {
            "present": ClientStateDocument(
                client_identifier="present",
                host_id="hostB",
                name="b.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("tail-b",),
            ),
        }
        assert current_frontier_tail_ids(state, remote_clients) == ("tail-b",)

    def test_current_frontier_falls_back_to_latest_client_tail_tuple(self, tmp_path: Path) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot.zip",
                "20260322154011=clientA.client",
            ]
        )
        remote_clients = {
            "clientA": ClientStateDocument(
                client_identifier="clientA",
                host_id="hostA",
                name="a.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("tail-a", "tail-b"),
            )
        }
        assert current_frontier_tail_ids(state, remote_clients) == (
            "tail-a",
            "tail-b",
        )

    @pytest.mark.asyncio
    async def test_empty_add_task_plan_raises(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        writer_state = _make_writer_state()
        from omnifocus.writer import AddTaskPlan

        with pytest.raises(OFError, match="Task creation plan produced no deltas"):
            await store._upload_task_plan(  # type: ignore[attr-defined]
                AddTaskPlan(task_id="task-1", deltas=()),
                encrypted_plist=None,
                key_slot=None,
                writer_state=writer_state,
            )

    @pytest.mark.asyncio
    async def test_fetch_latest_deltas_returns_newest_files(self, tmp_path: Path) -> None:
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=tail0+head1.zip",
                "20260322154012=head1+head2.zip",
            ],
        )
        client.get_file = AsyncMock(side_effect=lambda name: name.encode("utf-8"))
        files = await store.fetch_latest_deltas(count=2)
        assert [name for name, _ in files] == [
            "20260322154011=tail0+head1.zip",
            "20260322154012=head1+head2.zip",
        ]

    @pytest.mark.asyncio
    async def test_fetch_file_returns_named_payload(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        client.get_file = AsyncMock(return_value=b"payload")
        assert await store.fetch_file("encrypted") == b"payload"

    @pytest.mark.asyncio
    async def test_fetch_latest_deltas_for_client_uses_client_tail(self, tmp_path: Path) -> None:
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=tail0+head1.zip",
                "20260322154012=client-a.client",
            ],
        )

        async def get_file(name: str) -> bytes:
            if name.endswith(".client"):
                return (
                    b'<?xml version="1.0" encoding="UTF-8"?>'
                    b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                    b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
                    b'<plist version="1.0"><dict>'
                    b"<key>clientIdentifier</key><string>client-a</string>"
                    b"<key>hostID</key><string>host-a</string>"
                    b"<key>name</key><string>a.local</string>"
                    b"<key>registrationDate</key><date>2026-03-22T12:00:00Z</date>"
                    b"<key>lastSyncDate</key><date>2026-03-22T12:00:00Z</date>"
                    b"<key>tailIdentifiers</key><array><string>head1</string></array>"
                    b"</dict></plist>"
                )
            return name.encode("utf-8")

        client.get_file = AsyncMock(side_effect=get_file)
        files = await store.fetch_latest_deltas(client_id="client-a")
        assert [name for name, _ in files] == ["20260322154011=tail0+head1.zip"]

    @pytest.mark.asyncio
    async def test_fetch_latest_deltas_unknown_client_raises(self, tmp_path: Path) -> None:
        store, _ = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=tail0+head1.zip",
            ],
        )
        with pytest.raises(OFError, match="No client state found"):
            await store.fetch_latest_deltas(client_id="client-a")

    @pytest.mark.asyncio
    async def test_fetch_latest_deltas_client_without_tail_raises(self, tmp_path: Path) -> None:
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=client-a.client",
            ],
        )

        async def get_file(name: str) -> bytes:
            return (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
                b'<plist version="1.0"><dict>'
                b"<key>clientIdentifier</key><string>client-a</string>"
                b"<key>hostID</key><string>host-a</string>"
                b"<key>name</key><string>a.local</string>"
                b"<key>registrationDate</key><date>2026-03-22T12:00:00Z</date>"
                b"<key>lastSyncDate</key><date>2026-03-22T12:00:00Z</date>"
                b"<key>tailIdentifiers</key><array></array>"
                b"</dict></plist>"
            )

        client.get_file = AsyncMock(side_effect=get_file)
        with pytest.raises(OFError, match="has no advertised tail identifier"):
            await store.fetch_latest_deltas(client_id="client-a")

    @pytest.mark.asyncio
    async def test_fetch_latest_deltas_client_without_matching_delta_raises(
        self, tmp_path: Path
    ) -> None:
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=client-a.client",
            ],
        )

        async def get_file(name: str) -> bytes:
            return (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
                b'<plist version="1.0"><dict>'
                b"<key>clientIdentifier</key><string>client-a</string>"
                b"<key>hostID</key><string>host-a</string>"
                b"<key>name</key><string>a.local</string>"
                b"<key>registrationDate</key><date>2026-03-22T12:00:00Z</date>"
                b"<key>lastSyncDate</key><date>2026-03-22T12:00:00Z</date>"
                b"<key>tailIdentifiers</key><array><string>missing-head</string></array>"
                b"</dict></plist>"
            )

        client.get_file = AsyncMock(side_effect=get_file)
        with pytest.raises(OFError, match="No delta ZIP found"):
            await store.fetch_latest_deltas(client_id="client-a")

    @pytest.mark.asyncio
    async def test_fetch_latest_client_returns_requested_client(self, tmp_path: Path) -> None:
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=client-a.client",
                "20260322154012=client-b.client",
            ],
        )
        client.get_file = AsyncMock(side_effect=lambda name: name.encode("utf-8"))
        name, payload = await store.fetch_latest_client(client_id="client-b")
        assert name == "20260322154012=client-b.client"
        assert payload == b"20260322154012=client-b.client"

    @pytest.mark.asyncio
    async def test_fetch_latest_client_without_any_clients_raises(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path, filenames=["00000000000000=base+tail0.zip"])
        with pytest.raises(OFError, match="No client state files found"):
            await store.fetch_latest_client()

    @pytest.mark.asyncio
    async def test_fetch_latest_client_unknown_client_raises(self, tmp_path: Path) -> None:
        store, _ = _make_store(
            tmp_path,
            filenames=["00000000000000=base+tail0.zip", "20260322154012=client-b.client"],
        )
        with pytest.raises(OFError, match="No client state found for client"):
            await store.fetch_latest_client(client_id="client-a")

    @pytest.mark.asyncio
    async def test_decrypt_latest_delta_plaintext_without_encrypted_plist(
        self, tmp_path: Path
    ) -> None:
        xml = '<?xml version="1.0" encoding="UTF-8"?><omnifocus xmlns="x"/>'
        zip_bytes = make_zip(xml)
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=tail0+head1.zip",
            ],
        )

        async def get_file(name: str) -> bytes:
            if name == "encrypted":
                raise OFWebDAVError("Not found", status_code=404)
            return zip_bytes

        client.get_file = AsyncMock(side_effect=get_file)
        filename, contents_xml = await store.decrypt_latest_delta()
        assert filename == "20260322154011=tail0+head1.zip"
        assert contents_xml == xml

    @pytest.mark.asyncio
    async def test_decrypt_latest_delta_with_no_deltas_raises(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path, filenames=["00000000000000=base+tail0.zip"])
        with pytest.raises(OFError, match="No delta ZIPs found"):
            await store.decrypt_latest_delta()

    @pytest.mark.asyncio
    async def test_decrypt_latest_delta_re_raises_non_404_encrypted_fetch_error(
        self, tmp_path: Path
    ) -> None:
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=tail0+head1.zip",
            ],
        )

        async def get_file(name: str) -> bytes:
            if name == "encrypted":
                raise OFWebDAVError("Forbidden", status_code=403)
            return make_zip(_EMPTY_XML)

        client.get_file = AsyncMock(side_effect=get_file)
        with pytest.raises(OFWebDAVError, match="Forbidden"):
            await store.decrypt_latest_delta()

    @pytest.mark.asyncio
    async def test_update_task_uploads_task_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        result = await store.update_task(_make_task())
        assert result == {"status": "updated", "task_id": "t1", "name": "Write tests"}
        uploaded = client.put_file.await_args_list[0].args[1]
        assert "Write tests" in _read_contents_xml(uploaded)

    @pytest.mark.asyncio
    async def test_update_task_uploads_project_parent_and_tags(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        task = dataclasses.replace(
            _make_task(),
            parent_task_id="p2",
            project_id="p2",
            inbox=False,
            tag_ids=("tag1", "tag2"),
        )
        await store.update_task(task)
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert '<task idref="p2"/>' in xml
        assert '<context idref="tag1"/>' in xml
        assert '<context idref="tag2"/>' in xml

    @pytest.mark.asyncio
    async def test_complete_task_uploads_task_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        result = await store.complete_task(_make_task())
        assert result == {"status": "completed", "task_id": "t1", "name": "Write tests"}
        uploaded = client.put_file.await_args_list[0].args[1]
        assert "<completed>" in _read_contents_xml(uploaded)

    @pytest.mark.asyncio
    async def test_complete_project_uploads_project_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        result = await store.complete_project(_make_project())
        assert result == {"status": "completed", "project_id": "p1", "name": "Engineering"}
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert "<status>done</status>" in xml
        assert "<name>Engineering</name>" in xml

    @pytest.mark.asyncio
    async def test_drop_task_uploads_hidden_task_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        result = await store.drop_task(_make_task())
        assert result == {"status": "dropped", "task_id": "t1", "name": "Write tests"}
        uploaded = client.put_file.await_args_list[0].args[1]
        assert "<hidden>" in _read_contents_xml(uploaded)

    @pytest.mark.asyncio
    async def test_drop_project_uploads_dropped_project_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        result = await store.drop_project(_make_project())
        assert result == {"status": "dropped", "project_id": "p1", "name": "Engineering"}
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert "<status>dropped</status>" in xml

    @pytest.mark.asyncio
    async def test_add_project_uploads_project_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        result = await store.add_project(name="Launch", folder_id="f1", status="inactive")
        assert result == {"status": "created", "project_id": result["project_id"], "name": "Launch"}
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert "<name>Launch</name>" in xml
        assert "<status>inactive</status>" in xml

    @pytest.mark.asyncio
    async def test_update_project_uploads_project_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        result = await store.update_project(_make_project())
        assert result == {"status": "updated", "project_id": "p1", "name": "Engineering"}
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert "<name>Engineering</name>" in xml
        assert "<review-interval>@1m</review-interval>" in xml

    @pytest.mark.asyncio
    async def test_mark_project_reviewed_updates_last_review(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        reviewed_at = datetime(2026, 3, 25, 10, 0, 0, tzinfo=UTC)
        result = await store.mark_project_reviewed(_make_project(), reviewed_at=reviewed_at)
        assert result == {
            "status": "reviewed",
            "project_id": "p1",
            "name": "Engineering",
            "next_review_recalculated": True,
        }
        xml = _read_contents_xml(client.put_file.await_args_list[0].args[1])
        assert "<last-review>2026-03-25T10:00:00.000Z</last-review>" in xml
        assert "<next-review>2026-04-25T10:00:00.000Z</next-review>" in xml

    @pytest.mark.asyncio
    async def test_mark_project_reviewed_keeps_unparseable_interval(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        project = dataclasses.replace(_make_project(), review_interval="bogus", next_review=None)
        result = await store.mark_project_reviewed(project, reviewed_at=NOW)
        assert result == {
            "status": "reviewed",
            "project_id": "p1",
            "name": "Engineering",
            "next_review_recalculated": False,
        }
        xml = _read_contents_xml(client.put_file.await_args_list[0].args[1])
        assert "<last-review>2026-03-22T12:00:00.000Z</last-review>" in xml
        assert "<review-interval>bogus</review-interval>" in xml

    @pytest.mark.asyncio
    async def test_add_folder_uploads_folder_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        result = await store.add_folder(name="Engineering", parent_folder_id="parent1")
        assert result == {
            "status": "created",
            "folder_id": result["folder_id"],
            "name": "Engineering",
        }
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert "<name>Engineering</name>" in xml
        assert '<folder idref="parent1"/>' in xml

    @pytest.mark.asyncio
    async def test_update_folder_uploads_folder_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        folder = dataclasses.replace(_make_folder(), parent_folder_id="parent1")
        result = await store.update_folder(folder)
        assert result == {"status": "updated", "folder_id": "f1", "name": "Engineering"}
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert "<name>Engineering</name>" in xml
        assert '<folder idref="parent1"/>' in xml

    @pytest.mark.asyncio
    async def test_drop_folder_uploads_folder_deletion_marker(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        result = await store.drop_folder(_make_folder())
        assert result == {"status": "dropped", "folder_id": "f1", "name": "Engineering"}
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert '<folder id="f1" op="delete">' in xml
        assert "<name>" not in xml

    @pytest.mark.asyncio
    async def test_add_tag_uploads_tag_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        result = await store.add_tag(name="@home", parent_tag_id="parent1", note="Desk")
        assert result == {"status": "created", "tag_id": result["tag_id"], "name": "@home"}
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert "<name>@home</name>" in xml
        assert '<context idref="parent1"/>' in xml
        assert "<note>Desk</note>" in xml

    @pytest.mark.asyncio
    async def test_update_tag_uploads_tag_transaction(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        tag = Tag(
            id="tag1",
            name="@home",
            parent_tag_id="parent1",
            rank=100,
            added=NOW,
            modified=NOW,
            note="Desk",
        )
        result = await store.update_tag(tag)
        assert result == {"status": "updated", "tag_id": "tag1", "name": "@home"}
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert "<name>@home</name>" in xml
        assert '<context idref="parent1"/>' in xml
        assert "<note>Desk</note>" in xml

    @pytest.mark.asyncio
    async def test_drop_tag_marks_tag_hidden(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        tag = Tag(id="tag1", name="@home", parent_tag_id=None, rank=100, added=NOW, modified=NOW)
        result = await store.drop_tag(tag)
        assert result == {"status": "dropped", "tag_id": "tag1", "name": "@home"}
        uploaded = client.put_file.await_args_list[0].args[1]
        xml = _read_contents_xml(uploaded)
        assert "<hidden>" in xml

    @pytest.mark.asyncio
    async def test_upload_transaction_rejects_missing_writable_key_slot(
        self, tmp_path: Path
    ) -> None:
        store, _ = _make_store(tmp_path)
        with pytest.raises(OFEncryptionError, match="Encrypted bundle has no writable key slot"):
            await store._upload_transaction(  # noqa: SLF001
                "20260322154011=client+parent.zip",
                b"payload",
                encrypted_plist=b"plist",
                key_slot=None,
                writer_state=_make_writer_state(),
            )

    @pytest.mark.asyncio
    async def test_upload_write_plan_rejects_empty_plan(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        with pytest.raises(OFError, match="Write plan produced no deltas"):
            await store._upload_write_plan(  # noqa: SLF001
                WritePlan(deltas=()),
                encrypted_plist=None,
                key_slot=None,
                writer_state=_make_writer_state(),
            )

    def test_load_writer_state_invalid_json_returns_none(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        (tmp_path / "writer_state.json").write_text("{not json")
        assert store._load_writer_state() is None  # noqa: SLF001

    def test_load_writer_identity_invalid_json_returns_none(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        (tmp_path / "writer_identity.json").write_text("{not json")
        assert store._load_writer_identity() is None  # noqa: SLF001

    def test_load_writer_state_invalid_shape_returns_none(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        (tmp_path / "writer_state.json").write_text(json.dumps({"client_id": 1}))
        assert store._load_writer_state() is None  # noqa: SLF001

    def test_load_writer_identity_invalid_shape_returns_none(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        (tmp_path / "writer_identity.json").write_text(json.dumps({"client_id": 1}))
        assert store._load_writer_identity() is None  # noqa: SLF001

    def test_load_writer_identity_with_valid_payload_returns_identity(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        payload = {
            "client_id": "abc",
            "host_id": "host",
            "device_name": "OmniFocus-CLI",
            "registration_date": NOW.isoformat(),
        }
        (tmp_path / "writer_identity.json").write_text(json.dumps(payload))
        identity = store._load_writer_identity()  # noqa: SLF001
        assert identity is not None
        assert identity.client_id == "abc"
        assert identity.host_id == "host"
        assert identity.device_name == "OmniFocus-CLI"

    def test_load_writer_identity_invalid_registration_date_returns_none(
        self, tmp_path: Path
    ) -> None:
        store, _ = _make_store(tmp_path)
        payload = {
            "client_id": "abc",
            "host_id": "host",
            "device_name": "OmniFocus-CLI",
            "registration_date": "not-a-date",
        }
        (tmp_path / "writer_identity.json").write_text(json.dumps(payload))
        assert store._load_writer_identity() is None  # noqa: SLF001

    def test_load_writer_state_invalid_encrypted_flag_returns_none(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        (tmp_path / "writer_state.json").write_text(
            json.dumps(
                {
                    "client_id": "abc",
                    "host_id": "host",
                    "device_name": "air.local",
                    "registration_date": NOW.isoformat(),
                    "tail_identifiers": ["tail"],
                    "hardware_cpu_count": None,
                    "hardware_cpu_type": None,
                    "hardware_cpu_type_name": None,
                    "hardware_model": None,
                    "os_version": None,
                    "os_version_number": None,
                    "encrypted": "yes",
                    "bundle_fingerprint": None,
                }
            )
        )
        assert store._load_writer_state() is None  # noqa: SLF001

    def test_load_writer_state_with_valid_fingerprint_returns_state(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        payload = {
            "client_id": "abc",
            "host_id": "host",
            "device_name": "air.local",
            "registration_date": NOW.isoformat(),
            "tail_identifiers": ["tail"],
            "hardware_cpu_count": "10",
            "hardware_cpu_type": "16777228,0",
            "hardware_cpu_type_name": "arm64",
            "hardware_model": "Mac16,12",
            "os_version": "25D2128",
            "os_version_number": "26.3.1",
            "encrypted": False,
            "bundle_fingerprint": [
                "00000000000000=base+tail.zip",
                ["20260322154011=head+tail.zip"],
                ["20260322154012=client.client"],
            ],
        }
        (tmp_path / "writer_state.json").write_text(json.dumps(payload))
        state = store._load_writer_state()  # noqa: SLF001
        assert state is not None
        assert state.bundle_fingerprint == (
            "00000000000000=base+tail.zip",
            ("20260322154011=head+tail.zip",),
            ("20260322154012=client.client",),
        )

    def test_load_writer_state_invalid_tail_identifiers_returns_none(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        payload = {
            "client_id": "abc",
            "host_id": "host",
            "device_name": "air.local",
            "registration_date": NOW.isoformat(),
            "tail_identifiers": [123],
            "hardware_cpu_count": None,
            "hardware_cpu_type": None,
            "hardware_cpu_type_name": None,
            "hardware_model": None,
            "os_version": None,
            "os_version_number": None,
            "encrypted": False,
            "bundle_fingerprint": None,
        }
        (tmp_path / "writer_state.json").write_text(json.dumps(payload))
        assert store._load_writer_state() is None  # noqa: SLF001

    def test_load_writer_state_invalid_registration_date_returns_none(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        payload = {
            "client_id": "abc",
            "host_id": "host",
            "device_name": "air.local",
            "registration_date": "not-a-date",
            "tail_identifiers": ["tail"],
            "hardware_cpu_count": None,
            "hardware_cpu_type": None,
            "hardware_cpu_type_name": None,
            "hardware_model": None,
            "os_version": None,
            "os_version_number": None,
            "encrypted": False,
            "bundle_fingerprint": None,
        }
        (tmp_path / "writer_state.json").write_text(json.dumps(payload))
        assert store._load_writer_state() is None  # noqa: SLF001

    def test_load_writer_state_invalid_optional_system_field_returns_none(
        self, tmp_path: Path
    ) -> None:
        store, _ = _make_store(tmp_path)
        payload = {
            "client_id": "abc",
            "host_id": "host",
            "device_name": "air.local",
            "registration_date": NOW.isoformat(),
            "tail_identifiers": ["tail"],
            "hardware_cpu_count": 10,
            "hardware_cpu_type": None,
            "hardware_cpu_type_name": None,
            "hardware_model": None,
            "os_version": None,
            "os_version_number": None,
            "encrypted": False,
            "bundle_fingerprint": None,
        }
        (tmp_path / "writer_state.json").write_text(json.dumps(payload))
        assert store._load_writer_state() is None  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_prepare_writer_rejects_missing_tail_identifier(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path, filenames=["00000000000000=base.zip"])
        client.get_file = AsyncMock(return_value=make_zip(_EMPTY_XML))
        with pytest.raises(OFError, match="Bundle has no known tail identifier"):
            await store._prepare_writer()  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_prepare_writer_rejects_diverged_frontier(self, tmp_path: Path) -> None:
        """A forked DAG (two un-merged frontier tips) must refuse a headless write
        rather than silently chaining onto one tip and leaving the fork unmerged."""
        store, _ = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=snap+root.zip",
                "20260101000000=root+tipA.zip",
                "20260101000001=root+tipB.zip",
            ],
        )
        with pytest.raises(OFError, match="diverged into 2 tips"):
            await store._prepare_writer()  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_partial_multi_delta_upload_rolls_back(self, tmp_path: Path) -> None:
        """If a later delta upload fails, the files already PUT this write are
        best-effort deleted so the bundle is not left half-written."""
        store, client = _make_store(tmp_path)
        put_calls = {"n": 0}

        async def _put(name: str, data: bytes) -> None:
            put_calls["n"] += 1
            if put_calls["n"] == 3:
                raise OFWebDAVError("upload boom", status_code=503)

        client.put_file = AsyncMock(side_effect=_put)
        deleted: list[str] = []

        async def _delete(name: str) -> None:
            deleted.append(name)
            if name.endswith(".client"):
                raise OFWebDAVError("already gone", status_code=404)

        client.delete_file = AsyncMock(side_effect=_delete)

        with pytest.raises(OFWebDAVError, match="upload boom"):
            await store.add_task(name="probe", inbox=True)

        # delta + client written before the 3rd-call failure were rolled back
        assert any(name.endswith(".zip") for name in deleted)
        assert any(name.endswith(".client") for name in deleted)

    def test_atomic_write_bytes_replaces_without_leftover_temp(self, tmp_path: Path) -> None:
        target = tmp_path / "state.json"
        _atomic_write_bytes(target, b"first")
        assert target.read_bytes() == b"first"
        _atomic_write_bytes(target, b"second")
        assert target.read_bytes() == b"second"
        assert not (tmp_path / ".state.json.tmp").exists()
        assert list(tmp_path.iterdir()) == [target]

    @pytest.mark.asyncio
    async def test_prepare_writer_uses_remote_template_only_for_tail_not_device_profile(
        self, tmp_path: Path
    ) -> None:
        client_doc = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>clientIdentifier</key><string>remote-client</string>
<key>hostID</key><string>ED325E58-F612-4653-BD34-7006A7D6DD52</string>
<key>name</key><string>air.local</string>
<key>registrationDate</key><date>2026-03-22T12:00:00Z</date>
<key>lastSyncDate</key><date>2026-03-22T12:00:00Z</date>
<key>tailIdentifiers</key><array><string>tail0</string></array>
<key>HardwareCPUCount</key><string>10</string>
<key>HardwareCPUType</key><string>16777228,0</string>
<key>HardwareCPUTypeName</key><string>arm64</string>
<key>HardwareModel</key><string>Mac16,12</string>
<key>OSVersion</key><string>25D2128</string>
<key>OSVersionNumber</key><string>26.3.1</string>
</dict></plist>"""
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=remote-client.client",
            ],
        )

        async def get_file(name: str) -> bytes:
            if name.endswith(".client"):
                return client_doc
            return make_zip(_EMPTY_XML)

        client.get_file = AsyncMock(side_effect=get_file)
        _, _, _, writer_state = await store._prepare_writer()  # noqa: SLF001
        assert writer_state.device_name == "OmniFocus-CLI"
        assert len(writer_state.host_id) == 36
        assert writer_state.hardware_model is None
        assert writer_state.os_version is None
        assert writer_state.os_version_number is None
        assert writer_state.tail_identifiers == ("tail0",)

    @pytest.mark.asyncio
    async def test_prepare_writer_prefers_explicit_env_identity_over_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client_doc = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>clientIdentifier</key><string>remote-client</string>
<key>hostID</key><string>REMOTE-HOST</string>
<key>name</key><string>remote.local</string>
<key>registrationDate</key><date>2026-03-22T12:00:00Z</date>
<key>lastSyncDate</key><date>2026-03-22T12:00:00Z</date>
<key>tailIdentifiers</key><array><string>tail0</string></array>
</dict></plist>"""
        monkeypatch.setenv("OF_CLIENT_ID", "STATICCLIENT01")
        monkeypatch.setenv("OF_DEVICE_NAME", "air.local")
        monkeypatch.setenv("OF_DEVICE_HOST_ID", "ED325E58-F612-4653-BD34-7006A7D6DD52")
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=remote-client.client",
            ],
        )

        async def get_file(name: str) -> bytes:
            if name.endswith(".client"):
                return client_doc
            return make_zip(_EMPTY_XML)

        client.get_file = AsyncMock(side_effect=get_file)
        _, _, _, writer_state = await store._prepare_writer()  # noqa: SLF001
        assert writer_state.client_id == "STATICCLIENT01"
        assert writer_state.device_name == "air.local"
        assert writer_state.host_id == "ED325E58-F612-4653-BD34-7006A7D6DD52"

    def test_current_tail_prefers_shared_client_tail(self, tmp_path: Path) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot.zip",
                "20260322154011=clientA.client",
                "20260322154012=clientB.client",
            ]
        )
        remote_clients = {
            "clientA": ClientStateDocument(
                client_identifier="clientA",
                host_id="hostA",
                name="a.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("shared-tail",),
            ),
            "clientB": ClientStateDocument(
                client_identifier="clientB",
                host_id="hostB",
                name="b.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("shared-tail",),
            ),
        }
        assert current_tail_id(state, remote_clients) == "shared-tail"

    def test_current_tail_prefers_latest_client_when_clients_disagree(self, tmp_path: Path) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+baseline-tail.zip",
                "20260322154011=clientA.client",
                "20260322154012=clientB.client",
            ]
        )
        remote_clients = {
            "clientA": ClientStateDocument(
                client_identifier="clientA",
                host_id="hostA",
                name="a.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("tail-a",),
            ),
            "clientB": ClientStateDocument(
                client_identifier="clientB",
                host_id="hostB",
                name="b.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("tail-b",),
            ),
        }
        assert current_tail_id(state, remote_clients) == "tail-b"

    def test_current_tail_uses_latest_delta_without_clients(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
            ]
        )
        assert current_tail_id(state, {}) == "head1"

    def test_current_tail_uses_latest_delta_when_baseline_has_no_tail(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot.zip",
                "20260322154011=tail0+head1.zip",
            ]
        )
        assert current_tail_id(state, {}) == "head1"

    def test_current_tail_uses_baseline_tail_without_clients_or_deltas(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(["00000000000000=snapshot+tail0.zip"])
        assert current_tail_id(state, {}) == "tail0"

    def test_current_tail_returns_none_when_bundle_has_no_tail_information(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(["00000000000000=snapshot.zip"])
        assert current_tail_id(state, {}) is None

    def test_tail_reachable_accepts_baseline_tail(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
            ]
        )
        assert tail_reachable_from_baseline(state, "tail0") is True

    def test_reachable_delta_tail_ids_skips_baseline_tail(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
            ]
        )
        assert reachable_delta_tail_ids(state, ("tail0", "head1")) == {"head1"}

    def test_reachable_delta_tail_ids_returns_selected_set(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
                "20260322154012=head1+head2.zip",
            ]
        )
        assert reachable_delta_tail_ids(state, ("head2",)) == {
            "head1",
            "head2",
        }

    def test_reachable_delta_tail_ids_ignores_already_selected_tail(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
                "20260322154012=head1+head2.zip",
            ]
        )
        assert reachable_delta_tail_ids(state, ("head1", "head2")) == {
            "head1",
            "head2",
        }

    def test_tail_reachable_uses_inner_baseline_branch(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
            ]
        )
        assert tail_reachable_from_baseline(state, "head1") is True

    def test_tail_reachable_uses_memoized_parent_result(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
                "20260322154012=head1+head2.zip",
                "20260322154013=head1+head2+head3.zip",
            ]
        )
        assert tail_reachable_from_baseline(state, "head3") is True

    def test_tail_reachable_returns_true_without_baseline_tail_or_deltas(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        no_tail_state = build_bundle_state(["00000000000000=snapshot.zip"])
        assert tail_reachable_from_baseline(no_tail_state, "anything") is True  # noqa: SLF001

        no_delta_state = build_bundle_state(["00000000000000=snapshot+tail0.zip"])
        assert tail_reachable_from_baseline(no_delta_state, "tail0") is True

    def test_tail_reachable_rejects_cycles(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=head2+head1.zip",
                "20260322154012=head1+head2.zip",
            ]
        )
        assert tail_reachable_from_baseline(state, "head1") is False

    def test_current_tail_skips_unreachable_latest_client_tail(self, tmp_path: Path) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
                "20260322154012=clientA.client",
                "20260322154013=clientB.client",
            ]
        )
        remote_clients = {
            "clientA": ClientStateDocument(
                client_identifier="clientA",
                host_id="hostA",
                name="a.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("head1",),
            ),
            "clientB": ClientStateDocument(
                client_identifier="clientB",
                host_id="hostB",
                name="b.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("missing-head",),
            ),
        }
        assert current_tail_id(state, remote_clients) == "head1"

    def test_current_tail_skips_client_without_tail_in_delta_mode(self, tmp_path: Path) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
                "20260322154012=clientA.client",
            ]
        )
        remote_clients = {
            "clientA": ClientStateDocument(
                client_identifier="clientA",
                host_id="hostA",
                name="a.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=(),
            )
        }
        assert current_tail_id(state, remote_clients) == "head1"

    def test_current_tail_falls_back_to_baseline_when_no_reachable_delta_exists(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=missing-parent+orphan-head.zip",
                "20260322154012=clientA.client",
            ]
        )
        remote_clients = {
            "clientA": ClientStateDocument(
                client_identifier="clientA",
                host_id="hostA",
                name="a.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("missing-head",),
            )
        }
        assert current_tail_id(state, remote_clients) == "tail0"

    def test_current_tail_uses_reachable_client_tail_when_latest_delta_is_unreachable(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+reachable-head.zip",
                "20260322154012=missing-parent+orphan-head.zip",
                "20260322154013=clientA.client",
            ]
        )
        remote_clients = {
            "clientA": ClientStateDocument(
                client_identifier="clientA",
                host_id="hostA",
                name="a.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("reachable-head",),
            )
        }
        assert current_tail_id(state, remote_clients) == "reachable-head"

    def test_current_tail_skips_empty_client_tail_and_falls_back_to_baseline(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=missing-parent+orphan-head.zip",
                "20260322154012=clientA.client",
            ]
        )
        remote_clients = {
            "clientA": ClientStateDocument(
                client_identifier="clientA",
                host_id="hostA",
                name="a.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=(),
            )
        }
        assert current_tail_id(state, remote_clients) == "tail0"

    def test_tail_depends_on_rejects_same_tail(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(["00000000000000=snapshot+tail0.zip"])
        assert tail_depends_on(state, "tail0", "tail0") is False

    def test_tail_depends_on_returns_false_for_missing_tail(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(["00000000000000=snapshot+tail0.zip"])
        assert tail_depends_on(state, "missing", "tail0") is False

    def test_tail_depends_on_breaks_cycles_with_visited_set(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=head2+head1.zip",
                "20260322154012=head1+head2.zip",
            ]
        )
        assert tail_depends_on(state, "head1", "missing") is False

    def test_tail_depends_on_returns_true_for_direct_parent(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
            ]
        )
        assert tail_depends_on(state, "head1", "tail0") is True

    def test_maximal_tail_ids_drops_ancestor_tails(self, tmp_path: Path) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail0.zip",
                "20260322154011=tail0+head1.zip",
                "20260322154012=head1+head2.zip",
            ]
        )
        maximal = maximal_tail_ids(state, ("head1", "head2"))
        assert maximal == ("head2",)

    def test_maybe_decrypt_returns_plaintext_without_doc_keys(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        payload = b"plain"
        assert store._maybe_decrypt(payload, None) == payload  # noqa: SLF001

    def test_maybe_decrypt_returns_plaintext_for_unencrypted_file_in_encrypted_bundle(
        self, tmp_path: Path
    ) -> None:
        from typing import cast

        store, _ = _make_store(tmp_path)
        payload = b"plain"
        doc_keys = cast(dict[int, tuple[bytes, bytes]], {1: (b"a", b"b")})
        assert store._maybe_decrypt(payload, doc_keys) == payload  # noqa: SLF001

    def test_select_client_template_returns_none_without_remote_clients(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(["00000000000000=snapshot+tail.zip"])
        assert store._select_client_template(state, {}) is None  # noqa: SLF001

    def test_select_client_template_skips_missing_latest_document(self, tmp_path: Path) -> None:
        from omnifocus.sync.client_state import ClientStateDocument
        from omnifocus.sync.protocol import build_bundle_state

        store, _ = _make_store(tmp_path)
        state = build_bundle_state(
            [
                "00000000000000=snapshot+tail.zip",
                "20260322154011=clientA.client",
                "20260322154012=clientB.client",
            ]
        )
        remote_clients = {
            "clientA": ClientStateDocument(
                client_identifier="clientA",
                host_id="hostA",
                name="a.local",
                registration_date=NOW,
                last_sync_date=NOW,
                tail_identifiers=("tail-a",),
            )
        }
        template = store._select_client_template(state, remote_clients)  # noqa: SLF001
        assert template is not None
        assert template.client_identifier == "clientA"

    @pytest.mark.asyncio
    async def test_upload_transaction_writes_delta_and_client_state(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        await store._upload_transaction(  # noqa: SLF001
            "20260322154011=client+parent.zip",
            make_zip(_EMPTY_XML),
            encrypted_plist=None,
            key_slot=None,
            writer_state=_make_writer_state(),
        )
        assert client.put_file.await_count == 2
        assert client.put_file.await_args_list[0].args[0].endswith(".zip")
        assert client.put_file.await_args_list[1].args[0].endswith(".client")
        payload = client.put_file.await_args_list[1].args[1]
        assert b"<string>tail123</string>" in payload

    def test_decrypt_transaction_contents_xml_plain_zip(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        xml = store.decrypt_transaction_contents_xml(  # noqa: SLF001
            encrypted_plist_bytes=b"ignored",
            file_bytes=make_zip(_EMPTY_XML),
        )
        assert "<omnifocus" in xml

    def test_decrypt_transaction_contents_xml_encrypted_zip(self, tmp_path: Path) -> None:
        from omnifocus.crypto.encryption import create_encrypted_bundle

        passphrase = "pw"  # noqa: S105
        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, encrypted_file = create_encrypted_bundle(plaintext, passphrase)
        store, _ = _make_store(tmp_path, passphrase=passphrase)
        xml = store.decrypt_transaction_contents_xml(  # noqa: SLF001
            encrypted_plist_bytes=encrypted_plist,
            file_bytes=encrypted_file,
        )
        assert "<omnifocus" in xml

    def test_decrypt_transaction_contents_xml_encrypted_without_passphrase_raises(
        self, tmp_path: Path
    ) -> None:
        from omnifocus.crypto.encryption import create_encrypted_bundle

        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, encrypted_file = create_encrypted_bundle(plaintext, "pw")
        store, _ = _make_store(tmp_path, passphrase=None)
        with pytest.raises(OFEncryptionError, match="no passphrase is available"):
            store.decrypt_transaction_contents_xml(  # noqa: SLF001
                encrypted_plist_bytes=encrypted_plist,
                file_bytes=encrypted_file,
            )

    def test_decrypt_transaction_contents_xml_unknown_key_slot_raises(self, tmp_path: Path) -> None:
        from omnifocus.crypto.encryption import create_encrypted_bundle, encrypt_file

        plaintext = make_zip(_EMPTY_XML)
        encrypted_plist, _ = create_encrypted_bundle(plaintext, "pw", slot_id=1)
        bad_file = encrypt_file(plaintext, b"A" * 16, b"B" * 16, key_id=99)
        store, _ = _make_store(tmp_path, passphrase="pw")  # noqa: S106
        with pytest.raises(OFEncryptionError, match="Key slot 99 not found"):
            store.decrypt_transaction_contents_xml(  # noqa: SLF001
                encrypted_plist_bytes=encrypted_plist,
                file_bytes=bad_file,
            )

    @pytest.mark.asyncio
    async def test_bundle_state_exposes_remote_clients(self, tmp_path: Path) -> None:
        client_doc = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>clientIdentifier</key><string>client123</string>
<key>hostID</key><string>host-123</string>
<key>name</key><string>air.local</string>
<key>registrationDate</key><date>2026-03-22T12:00:00Z</date>
<key>lastSyncDate</key><date>2026-03-22T12:00:00Z</date>
<key>tailIdentifiers</key><array><string>tail123</string></array>
</dict></plist>"""
        store, client = _make_store(
            tmp_path,
            filenames=[
                "00000000000000=base+tail0.zip",
                "20260322154011=client123.client",
                "delta_transactions.capability",
            ],
        )

        async def get_file(name: str) -> bytes:
            if name.endswith(".client"):
                return client_doc
            return make_zip(_EMPTY_XML)

        client.get_file = AsyncMock(side_effect=get_file)
        result = await store.bundle_state()
        assert result["baseline"]["tail_id"] == "tail0"
        assert result["clients"][0]["client_id"] == "client123"
        assert result["clients"][0]["tail_identifiers"] == ["tail123"]
        assert result["capabilities"] == ["delta_transactions"]


class TestContextManager:
    @pytest.mark.asyncio
    async def test_aclose_called_on_exit(self, tmp_path: Path) -> None:
        store, client = _make_store(tmp_path)
        async with store:
            pass
        client.aclose.assert_called_once()
