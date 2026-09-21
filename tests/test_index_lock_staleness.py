"""D1: the index rebuild lock must not become a permanent doorstop.

The lock file used to hold the literal text ``building`` and nothing else, so a
worker killed mid-rebuild left it behind and every later rebuild answered
``index_build_busy`` until a human deleted the file.  It now names its holder and
is taken over when it can no longer be live.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.retrieval.index_builder as index_builder
from src.retrieval.index_builder import IndexBuilder, IndexBuildBusyError


@pytest.fixture()
def builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        index_builder, "get_ingestion_settings",
        lambda: SimpleNamespace(index_build_lock_ttl_s=3600),
    )
    return IndexBuilder(
        registry=SimpleNamespace(data_dir=tmp_path),
        embedder=SimpleNamespace(),
        index_dir=tmp_path,
    )


def _write_lock(builder: IndexBuilder, payload: dict | str) -> Path:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    builder._lock_path.write_text(body, encoding="utf-8")
    return builder._lock_path


def test_acquire_writes_a_identifiable_holder(builder: IndexBuilder) -> None:
    builder._acquire_lock()
    try:
        holder = json.loads(builder._lock_path.read_text(encoding="utf-8"))
    finally:
        builder._release_lock()
    assert holder["pid"] == os.getpid()
    assert holder["host"] == socket.gethostname()
    assert holder["started_at"] <= time.time()


def test_live_lock_from_this_host_is_respected(builder: IndexBuilder) -> None:
    """Our own pid is alive and the stamp is fresh -- a second rebuild must wait."""

    _write_lock(builder, {"pid": os.getpid(), "host": socket.gethostname(), "started_at": time.time()})
    with pytest.raises(IndexBuildBusyError):
        builder._acquire_lock()


def test_expired_lock_is_taken_over(builder: IndexBuilder) -> None:
    _write_lock(builder, {"pid": os.getpid(), "host": socket.gethostname(), "started_at": time.time() - 7200})
    builder._acquire_lock()
    try:
        holder = json.loads(builder._lock_path.read_text(encoding="utf-8"))
        assert holder["pid"] == os.getpid()
        assert holder["started_at"] > time.time() - 60
    finally:
        builder._release_lock()


def test_dead_holder_on_this_host_is_taken_over(builder: IndexBuilder) -> None:
    # pid 1 exists in a container but not necessarily here; use an implausible pid
    # and assert through the helper the code itself uses.
    dead_pid = 999_999
    assert index_builder._pid_alive(dead_pid) is False
    _write_lock(builder, {"pid": dead_pid, "host": socket.gethostname(), "started_at": time.time()})
    builder._acquire_lock()
    holder = json.loads(builder._lock_path.read_text(encoding="utf-8"))
    assert holder["pid"] == os.getpid()
    builder._release_lock()


def test_legacy_text_lock_is_taken_over(builder: IndexBuilder) -> None:
    """A pre-D1 lock (or a half-written file) cannot be validated, so it is stale."""

    _write_lock(builder, "building\n")
    builder._acquire_lock()
    holder = json.loads(builder._lock_path.read_text(encoding="utf-8"))
    assert holder["pid"] == os.getpid()
    builder._release_lock()


def test_lock_without_start_time_is_taken_over(builder: IndexBuilder) -> None:
    _write_lock(builder, {"pid": os.getpid(), "host": socket.gethostname()})
    builder._acquire_lock()
    builder._release_lock()


def test_other_host_fresh_lock_is_respected(builder: IndexBuilder) -> None:
    """A different host may still be building; only the TTL can free that lock."""

    _write_lock(builder, {"pid": 999_999, "host": "some-other-worker", "started_at": time.time()})
    with pytest.raises(IndexBuildBusyError):
        builder._acquire_lock()


def test_broken_ttl_env_falls_back_to_an_hour(builder: IndexBuilder, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom():
        raise ValueError("index_build_lock_ttl_s must be at least 60 seconds")

    monkeypatch.setattr(index_builder, "get_ingestion_settings", boom)
    assert builder._lock_ttl_s() == 3600.0


def test_release_is_idempotent(builder: IndexBuilder) -> None:
    builder._acquire_lock()
    builder._release_lock()
    builder._release_lock()
    assert not builder._lock_path.exists()
