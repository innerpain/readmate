"""C4: a session can be renamed and deleted.

``create_session`` and ``list_sessions`` always carried a ``title``; the write
surface was missing, so every session kept the name it was born with.  Deleting
must take the transcript (messages + pending memory candidates cascade) and leave
documents/collections/index alone.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.agent_routes as agent_routes
from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.storage.agent_db import AgentDB


class _EmptyRegistry:
    def list_documents(self) -> list:
        return []


@pytest.fixture()
def client_db(tmp_path: Path):
    db = AgentDB(tmp_path / "app.db")
    memory = MemoryService(
        db=db,
        collections=CollectionService(db=db, registry=_EmptyRegistry()),
        settings=SimpleNamespace(digest_chars=600),
    )

    def fake_build_runtime():
        return SimpleNamespace(), db, memory, None

    original = agent_routes.build_runtime
    agent_routes.build_runtime = fake_build_runtime
    try:
        from src.api.app import app

        yield TestClient(app), db
    finally:
        agent_routes.build_runtime = original


def test_rename_session_route_updates_the_title(client_db) -> None:
    client, db = client_db
    session_id = client.post("/agent/sessions", json={"collection_id": None, "mode": "deep"}).json()["session_id"]

    renamed = client.patch(f"/agent/sessions/{session_id}", json={"title": "RAG 记忆机制"})
    assert renamed.status_code == 200
    assert renamed.json() == {"session_id": session_id, "title": "RAG 记忆机制"}

    rows = {row["id"]: row for row in client.get("/agent/sessions").json()["sessions"]}
    assert rows[session_id]["title"] == "RAG 记忆机制"
    assert db.get_session(session_id)["title"] == "RAG 记忆机制"


def test_rename_rejects_blank_and_unknown_session(client_db) -> None:
    client, _ = client_db
    session_id = client.post("/agent/sessions", json={"collection_id": None, "mode": "deep"}).json()["session_id"]

    # Field(min_length=1) rejects "" before the route sees it; the route's own
    # 400 covers whitespace-only titles, which pydantic cannot spot.
    assert client.patch(f"/agent/sessions/{session_id}", json={"title": ""}).status_code == 422
    blank = client.patch(f"/agent/sessions/{session_id}", json={"title": "   "})
    assert blank.status_code == 400
    assert blank.json()["detail"]["code"] == "title_empty"

    missing = client.patch("/agent/sessions/ses_missing", json={"title": "x"})
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "session_not_found"


def test_delete_session_removes_transcript_only(client_db) -> None:
    client, db = client_db
    collection_id = client.post("/agent/collections", json={"name": "papers"}).json()["collection_id"]
    session_id = client.post("/agent/sessions", json={"collection_id": collection_id, "mode": "deep"}).json()["session_id"]
    db.append_message(session_id, "user", "hello", payload_json=None)
    db.append_message(session_id, "assistant", "hi", payload_json=None)
    other_session = client.post("/agent/sessions", json={"collection_id": collection_id, "mode": "chat"}).json()["session_id"]

    removed = client.delete(f"/agent/sessions/{session_id}")
    assert removed.status_code == 200
    body = removed.json()
    assert body["deleted"] is True
    assert body["deleted_messages"] == 2

    remaining = {row["id"] for row in client.get("/agent/sessions").json()["sessions"]}
    assert session_id not in remaining
    assert other_session in remaining, "only the requested session may go"
    assert db.count_messages(session_id) == 0, "the transcript must cascade away"

    # The evidence side is untouched: the collection still lists as before.
    collections = {row["id"]: row for row in client.get("/agent/collections").json()["collections"]}
    assert collection_id in collections


def test_delete_unknown_session_is_404(client_db) -> None:
    client, _ = client_db
    missing = client.delete("/agent/sessions/ses_missing")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "session_not_found"
