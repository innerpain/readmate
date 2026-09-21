"""FE-2 / FE-3 / FE-4 route tests.

``build_runtime`` is monkeypatched with a tmp AgentDB (same pattern as
``test_agent_ui_routes.py``) so the routes run unchanged without a model, a
retrieval stack or the real ``data/app.db``.  The document route lives in
``src/api/app.py`` and takes the registry as a dependency, so that one is
exercised through ``app.dependency_overrides``.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.agent_routes as agent_routes
from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.agent.runtime import AgentRunResult
from src.storage.agent_db import AgentDB
from src.storage.document_registry import DocumentRegistry

_PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


class _EmptyRegistry:
    def list_documents(self) -> list:
        return []


class _StubRuntime:
    def run(self, session_id, user_text, mode=None, **kwargs):
        return AgentRunResult(
            answer="stub", citations=[], non_source=[], warnings=[], tool_trace=[],
            mode=mode or "deep", gate={}, rounds=[], dropped_citations=[],
        )


@pytest.fixture()
def client(tmp_path: Path):
    db = AgentDB(tmp_path / "app.db")
    memory = MemoryService(
        db=db,
        collections=CollectionService(db=db, registry=_EmptyRegistry()),
        settings=SimpleNamespace(digest_chars=600),
    )
    original = agent_routes.build_runtime
    agent_routes.build_runtime = lambda: (_StubRuntime(), db, memory, None)
    try:
        from src.api.app import app

        yield TestClient(app), db
    finally:
        agent_routes.build_runtime = original


# ------------------------------------------------------------------ FE-2 routes


def test_collection_rename_round_trips(client):
    test_client, db = client
    collection_id = db.upsert_collection("旧名字")

    response = test_client.patch(f"/agent/collections/{collection_id}", json={"name": "新名字"})
    assert response.status_code == 200
    assert response.json() == {"renamed": True, "collection_id": collection_id, "name": "新名字"}
    assert [row["name"] for row in db.list_collections()] == ["新名字"]


def test_collection_rename_conflict_is_a_409(client):
    test_client, db = client
    first = db.upsert_collection("A")
    db.upsert_collection("B")

    response = test_client.patch(f"/agent/collections/{first}", json={"name": "B"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "collection_name_taken"
    # and nothing changed
    assert sorted(row["name"] for row in db.list_collections()) == ["A", "B"]


def test_single_document_removal_from_a_collection(client):
    test_client, db = client
    collection_id = db.upsert_collection("papers")
    db.add_documents(collection_id, ["doc-1", "doc-2"])

    response = test_client.delete(f"/agent/collections/{collection_id}/documents/doc-1")
    assert response.status_code == 200
    assert response.json() == {
        "removed": True,
        "collection_id": collection_id,
        "document_id": "doc-1",
        "document_count": 1,
    }
    assert db.collection_document_ids(collection_id) == ["doc-2"]

    # idempotent: removing it again reports removed=False rather than erroring
    again = test_client.delete(f"/agent/collections/{collection_id}/documents/doc-1")
    assert again.status_code == 200
    assert again.json()["removed"] is False


def test_single_document_removal_unknown_collection_is_404(client):
    test_client, _ = client
    response = test_client.delete("/agent/collections/col_missing/documents/doc-1")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "collection_not_found"


def test_document_meta_route_renames_and_disables(tmp_path):
    """PATCH /documents/{id} (app-level route, registry injected)."""

    from src.api.app import app, get_document_registry

    registry = DocumentRegistry(data_dir=tmp_path / "data")
    document = registry.register_upload("attention.pdf", "application/pdf", io.BytesIO(_PDF))
    app.dependency_overrides[get_document_registry] = lambda: registry
    try:
        test_client = TestClient(app)
        renamed = test_client.patch(f"/documents/{document.document_id}", json={"alias": "注意力论文"})
        assert renamed.status_code == 200
        assert renamed.json()["document"]["alias"] == "注意力论文"

        disabled = test_client.patch(f"/documents/{document.document_id}", json={"enabled": False})
        assert disabled.status_code == 200
        body = disabled.json()["document"]
        assert body["enabled"] is False
        assert body["alias"] == "注意力论文"  # partial patch keeps the alias

        missing = test_client.patch("/documents/nope", json={"enabled": True})
        assert missing.status_code == 404
    finally:
        app.dependency_overrides.pop(get_document_registry, None)


# ------------------------------------------------------------------ FE-3 routes


def test_memory_entries_list_edit_and_delete(client):
    test_client, db = client
    session_id = db.create_session(collection_id=None, mode="deep")
    entry_id = db.add_candidate(session_id, "preference", "解释风格", "先给数据流")
    db.mark_candidates([entry_id], "confirmed")

    listed = test_client.get("/agent/memory/entries")
    assert listed.status_code == 200
    assert [row["id"] for row in listed.json()["entries"]] == [entry_id]

    edited = test_client.patch(f"/agent/memory/entries/{entry_id}", json={"value": "先给结论再给数据流"})
    assert edited.status_code == 200
    assert edited.json()["entry"]["value"] == "先给结论再给数据流"
    assert edited.json()["entry"]["key"] == "解释风格"  # untouched

    assert test_client.get("/agent/memory/entries?status=pending").json()["entries"] == []

    deleted = test_client.delete(f"/agent/memory/entries/{entry_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": 1, "entry_id": entry_id}
    assert test_client.get("/agent/memory/entries").json()["entries"] == []


def test_memory_entry_patch_unknown_id_is_404(client):
    test_client, _ = client
    response = test_client.patch("/agent/memory/entries/999", json={"value": "x"})
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "memory_entry_not_found"


def test_profile_and_preferences_are_editable(client):
    test_client, db = client

    profile = test_client.patch("/agent/profile", json={"major": "数据科学与大数据"})
    assert profile.status_code == 200
    assert profile.json()["profile"]["major"] == "数据科学与大数据"
    assert db.get_profile()["major"] == "数据科学与大数据"

    # partial edit keeps the earlier field
    profile = test_client.patch("/agent/profile", json={"goal": "AI 应用开发实习"})
    assert profile.json()["profile"]["major"] == "数据科学与大数据"
    assert profile.json()["profile"]["goal"] == "AI 应用开发实习"

    saved = test_client.put("/agent/preferences/回答语言", json={"value": "中文"})
    assert saved.status_code == 200
    assert saved.json()["preferences"]["回答语言"] == "中文"
    assert db.get_preferences()["回答语言"] == "中文"

    dropped = test_client.delete("/agent/preferences/回答语言")
    assert dropped.json() == {"deleted": 1, "key": "回答语言"}
    assert db.get_preferences() == {}


# ------------------------------------------------------------------ FE-4 routes


def test_session_listing_supports_offset(client):
    test_client, db = client
    for index in range(3):
        session_id = db.create_session(collection_id=None, mode="deep", title=f"s{index}")
        db.set_session_title(session_id, f"s{index}")

    first_page = test_client.get("/agent/sessions?limit=2").json()["sessions"]
    second_page = test_client.get("/agent/sessions?limit=2&offset=2").json()["sessions"]
    assert len(first_page) == 2
    assert len(second_page) == 1
    assert {row["id"] for row in first_page}.isdisjoint({row["id"] for row in second_page})
