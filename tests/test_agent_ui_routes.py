"""Tests for the B0 UI-support routes (frontend plan §2): listings and reject.

build_runtime is monkeypatched with a tmp AgentDB + real MemoryService so no
retrieval stack or LLM is touched; the routes themselves run unchanged.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.agent_routes as agent_routes
from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.agent.runtime import AgentRunResult
from src.storage.agent_db import AgentDB


class _EmptyRegistry:
    def list_documents(self) -> list:
        return []


class _StubRuntime:
    """Minimal runtime so /agent/chat exercises the *route* logic (C1 validation,
    session binding) without a model or the retrieval stack."""

    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def run(self, session_id, user_text, mode=None, **kwargs):
        self.last_kwargs = {"session_id": session_id, "user_text": user_text, "mode": mode, **kwargs}
        return AgentRunResult(
            answer="stub", citations=[], non_source=[], warnings=[], tool_trace=[],
            mode=mode or "deep", gate={}, rounds=[], dropped_citations=[],
        )


@pytest.fixture()
def client_db(tmp_path: Path):
    db = AgentDB(tmp_path / "app.db")
    stub = _StubRuntime()
    memory = MemoryService(
        db=db,
        collections=CollectionService(db=db, registry=_EmptyRegistry()),
        settings=SimpleNamespace(digest_chars=600),
    )

    def fake_build_runtime():
        return stub, db, memory, None

    original = agent_routes.build_runtime
    agent_routes.build_runtime = fake_build_runtime
    try:
        from src.api.app import app

        yield TestClient(app), db, stub
    finally:
        agent_routes.build_runtime = original


def test_list_collections_route(client_db) -> None:
    client, _, _ = client_db
    created = client.post("/agent/collections", json={"name": "论文集", "document_ids": ["doc-a"]})
    assert created.status_code == 200
    listing = client.get("/agent/collections").json()["collections"]
    assert listing and listing[0]["name"] == "论文集"
    assert listing[0]["document_count"] == 1
    assert "updated_at" in listing[0]


def test_list_sessions_route(client_db) -> None:
    client, _, _ = client_db
    other = client.post("/agent/collections", json={"name": "other集"}).json()["collection_id"]
    session_a = client.post("/agent/sessions", json={"collection_id": None, "mode": "deep"}).json()["session_id"]
    session_b = client.post("/agent/sessions", json={"collection_id": other, "mode": "chat"}).json()["session_id"]
    all_sessions = {item["id"] for item in client.get("/agent/sessions").json()["sessions"]}
    assert {session_a, session_b} <= all_sessions
    scoped = client.get("/agent/sessions", params={"collection_id": other}).json()["sessions"]
    assert [item["id"] for item in scoped] == [session_b]


def test_memory_candidates_and_reject_routes(client_db) -> None:
    client, db, _ = client_db
    session_id = client.post("/agent/sessions", json={"mode": "deep"}).json()["session_id"]
    first = db.add_candidate(session_id, "profile", "goal", "读懂 Transformer")
    second = db.add_candidate(session_id, "preference", "style", "先给结论")

    pending = client.get("/agent/memory/candidates", params={"session_id": session_id}).json()["candidates"]
    assert [item["id"] for item in pending] == [first, second]

    rejected = client.post("/agent/memory/reject", json={"session_id": session_id, "candidate_ids": [first]})
    assert rejected.status_code == 200
    assert rejected.json() == {"rejected": 1}

    remaining = client.get("/agent/memory/candidates", params={"session_id": session_id}).json()["candidates"]
    assert [item["id"] for item in remaining] == [second]

    confirmed = client.post("/agent/memory/confirm", json={"session_id": session_id, "candidate_ids": [second]})
    assert confirmed.json() == {"confirmed": 1, "landed": 1}
    assert db.get_preferences() == {"style": "先给结论"}


def test_confirm_route_lands_progress_candidate(client_db) -> None:
    """B1 guard: a kind=progress candidate must actually write the progress table,
    not just flip its status -- the route now passes the session's collection_id."""

    client, db, _ = client_db
    collection_id = client.post("/agent/collections", json={"name": "进度集"}).json()["collection_id"]
    session_id = client.post("/agent/sessions", json={"collection_id": collection_id, "mode": "deep"}).json()["session_id"]
    candidate_id = db.add_candidate(session_id, "progress", "last_focus", "第 3 章 注意力机制")

    before = db.get_progress(collection_id)["last_focus"]
    confirmed = client.post("/agent/memory/confirm", json={"session_id": session_id, "candidate_ids": [candidate_id]})
    assert confirmed.json() == {"confirmed": 1, "landed": 1}
    assert before == ""
    assert db.get_progress(collection_id)["last_focus"] == "第 3 章 注意力机制"


def test_confirm_route_progress_without_collection_marks_but_does_not_land(client_db) -> None:
    """Without a bound collection the progress row cannot land; the route must
    report landed=0 so the gap is visible instead of silent."""

    client, db, _ = client_db
    session_id = client.post("/agent/sessions", json={"collection_id": None, "mode": "deep"}).json()["session_id"]
    candidate_id = db.add_candidate(session_id, "progress", "last_focus", "无归属进度")

    confirmed = client.post("/agent/memory/confirm", json={"session_id": session_id, "candidate_ids": [candidate_id]})
    assert confirmed.json() == {"confirmed": 1, "landed": 0}


def test_memory_note_rejects_illegal_kind(client_db) -> None:
    """B4 guard: an unsupported kind raises instead of silently defaulting to profile."""

    _, db, _ = client_db
    from src.adapter.contracts import INVALID_ARGUMENTS, AdapterError
    from src.agent.collections import CollectionService
    from src.agent.memory.service import MemoryService

    memory = MemoryService(
        db=db,
        collections=CollectionService(db=db, registry=_EmptyRegistry()),
        settings=SimpleNamespace(digest_chars=600),
    )
    session_id = db.create_session(collection_id=None, mode="deep")
    with pytest.raises(AdapterError) as raised:
        memory.note(session_id=session_id, kind="nonsense", key="k", value="v")
    assert raised.value.code == INVALID_ARGUMENTS


# ------------------------------------------------------------- R7 · C1 ghost-collection fix


def test_chat_with_unknown_collection_is_400_and_creates_nothing(client_db) -> None:
    """R7-4 / D-7a: passing an unknown collection_id must be a stable 400, not a
    silent create of a 0-document ghost (the old ``upsert_collection(name=id)``)."""

    client, _, stub = client_db
    before = {c["id"] for c in client.get("/agent/collections").json()["collections"]}
    resp = client.post("/agent/chat", json={"message": "hi", "collection_id": "col_ghost_123"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "collection_not_found"
    after = {c["id"] for c in client.get("/agent/collections").json()["collections"]}
    assert after == before
    stub.run  # runtime never reached (validation short-circuits)


def test_valid_chat_does_not_create_a_ghost_collection(client_db) -> None:
    """A real chat against an existing collection returns 200 and adds no new
    collection row."""

    client, _, stub = client_db
    cid = client.post("/agent/collections", json={"name": "真集", "document_ids": ["doc-a"]}).json()["collection_id"]
    before = {c["id"] for c in client.get("/agent/collections").json()["collections"]}
    resp = client.post("/agent/chat", json={"message": "hi", "collection_id": cid})
    assert resp.status_code == 200
    after = {c["id"] for c in client.get("/agent/collections").json()["collections"]}
    assert after == before  # the id was reused, never re-“upserted” as a name
    assert stub.last_kwargs["collection_ids"] == [cid]


def test_multi_collection_turn_binds_no_home_and_passes_union(client_db) -> None:
    """R7-1 / D-5b: a multi-collection turn forwards both ids to the runtime and
    creates a session with no single home collection."""

    client, _, stub = client_db
    a = client.post("/agent/collections", json={"name": "A", "document_ids": ["d1"]}).json()["collection_id"]
    b = client.post("/agent/collections", json={"name": "B", "document_ids": ["d2"]}).json()["collection_id"]
    resp = client.post("/agent/chat", json={"message": "hi", "collection_ids": [a, b]})
    assert resp.status_code == 200
    assert set(stub.last_kwargs["collection_ids"]) == {a, b}
    created = client.get("/agent/sessions").json()["sessions"]
    home = next(s for s in created if s["id"] == resp.json()["session_id"])
    assert home["collection_id"] is None  # message-scope only, not persisted


# ------------------------------------------------------------- R7-2 · read-only file/figure routes


def _figure_env(tmp_path: Path):
    data = tmp_path / "data"
    (data / "uploads").mkdir(parents=True)
    doc_id = "11111111-1111-1111-1111-111111111111"
    (data / "uploads" / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4 fake body")
    figures = data / "parsed" / doc_id / "figures"
    figures.mkdir(parents=True)
    (figures / "img_1.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    (figures / "note.txt").write_text("nope", encoding="utf-8")
    record = SimpleNamespace(document_id=doc_id, stored_filename=f"{doc_id}.pdf", original_filename="a.pdf")

    def get_document(d):
        if d == doc_id:
            return record
        from src.storage.document_registry import DocumentNotFoundError
        raise DocumentNotFoundError(d)

    registry = SimpleNamespace(data_dir=data, uploads_dir=data / "uploads", get_document=get_document)
    return registry, doc_id


def test_document_file_route_serves_pdf_inline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, doc_id = _figure_env(tmp_path)
    monkeypatch.setattr(agent_routes, "DocumentRegistry", lambda: registry)
    client = TestClient(_bare_app())
    resp = client.get(f"/agent/documents/{doc_id}/file")
    assert resp.status_code == 200
    assert resp.headers["content-disposition"] == "inline"
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF-")


def test_document_file_unknown_id_is_stable_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, _ = _figure_env(tmp_path)
    monkeypatch.setattr(agent_routes, "DocumentRegistry", lambda: registry)
    client = TestClient(_bare_app())
    resp = client.get("/agent/documents/does-not-exist/file")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "document_not_found"


def test_figures_list_returns_consumable_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P3: the response carries a base prefix + per-figure url so the frontend
    never has to guess ``image_path`` relative roots."""

    registry, doc_id = _figure_env(tmp_path)
    monkeypatch.setattr(agent_routes, "DocumentRegistry", lambda: registry)
    client = TestClient(_bare_app())
    resp = client.get(f"/agent/documents/{doc_id}/figures")
    assert resp.status_code == 200
    body = resp.json()
    assert body["base"] == f"/agent/documents/{doc_id}/figures/"
    urls = {item["url"] for item in body["figures"]}
    assert urls == {f"/agent/documents/{doc_id}/figures/img_1.png"}


def test_figure_served_inline_and_ext_whitelist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, doc_id = _figure_env(tmp_path)
    monkeypatch.setattr(agent_routes, "DocumentRegistry", lambda: registry)
    client = TestClient(_bare_app())

    ok = client.get(f"/agent/documents/{doc_id}/figures/img_1.png")
    assert ok.status_code == 200
    assert ok.headers["content-disposition"] == "inline"
    assert ok.headers["content-type"] == "image/png"

    bad_ext = client.get(f"/agent/documents/{doc_id}/figures/note.txt")
    assert bad_ext.status_code == 404
    assert bad_ext.json()["detail"]["code"] == "unsupported_type"

    missing = client.get(f"/agent/documents/{doc_id}/figures/gone.png")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "figure_not_found"


def test_figure_name_traversal_collapses_to_basename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct route-function call bypasses FastAPI's slash-routing so we can hand
    it a traversal-looking name; basename stripping must collapse it inside the
    figures dir and (since no such file exists there) yield figure_not_found."""

    registry, doc_id = _figure_env(tmp_path)
    monkeypatch.setattr(agent_routes, "DocumentRegistry", lambda: registry)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        agent_routes.document_figure(doc_id, "../../../../etc/passwd.png")
    assert exc.value.status_code == 404
    assert exc.value.detail["code"] in {"figure_not_found", "invalid_path"}


def test_figure_route_blocks_directory_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P10: requesting the figures dir itself (empty/`.` name) never enumerates
    the filesystem -- it resolves to the dir and 404s."""

    registry, doc_id = _figure_env(tmp_path)
    monkeypatch.setattr(agent_routes, "DocumentRegistry", lambda: registry)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        agent_routes.document_figure(doc_id, ".")
    assert exc.value.status_code == 404


# ------------------------------------------------------------- R7-3 · messages payload wrapper


def test_messages_endpoint_surfaces_parsed_payload(client_db) -> None:
    """R7-3: a persisted assistant row's ``payload_json`` (rounds / tool_trace /
    gate / observed chunks) is decoded into a structured ``payload`` field."""

    client, db, _ = client_db
    session_id = client.post("/agent/sessions", json={"mode": "deep"}).json()["session_id"]
    db.append_message(session_id, "assistant", "stub",
                      payload_json='{"rounds":[{"round":1}],"observed_chunk_ids":["c1"],"gate":{"searched":true}}')
    messages = client.get(f"/agent/sessions/{session_id}/messages").json()["messages"]
    row = next(m for m in messages if m["role"] == "assistant")
    assert row["payload"]["rounds"] == [{"round": 1}]
    assert row["payload"]["observed_chunk_ids"] == ["c1"]
    assert row["payload"]["gate"] == {"searched": True}


def test_messages_endpoint_survives_malformed_payload(client_db) -> None:
    """Malformed JSON must not break the listing; the row still returns content."""

    client, db, _ = client_db
    session_id = client.post("/agent/sessions", json={"mode": "deep"}).json()["session_id"]
    db.append_message(session_id, "assistant", "stub", payload_json="{not json")
    messages = client.get(f"/agent/sessions/{session_id}/messages").json()["messages"]
    row = next(m for m in messages if m["role"] == "assistant")
    assert "payload" not in row
    assert row["content"] == "stub"


def _bare_app():
    """Minimal FastAPI app mounting just the agent router for the file/figure
    route tests -- avoids pulling in the SPA catch-all that would otherwise own
    the ``/agent/documents/...`` path in a ``TestClient`` context."""

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(agent_routes.router)
    return app


# ------------------------------------------------------------- C6 / C8 · session-not-found contract


def test_reject_unknown_session_is_404(client_db) -> None:
    """C8: reject used to answer ``200 {"rejected": 0}`` for an id that does not
    exist while confirm answered 404 for the same input, so a typo read as
    "nothing to reject".  Both routes must report the missing session."""

    client, _, _ = client_db
    rejected = client.post("/agent/memory/reject", json={"session_id": "no-such-session", "candidate_ids": [1]})
    assert rejected.status_code == 404
    assert rejected.json()["detail"]["code"] == "session_not_found"

    confirmed = client.post("/agent/memory/confirm", json={"session_id": "no-such-session", "candidate_ids": [1]})
    assert confirmed.status_code == 404
    assert confirmed.json()["detail"]["code"] == "session_not_found"


def test_chat_with_unknown_session_is_404(tmp_path: Path) -> None:
    """C6: the runtime's first move is ``db.get_session(session_id)``, which
    raises KeyError; unhandled it surfaced as HTTP 500.  The SSE twin already
    guarded it -- the sync route must answer the same 404."""

    db = AgentDB(tmp_path / "app.db")
    memory = MemoryService(
        db=db,
        collections=CollectionService(db=db, registry=_EmptyRegistry()),
        settings=SimpleNamespace(digest_chars=600),
    )

    class _NoSessionRuntime:
        """Reaches for the session exactly like AgentRuntime.run does."""

        def run(self, session_id, user_text, mode=None, **kwargs):
            db.get_session(session_id)
            raise AssertionError("the route must not reach the answered path")

    original = agent_routes.build_runtime
    agent_routes.build_runtime = lambda: (_NoSessionRuntime(), db, memory, None)
    try:
        from src.api.app import app

        client = TestClient(app)
        resp = client.post("/agent/chat", json={"session_id": "no-such-session", "message": "hi"})
    finally:
        agent_routes.build_runtime = original

    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "session_not_found"
