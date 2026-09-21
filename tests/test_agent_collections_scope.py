"""R7 unit tests: multi-collection scope and digest progress gating (D-6b).

D-5b (message-level scope, nothing persisted) was revoked on 2026-09-19: 问题.md
第二轮第 8 条 requires the material to be chosen *before* the conversation starts, so
the scope is now locked on the session on its first turn and enforced afterwards --
see ``tests/test_session_scope_lock.py``.

Uses a real tmp AgentDB; only the adapter is stubbed (a spy captures the kwargs
``ToolRunner._search`` forwards), so no retrieval stack or model is touched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.adapter.contracts import COLLECTION_EMPTY, AdapterError, SearchHit
from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.agent.tools import ToolRunner
from src.config.settings import AgentSettings
from src.llm.types import ToolCall
from src.storage.agent_db import AgentDB


class _EmptyRegistry:
    def list_documents(self) -> list:
        return []


class _SpyAdapter:
    """Records the kwargs of the last search so the tests can assert what
    ``ToolRunner`` actually forwards (document_ids / gate_text / mode / streak)."""

    def __init__(self, hits):
        self._hits = list(hits)
        self.last_kwargs: dict = {}

    def search(self, collection_id, query, top_k=None, document_ids=None, gate_text=None, mode=None, previous_empty_searches=0):
        self.last_kwargs = {
            "collection_id": collection_id, "query": query, "top_k": top_k,
            "document_ids": document_ids, "gate_text": gate_text, "mode": mode,
            "previous_empty_searches": previous_empty_searches,
        }
        return self._hits


def _hit(chunk_id: str = "c1", document_id: str = "d1") -> SearchHit:
    return SearchHit(chunk_id=chunk_id, document_id=document_id, filename="a.pdf", page=1,
                     score=0.9, chunk_type="prose", presentation_content="body", excerpt="body")


def _collections(db):
    return CollectionService(db=db, registry=_EmptyRegistry())


def test_document_ids_multi_unions_across_collections(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    a = db.upsert_collection("A"); b = db.upsert_collection("B")
    db.add_documents(a, ["d1", "d2"]); db.add_documents(b, ["d2", "d3"])
    service = _collections(db)
    assert service.document_ids_multi([a, b]) == {"d1", "d2", "d3"}


def test_document_ids_multi_ignores_empty_and_dedups(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    a = db.upsert_collection("A"); empty = db.upsert_collection("E")
    db.add_documents(a, ["d1"])
    service = _collections(db)
    assert service.document_ids_multi([a, empty, a, ""]) == {"d1"}


def test_document_ids_multi_raises_when_all_empty(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    a = db.upsert_collection("A")
    service = _collections(db)
    with pytest.raises(AdapterError) as raised:
        service.document_ids_multi([a])
    assert raised.value.code == COLLECTION_EMPTY


def test_list_docs_multi_dedups_by_document(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    a = db.upsert_collection("A"); b = db.upsert_collection("B")
    db.add_documents(a, ["d1"]); db.add_documents(b, ["d1", "d2"])
    registry = SimpleNamespace(list_documents=lambda: [
        SimpleNamespace(document_id="d1", original_filename="one.pdf", status="ready", page_count=5, chunk_count=9, failure_code=None),
        SimpleNamespace(document_id="d2", original_filename="two.pdf", status="ready", page_count=7, chunk_count=11, failure_code=None),
    ])
    service = CollectionService(db=db, registry=registry)
    docs = service.list_docs_multi([a, b])
    assert {doc.document_id for doc in docs} == {"d1", "d2"}
    assert len(docs) == 2  # d1 present once despite two collections


def _memory(db):
    return MemoryService(db=db, collections=_collections(db), settings=SimpleNamespace(digest_chars=600))


def test_digest_injects_progress_only_for_single_collection(tmp_path):
    """D-6b: progress is per-collection; only a single-collection scope gets it."""

    db = AgentDB(tmp_path / "app.db")
    a = db.upsert_collection("A"); b = db.upsert_collection("B")
    db.set_progress(a, last_focus="注意力机制")
    db.set_progress(b, last_focus="表格题")
    memory = _memory(db)

    assert "注意力机制" in memory.digest(None, collection_ids=[a])
    # multi -> neither collection's progress leaks into context
    both = memory.digest(None, collection_ids=[a, b])
    assert "注意力机制" not in both and "表格题" not in both
    # legacy single-arg path still works
    assert "注意力机制" in memory.digest(a)


def test_tool_search_forwards_union_document_ids(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    a = db.upsert_collection("A"); b = db.upsert_collection("B")
    db.add_documents(a, ["d1"]); db.add_documents(b, ["d2"])
    adapter = _SpyAdapter(hits=[_hit("c1", "d1")])
    runner = ToolRunner(adapter=adapter, collections=_collections(db), memory=None, settings=AgentSettings())
    call = ToolCall(id="1", name="search", arguments={"query": "anything"})

    runner._search(call, None, "anything", session_id=None, collection_ids=[a, b])
    assert adapter.last_kwargs["document_ids"] == {"d1", "d2"}
    assert adapter.last_kwargs["collection_id"] == a  # representative = first id
    assert adapter.last_kwargs["gate_text"] == "anything"


def test_tool_search_single_collection_uses_collection_id_resolution(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    a = db.upsert_collection("A")
    db.add_documents(a, ["d1"])
    adapter = _SpyAdapter(hits=[_hit()])
    runner = ToolRunner(adapter=adapter, collections=_collections(db), memory=None, settings=AgentSettings())
    call = ToolCall(id="1", name="search", arguments={"query": "anything"})

    runner._search(call, a, "anything")
    # single collection -> no override; adapter resolves scope from collection_id
    assert adapter.last_kwargs["document_ids"] is None
    assert adapter.last_kwargs["collection_id"] == a


def test_tool_search_no_scope_is_collection_empty(tmp_path):
    adapter = _SpyAdapter(hits=[])
    runner = ToolRunner(adapter=adapter, collections=_collections(AgentDB(tmp_path / "app.db")), memory=None, settings=AgentSettings())
    call = ToolCall(id="1", name="search", arguments={"query": "anything"})
    observation = runner._search(call, None, "anything")
    assert not observation.ok
    assert observation.error_code == COLLECTION_EMPTY
