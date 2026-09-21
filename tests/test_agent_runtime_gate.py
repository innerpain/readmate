"""The deep-read gate, the chat soft constraint and citation validation (A1-A3).

The gate is the product's spine: a deep-read turn may not be finished without a
successful search or read, and "success" means a non-empty observation, never a
well-worded answer.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.adapter.contracts import AdapterError, DocInfo, SearchHit
from src.adapter.mock_adapter import MockRagAdapter
from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.agent.runtime import MODE_CHAT, MODE_DEEP, AgentRuntime
from src.agent.tools import ToolRunner
from src.config.settings import get_agent_settings
from src.llm.fake import FakeLLMClient
from src.llm.types import LLMStep, ToolCall
from src.storage.agent_db import AgentDB


class _EmptyRegistry:
    def list_documents(self) -> list:
        return []


def _hit(chunk_id: str = "c1", document_id: str = "d1", page: int = 3) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="attention.pdf",
        page=page,
        section="3 Model Architecture",
        score=0.81,
        chunk_type="prose",
        presentation_content="encoders are a stack of six identical layers",
        excerpt="six identical layers",
    )


def _answer(chunk_id: str = "c1", page: int = 3, text: str = "The encoder has six identical layers.") -> str:
    return json.dumps(
        {
            "answer": text,
            "citations": [{"chunk_id": chunk_id, "page": page, "quote": "six identical layers"}],
            "non_source": [],
        }
    )


def _search_call(call_id: str = "1", query: str = "encoder layers") -> LLMStep:
    return LLMStep(tool_calls=[ToolCall(id=call_id, name="search", arguments={"query": query})])


def _build(tmp_path: Path, *, hits=(), script=(), mode: str = MODE_DEEP, docs=(), error: AdapterError | None = None):
    settings = get_agent_settings()
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("col")
    document_ids = [hit.document_id for hit in hits] or ["d1"]
    db.add_documents(collection_id, document_ids)
    adapter = MockRagAdapter(hits=list(hits), docs=list(docs), error=error)
    collections = CollectionService(db=db, registry=_EmptyRegistry())
    memory = MemoryService(db=db, collections=collections, settings=settings)
    runner = ToolRunner(adapter=adapter, collections=collections, memory=memory, settings=settings)
    runtime = AgentRuntime(llm=FakeLLMClient(list(script)), tools=runner, db=db, settings=settings)
    session_id = db.create_session(collection_id=collection_id, mode=mode)
    return runtime, db, session_id, adapter


def test_deep_mode_refuses_to_finish_without_any_evidence(tmp_path):
    runtime, db, session_id, adapter = _build(
        tmp_path,
        hits=[_hit()],
        script=[LLMStep(content=_answer()), LLMStep(content=_answer())],
    )
    result = runtime.run(session_id, "编码器有多少层？")
    assert result.refused is True
    assert result.failure == "no_evidence_after_gate"
    assert result.citations == []
    assert adapter.search_calls == []
    assert result.gate["searched"] is False
    # user message + the refusal are persisted; the blocked drafts are not.
    assert db.count_messages(session_id) == 2


def test_deep_mode_blocks_once_then_accepts_a_cited_answer(tmp_path):
    runtime, db, session_id, adapter = _build(
        tmp_path,
        hits=[_hit()],
        script=[LLMStep(content=_answer()), _search_call(), LLMStep(content=_answer())],
    )
    result = runtime.run(session_id, "编码器有多少层？")
    assert result.failure is None
    assert result.gate == {"blocks": 1, "searched": True, "read": False}
    assert len(adapter.search_calls) == 1
    assert [step["tool"] for step in result.tool_trace] == ["search"]
    assert result.tool_trace[0]["route"] is None  # the mock exposes no route decision
    assert len(result.citations) == 1
    assert result.citations[0]["chunk_id"] == "c1"
    assert result.citations[0]["document_id"] == "d1"
    assert result.citations[0]["filename"] == "attention.pdf"
    assert result.citations[0]["page"] == 3


def test_a_failed_search_does_not_satisfy_the_gate(tmp_path):
    runtime, db, session_id, adapter = _build(
        tmp_path,
        hits=[_hit()],
        error=AdapterError("knowledge_base_empty", "no snapshot"),
        script=[_search_call(), LLMStep(content=_answer()), LLMStep(content=_answer())],
    )
    result = runtime.run(session_id, "编码器有多少层？")
    assert result.refused is True
    assert result.failure == "no_evidence_after_gate"
    assert len(adapter.search_calls) == 1
    assert result.tool_trace[0]["ok"] is False
    assert result.tool_trace[0]["error"] == "knowledge_base_empty"


def test_chat_mode_greeting_skips_retrieval(tmp_path):
    runtime, db, session_id, adapter = _build(
        tmp_path,
        hits=[_hit()],
        mode=MODE_CHAT,
        script=[LLMStep(content=_answer())],
    )
    result = runtime.run(session_id, "你好")
    assert adapter.search_calls == []
    assert result.gate["searched"] is False
    assert result.citations == []
    assert "citations_dropped_unobserved" in result.warnings


def test_chat_mode_warns_once_when_a_related_question_skips_search(tmp_path):
    runtime, db, session_id, adapter = _build(
        tmp_path,
        hits=[_hit()],
        mode=MODE_CHAT,
        script=[LLMStep(content=_answer()), _search_call(query="论文第 3 节"), LLMStep(content=_answer())],
    )
    result = runtime.run(session_id, "这篇论文的第 3 节讲了什么")
    assert result.gate["blocks"] == 1
    assert len(adapter.search_calls) == 1
    assert len(result.citations) == 1


def test_citations_must_come_from_this_turns_observations(tmp_path):
    runtime, db, session_id, adapter = _build(
        tmp_path,
        hits=[_hit()],
        script=[_search_call(), LLMStep(content=_answer(chunk_id="cX", page=9))],
    )
    result = runtime.run(session_id, "编码器有多少层？")
    assert result.citations == []
    assert "citations_dropped_unobserved" in result.warnings
    assert result.tool_trace[0]["ok"] is True


def test_read_tool_counts_as_evidence_in_deep_mode(tmp_path):
    from src.adapter.contracts import ReadResult

    settings = get_agent_settings()
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("col")
    db.add_documents(collection_id, ["d1"])
    adapter = MockRagAdapter(
        reads={"c1": ReadResult(document_id="d1", chunk_id="c1", page=3, section="3", text="six layers", chunk_type="prose")}
    )
    collections = CollectionService(db=db, registry=_EmptyRegistry())
    memory = MemoryService(db=db, collections=collections, settings=settings)
    runner = ToolRunner(adapter=adapter, collections=collections, memory=memory, settings=settings)
    llm = FakeLLMClient(
        [
            LLMStep(tool_calls=[ToolCall(id="1", name="read", arguments={"chunk_id": "c1"})]),
            LLMStep(content=_answer()),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=runner, db=db, settings=settings)
    session_id = db.create_session(collection_id=collection_id, mode=MODE_DEEP)

    result = runtime.run(session_id, "把这个片段讲一下")
    assert result.failure is None
    assert result.gate["read"] is True
    assert result.gate["searched"] is False
    assert len(result.citations) == 1
    assert result.citations[0]["quote"] == "six identical layers"
