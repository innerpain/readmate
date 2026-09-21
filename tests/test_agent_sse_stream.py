"""SSE event plumbing (frontend plan §5 / V1): runtime emits, sync path untouched.

Uses FakeLLMClient.stream_complete (added for this) + MockRagAdapter — no
network, no real index.  Two layers are covered: the runtime's on_event
sequence, and the /agent/chat/stream HTTP route producing parseable frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.agent_routes as agent_routes
from src.adapter.mock_adapter import MockRagAdapter
from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.agent.runtime import AgentRuntime
from src.agent.tools import ToolRunner
from src.config.settings import get_agent_settings
from src.llm.fake import FakeLLMClient
from src.llm.types import LLMStep, ToolCall
from src.storage.agent_db import AgentDB


class _EmptyRegistry:
    def list_documents(self) -> list:
        return []


def _hit():
    from src.adapter.contracts import SearchHit

    return SearchHit(
        chunk_id="c1", document_id="d1", filename="attention.pdf", page=3,
        section="3 Model", score=0.8, chunk_type="prose",
        presentation_content="six identical layers", excerpt="six",
    )


def _answer(chunk_id: str = "c1") -> str:
    return json.dumps({"answer": "六个相同层。", "citations": [{"chunk_id": chunk_id, "page": 3, "quote": "six"}], "non_source": []})


def _search_step(call_id: str = "1") -> LLMStep:
    return LLMStep(tool_calls=[ToolCall(id=call_id, name="search", arguments={"query": "encoder"})])


def _runtime(tmp_path: Path, script):
    settings = get_agent_settings()
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("sse-col")
    db.add_documents(collection_id, ["d1"])
    adapter = MockRagAdapter(hits=[_hit()], docs=[])
    collections = CollectionService(db=db, registry=_EmptyRegistry())
    memory = MemoryService(db=db, collections=collections, settings=settings)
    runner = ToolRunner(adapter=adapter, collections=collections, memory=memory, settings=settings)
    runtime = AgentRuntime(llm=FakeLLMClient(list(script)), tools=runner, db=db, settings=settings)
    session_id = db.create_session(collection_id=collection_id, mode="deep")
    return runtime, session_id


def test_stream_emits_tool_and_answer_events_in_order(tmp_path):
    events: list[tuple[str, dict]] = []
    runtime, session_id = _runtime(tmp_path, [_search_step(), LLMStep(content=_answer())])
    result = runtime.run(session_id, "编码器几层？", on_event=lambda name, payload: events.append((name, payload)), stream=True)
    names = [name for name, _ in events]
    # optimistic stream: the search round has no content, so no delta/reset noise
    assert "answer_reset" not in names
    assert names[0] == "tool_call" and names[1] == "tool_result"
    call = next(payload for name, payload in events if name == "tool_call")
    assert call["tool"] == "search" and call["round"] == 1 and call["call_index"] == 0
    done = [payload for name, payload in events if name == "tool_result"][0]
    assert done["ok"] is True and done["hits"] == 1
    assert result.failure is None
    # deltas concatenated must equal the raw final JSON envelope (turn body intact)
    deltas = "".join(payload["text"] for name, payload in events if name == "answer_delta")
    assert deltas == _answer()


def test_tool_round_text_gets_reset_before_tool_call(tmp_path):
    # A round that streams prose AND calls a tool must emit answer_reset first.
    step = LLMStep(content="我先查一下", tool_calls=[ToolCall(id="1", name="search", arguments={"query": "q"})])
    events: list[str] = []
    runtime, session_id = _runtime(tmp_path, [step, LLMStep(content=_answer())])
    runtime.run(session_id, "查一下", on_event=lambda name, payload: events.append(name), stream=True)
    assert "answer_reset" in events
    assert events.index("answer_reset") < events.index("tool_call")


def test_callback_without_stream_gets_loop_events_but_no_deltas(tmp_path):
    """Contract: on_event (if given) always sees tool/gate events; the
    answer_delta / answer_reset pair requires stream=True.  The default call
    (no kwargs) stays byte-for-byte the old path."""
    events: list[str] = []
    runtime, session_id = _runtime(tmp_path, [_search_step(), LLMStep(content=_answer())])
    result = runtime.run(session_id, "编码器几层？", on_event=lambda name, payload: events.append(name))
    assert events == ["tool_call", "tool_result"]
    assert "answer_delta" not in events and "answer_reset" not in events
    assert result.gate["searched"] is True

    plain, session_id2 = _runtime(tmp_path, [_search_step(), LLMStep(content=_answer())])
    assert plain.run(session_id2, "编码器几层？").gate["searched"] is True  # no callback, no crash


@pytest.fixture()
def client_with_runtime(tmp_path: Path):
    runtime, session_id = _runtime(tmp_path, [_search_step(), LLMStep(content=_answer())])
    db = runtime.db

    def fake_build_runtime():
        return runtime, db, None, None

    original = agent_routes.build_runtime
    agent_routes.build_runtime = fake_build_runtime
    try:
        from src.api.app import app

        yield TestClient(app), session_id
    finally:
        agent_routes.build_runtime = original


class _NS(SimpleNamespace):
    """Attribute-access namespace; missing attrs return None via getattr."""


def test_stream_complete_aggregates_tool_call_deltas():
    """OpenAI chunking: id/name arrive on the first fragment of each index,
    arguments concatenate across fragments; content deltas leak through
    on_delta in order."""
    from src.llm.client import LLMClient
    from src.llm.types import ModelProfile

    chunks = [
        _NS(choices=[_NS(delta=_NS(content="你好", tool_calls=None))]),
        _NS(choices=[_NS(delta=_NS(content=None, tool_calls=[
            _NS(index=0, id="call_a", function=_NS(name="search", arguments='{"que'))]))]),
        _NS(choices=[_NS(delta=_NS(content=None, tool_calls=[
            _NS(index=0, id=None, function=_NS(name=None, arguments='ry":"q"}'))]))]),
        _NS(choices=[_NS(delta=_NS(content=None, tool_calls=[
            _NS(index=1, id="call_b", function=_NS(name="list_docs", arguments="{}"))]))]),
    ]

    class _FakeCompletions:
        def create(self, **kwargs):
            assert kwargs.get("stream") is True
            return iter(chunks)

    client = LLMClient(ModelProfile(name="t", model="m"))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    seen: list[str] = []
    step = client.stream_complete([{"role": "user", "content": "hi"}], tools=[], on_delta=seen.append)
    assert step.content == "你好"
    assert seen == ["你好"]
    assert [(call.id, call.name, call.arguments) for call in step.tool_calls] == [
        ("call_a", "search", {"query": "q"}),
        ("call_b", "list_docs", {}),
    ]


def test_stream_route_returns_sse_frames(client_with_runtime):
    client, session_id = client_with_runtime
    response = client.post("/agent/chat/stream", json={"message": "编码器几层？", "session_id": session_id, "mode": "deep"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    frames = [block for block in body.split("\n\n") if block.startswith("event:")]
    names = [block.split("\n")[0].removeprefix("event: ").strip() for block in frames]
    assert names[0] == "turn_start"
    assert "tool_call" in names and "tool_result" in names
    assert names[-1] == "turn_end"  # the sync-shaped payload closes every turn
    turn_end = json.loads([block.split("data: ", 1)[1] for block in frames if block.startswith("event: turn_end")][0])
    assert turn_end["session_id"] == session_id
    assert len(turn_end["citations"]) == 1


# ------------------------------------------------------------- A9 · gate resets streamed text

def test_gate_blocked_round_resets_streamed_text(tmp_path):
    """A9: a round the gate refuses must reset its streamed text *before* the
    gate_block event.  Otherwise the UI keeps showing a confident answer (the
    model did write one) and only ``turn_end`` replaces it with the refusal,
    which reads as the answer being silently taken away."""

    events: list[str] = []
    # deep mode, no tool call, and the question is about the material -> the
    # evidence gate blocks this very round
    runtime, session_id = _runtime(tmp_path, [LLMStep(content="编码器由六个相同层组成。")])
    runtime.run(session_id, "编码器几层？", on_event=lambda name, payload: events.append(name), stream=True)

    assert "answer_delta" in events, "the fake streams the round's text first"
    assert "answer_reset" in events
    assert "gate_block" in events
    assert events.index("answer_reset") < events.index("gate_block")


def test_gate_reset_is_stream_only(tmp_path):
    """The reset stays part of the streaming contract: with stream=False there
    are no deltas, so there is nothing to reset (and the old path is unchanged)."""

    events: list[str] = []
    runtime, session_id = _runtime(tmp_path, [LLMStep(content="编码器由六个相同层组成。")])
    runtime.run(session_id, "编码器几层？", on_event=lambda name, payload: events.append(name))
    assert "answer_reset" not in events
    assert events.count("gate_block") == 2  # max_gate_blocks, unchanged


# ------------------------------------------------------------- Q2 · deep mode and off-topic questions

def test_deep_mode_lets_a_non_material_question_finish(tmp_path):
    """Q2: deep mode means "search when the question is about the material".

    A greeting used to be structurally refused (measured: "你好" ->
    ``no_evidence_after_gate`` in 4.8s), which is wrong: there is nothing in the
    collection to search for.  It must finish, and it must carry the honest
    "answered without evidence" mark.
    """

    events: list[str] = []
    runtime, session_id = _runtime(tmp_path, [LLMStep(content="你好！我是 ReadMate。")])
    result = runtime.run(session_id, "你好", on_event=lambda name, payload: events.append(name), stream=True)

    assert "gate_block" not in events
    assert result.failure is None
    # The user-visible mark is the one that must be there: the answer was produced
    # without evidence, and the UI keys its "unsourced" styling off it.  The
    # internal ``deep_nonmaterial_pass`` state note only rides the failure payload.
    assert result.warnings.count("unverified_nonmaterial") == 1
    assert result.gate["searched"] is False


def test_deep_mode_still_blocks_a_material_question_without_evidence(tmp_path):
    """The hard gate keeps its meaning: a question about the material still
    cannot be answered without evidence."""

    runtime, session_id = _runtime(tmp_path, [LLMStep(content="编码器由六个相同层组成。")])
    result = runtime.run(session_id, "编码器几层？")
    assert result.failure == "no_evidence_after_gate"
    assert "unverified_nonmaterial" not in result.warnings
