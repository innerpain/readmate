"""The ReAct round ledger: what the agent eval reads for the trajectory layer.

A flat tool call list cannot answer "was that 9 calls in 3 rounds or in 9?", nor
"which round was closable before the budget ran out?" -- those two questions are
the whole point of instrumenting the loop, so they are pinned here.

A6 (2026-09-17): the same ledger must reach the DB on failure turns too, so an
audit can query ``payload_json -> '$.gate.searched'`` for every assistant row.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from src.adapter.contracts import ReadResult, SearchHit
from src.adapter.mock_adapter import MockRagAdapter
from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.agent.runtime import MODE_DEEP, AgentRuntime, _digest_tool_trace
from src.agent.tools import ToolRunner
from src.config.settings import get_agent_settings
from src.llm.fake import FakeLLMClient
from src.llm.types import LLMStep, ToolCall
from src.storage.agent_db import AgentDB


class _EmptyRegistry:
    def list_documents(self) -> list:
        return []


def _hit(chunk_id: str = "c1") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id="d1",
        filename="attention.pdf",
        page=3,
        section="3 Model Architecture",
        score=0.81,
        chunk_type="prose",
        presentation_content="encoders are a stack of six identical layers",
        excerpt="six identical layers",
    )


def _answer(chunk_id: str = "c1") -> str:
    return json.dumps({
        "answer": "The encoder has six identical layers.",
        "citations": [{"chunk_id": chunk_id, "page": 3, "quote": "six identical layers"}],
        "non_source": [],
    })


def _search_call(call_id: str = "1") -> LLMStep:
    return LLMStep(tool_calls=[ToolCall(id=call_id, name="search", arguments={"query": "encoder layers"})])


def _build(tmp_path: Path, script: list[LLMStep], *, max_steps: int | None = None):
    settings = get_agent_settings()
    if max_steps is not None:
        settings = replace(settings, max_steps=max_steps)
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("col")
    db.add_documents(collection_id, ["d1"])
    adapter = MockRagAdapter(hits=[_hit()])
    collections = CollectionService(db=db, registry=_EmptyRegistry())
    memory = MemoryService(db=db, collections=collections, settings=settings)
    runner = ToolRunner(adapter=adapter, collections=collections, memory=memory, settings=settings)
    runtime = AgentRuntime(llm=FakeLLMClient(script), tools=runner, db=db, settings=settings)
    session_id = db.create_session(collection_id=collection_id, mode=MODE_DEEP)
    return runtime, session_id


def test_each_loop_round_is_recorded_with_its_calls_and_closability(tmp_path):
    runtime, session_id = _build(tmp_path, [_search_call(), LLMStep(content=_answer())])
    result = runtime.run(session_id, "编码器有多少层？")

    assert result.failure is None
    assert [record["round"] for record in result.rounds] == [1, 2]
    assert result.rounds[0]["tool_calls"] == ["search"]
    assert result.rounds[0]["final_answer"] is False
    assert result.rounds[1]["tool_calls"] == []
    assert result.rounds[1]["final_answer"] is True
    assert all(record["duration_s"] >= 0 for record in result.rounds)
    assert result.rounds[1]["started_offset_s"] >= result.rounds[0]["started_offset_s"]


def test_tool_trace_carries_the_round_and_the_timing(tmp_path):
    runtime, session_id = _build(tmp_path, [_search_call(), LLMStep(content=_answer())])
    result = runtime.run(session_id, "编码器有多少层？")

    step = result.tool_trace[0]
    assert step["tool"] == "search"
    assert step["round"] == 1
    assert step["duration_s"] is not None
    assert step["offset_s"] is not None
    # the keys the existing gate test relies on must survive instrumentation
    assert {"tool", "arguments", "ok", "error", "route"} <= set(step)


def test_gate_keys_are_unchanged_by_instrumentation(tmp_path):
    """tests/test_agent_runtime_gate.py compares `gate` with ==: never add keys here."""

    runtime, session_id = _build(tmp_path, [_search_call(), LLMStep(content=_answer())])
    result = runtime.run(session_id, "编码器有多少层？")
    assert set(result.gate) == {"blocks", "searched", "read"}


def test_multiple_calls_in_one_round_stay_in_one_round(tmp_path):
    """9 calls can be 3 rounds: the two counts must not be conflated."""

    script = [
        LLMStep(tool_calls=[
            ToolCall(id="1", name="search", arguments={"query": "encoder"}),
            ToolCall(id="2", name="search", arguments={"query": "decoder"}),
        ]),
        LLMStep(content=_answer()),
    ]
    runtime, session_id = _build(tmp_path, script)
    result = runtime.run(session_id, "编码器有多少层？")

    assert len(result.tool_trace) == 2
    assert len(result.rounds) == 2
    assert result.rounds[0]["tool_call_count"] == 2


def test_step_budget_exhaustion_is_visible_as_rounds_without_a_closable_one(tmp_path):
    settings = get_agent_settings()
    script = [_search_call(str(index)) for index in range(settings.max_steps)]
    runtime, session_id = _build(tmp_path, script)
    result = runtime.run(session_id, "编码器有多少层？")

    assert result.failure == "max_steps_without_answer"
    assert len(result.rounds) == settings.max_steps
    assert not any(record["final_answer"] for record in result.rounds)
    assert result.gate["searched"] is True      # evidence was found; only the closing round was missing


def test_dropped_citations_are_exposed_not_just_flagged(tmp_path):
    runtime, session_id = _build(tmp_path, [_search_call(), LLMStep(content=_answer(chunk_id="cX"))])
    result = runtime.run(session_id, "编码器有多少层？")

    assert result.citations == []
    assert result.dropped_citations == ["cX"]
    assert "citations_dropped_unobserved" in result.warnings


def test_dropped_citations_is_empty_on_a_clean_turn(tmp_path):
    runtime, session_id = _build(tmp_path, [_search_call(), LLMStep(content=_answer())])
    result = runtime.run(session_id, "编码器有多少层？")

    assert result.dropped_citations == []
    assert len(result.citations) == 1


def test_failure_path_also_carries_the_round_ledger(tmp_path):
    settings = get_agent_settings()
    runtime, session_id = _build(tmp_path, [_search_call(str(index)) for index in range(settings.max_steps)])
    result = runtime.run(session_id, "编码器有多少层？")

    assert result.rounds
    assert result.dropped_citations == []


# ------------------------------------------------------- the persisted envelope


def _assistant_payload(db: AgentDB, session_id: str) -> dict:
    """The last assistant row's payload_json -- what an audit actually reads."""

    rows = [row for row in db.list_messages(session_id) if row["role"] == "assistant"]
    assert rows, "the turn wrote no assistant message"
    assert len(rows) == 1, "one turn must persist exactly one assistant row"
    return json.loads(rows[-1]["payload_json"])


def test_a_failed_turn_is_auditable_from_the_db(tmp_path):
    """A6: payload_json used to be only {"failure": reason}, so 7 live failures
    could not answer "did gate.searched fire?" without re-running the turn."""

    settings = get_agent_settings()
    runtime, session_id = _build(tmp_path, [_search_call(str(index)) for index in range(settings.max_steps)])
    result = runtime.run(session_id, "编码器有多少层？")

    payload = _assistant_payload(runtime.db, session_id)
    assert payload["failure"] == "max_steps_without_answer"
    # the whole point: this key used to be structurally absent on failure rows
    assert payload["gate"] == {"blocks": 0, "searched": True, "read": False}
    assert payload["gate"] == result.gate
    assert [record["round"] for record in payload["rounds"]] == list(range(1, settings.max_steps + 1))
    assert payload["observed_chunk_ids"] == ["c1"]
    assert payload["warnings"] == ["max_steps_without_answer"]
    assert [call["tool"] for call in payload["tool_trace_digest"]] == ["search"] * settings.max_steps


def test_the_gate_failure_lands_the_same_envelope(tmp_path):
    """`rounds` is empty on this path (the model called nothing), so the ledger
    has to be there even when nothing is in it."""

    runtime, session_id = _build(
        tmp_path, [LLMStep(content=_answer()), LLMStep(content=_answer())]
    )
    runtime.run(session_id, "编码器有多少层？")

    payload = _assistant_payload(runtime.db, session_id)
    assert payload["failure"] == "no_evidence_after_gate"
    assert payload["gate"]["searched"] is False
    assert [record["gate_block"] for record in payload["rounds"]] == [
        "deep_mode_requires_evidence",
        "deep_mode_requires_evidence",
    ]


def test_a_successful_turn_carries_the_same_envelope(tmp_path):
    runtime, session_id = _build(tmp_path, [_search_call(), LLMStep(content=_answer())])
    result = runtime.run(session_id, "编码器有多少层？")

    payload = _assistant_payload(runtime.db, session_id)
    assert payload["failure"] is None
    assert payload["gate"] == result.gate
    assert payload["rounds"] == result.rounds
    assert payload["observed_chunk_ids"] == ["c1"]
    # what the frontend already reads must not move or change type
    assert [citation["chunk_id"] for citation in payload["citations"]] == ["c1"]
    assert payload["warnings"] == result.warnings


def test_the_persisted_trace_is_bounded_but_the_response_trace_is_not(tmp_path):
    """Volume control: the DB keeps the last 20 calls with a capped
    ``arguments``; the HTTP response keeps every call untouched."""

    huge = "x" * 5000
    runtime, session_id = _build(
        tmp_path,
        [
            LLMStep(tool_calls=[ToolCall(id="1", name="search", arguments={"query": huge})]),
            LLMStep(content=_answer()),
        ],
    )
    result = runtime.run(session_id, "编码器有多少层？")

    payload = _assistant_payload(runtime.db, session_id)
    assert len(payload["tool_trace_digest"][0]["arguments"]) <= 500
    assert result.tool_trace[0]["arguments"] == {"query": huge}


def test_the_digest_keeps_the_most_recent_calls(tmp_path):
    observations = [
        {"tool": "search", "arguments": {"query": f"q{index}"}, "ok": True, "error": None, "route": None}
        for index in range(25)
    ]

    digested = _digest_tool_trace(observations)

    assert len(digested) == 20
    assert json.loads(digested[0]["arguments"]) == {"query": "q5"}
    assert json.loads(digested[-1]["arguments"]) == {"query": "q24"}

# ------------------------------------------------------- R5: wrap-up + honest failure


def test_last_round_is_forced_to_answer_without_tools(tmp_path):
    """A2: the final budget step is reserved for answering -- no tool specs are
    offered, so the model cannot spend it on a search it would never see."""

    runtime, session_id = _build(
        tmp_path, [_search_call(), LLMStep(content=_answer())], max_steps=2
    )
    result = runtime.run(session_id, "编码器有多少层？")

    assert result.failure is None
    assert result.rounds[-1]["final_answer"] is True
    # Round 1 saw the tool specs; the last round saw none.
    assert runtime.llm.calls[0]["tools"]
    assert runtime.llm.calls[-1]["tools"] == []


def test_budget_exhaustion_with_evidence_is_not_reported_as_no_evidence(tmp_path):
    """A1/A2 honesty: when evidence was gathered but the loop never closed, the
    failure text must not claim nothing was retrieved."""

    runtime, session_id = _build(
        tmp_path,
        [_search_call("1"), _search_call("2")],
        max_steps=2,
    )
    result = runtime.run(session_id, "编码器有多少层？")

    assert result.failure == "max_steps_without_answer"
    assert result.gate["searched"] is True
    assert "没有取到资料证据" not in result.answer
    assert result.citations == []


def _page_read_call(call_id: str = "2") -> LLMStep:
    """A page read: the model names a document and a page, no chunk_id."""

    return LLMStep(
        tool_calls=[ToolCall(id=call_id, name="read", arguments={"document_id": "d1", "page": 3})]
    )


def test_a_page_read_makes_every_row_citable(tmp_path):
    """D57: a page read must leave the model holding evidence it can cite.

    The eval's ``read_by_page_contract_hole`` used to flag *any* page-based read while
    quoting a hole A3 had already closed: ``_read_page`` returns the page's rows'
    ``chunk_ids``, and every one of them becomes observable.  Pin the real contract --
    all of a read observation's ids reach ``observed_chunks``, and the step records
    which ones it contributed (that per-step record is what lets the eval tell a closed
    hole from a real one instead of guessing).
    """

    settings = get_agent_settings()
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("col")
    db.add_documents(collection_id, ["d1"])
    # a page read: no chunk_id of its own, three citable rows
    page = ReadResult(
        document_id="d1",
        chunk_id=None,
        chunk_ids=["c1", "c2", "c3"],
        page=3,
        text="a whole page of prose",
    )
    adapter = MockRagAdapter(hits=[_hit()], reads={"c1": page})
    collections = CollectionService(db=db, registry=_EmptyRegistry())
    memory = MemoryService(db=db, collections=collections, settings=settings)
    runner = ToolRunner(adapter=adapter, collections=collections, memory=memory, settings=settings)
    runtime = AgentRuntime(
        llm=FakeLLMClient([_page_read_call(), LLMStep(content=_answer())]),
        tools=runner,
        db=db,
        settings=settings,
    )
    session_id = db.create_session(collection_id=collection_id, mode=MODE_DEEP)

    result = runtime.run(session_id, "第 3 页讲了什么？")

    step = next(record for record in result.tool_trace if record["tool"] == "read")
    assert step["arguments"].get("chunk_id") is None, "this is meant to be a page read"
    assert step["observed"] == ["c1", "c2", "c3"], "every row's id must be recorded as contributed"
    assert {"c1", "c2", "c3"} <= set(result.observed_chunk_ids)
