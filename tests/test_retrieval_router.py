"""Dual-track retrieval router decisions (T5.0.2)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.adapter.route_planner import (
    TRACK_DIRECT,
    TRACK_PLANNED,
    RetrievalRouter,
    compute_signals,
)
from src.llm.fake import FakeLLMClient
from src.llm.types import LLMStep


class _RaisingLLM:
    def complete(self, messages, **kwargs):
        raise RuntimeError("router model unavailable")


class _StubPlanner:
    def plan(self, query: str):
        # D38: a plan is sub-queries + one English rendering; the keyword channel
        # that used to be fed from here is gone.
        return SimpleNamespace(
            subqueries=("sub-a", "sub-b"),
            english_query="english rendering",
        )


def test_llm_chooses_planned_track_and_exposes_plan_fields():
    payload = {
        "track": "planned",
        "subqueries": ["a", "b"],
        "english_query": "enc layers",
        "reason": "multi part",
    }
    router = RetrievalRouter(llm=FakeLLMClient([LLMStep(content=json.dumps(payload))]))
    decision = router.decide("orig")

    assert decision.track == TRACK_PLANNED
    assert decision.provider == "llm"
    assert decision.fallback is False
    dense = decision.dense_texts("orig")
    assert "a" in dense
    assert "enc layers" in dense
    assert not hasattr(decision, "lexical_query")


def test_llm_chooses_direct_track_uses_original_only():
    payload = {"track": "direct", "reason": "short fact"}
    router = RetrievalRouter(llm=FakeLLMClient([LLMStep(content=json.dumps(payload))]))
    decision = router.decide("orig")

    assert decision.track == TRACK_DIRECT
    assert decision.provider == "llm"
    assert decision.dense_texts("orig") == ["orig"]


def test_non_json_model_output_falls_back_to_rules():
    router = RetrievalRouter(llm=FakeLLMClient([LLMStep(content="not json")]))
    query = "What is attention? Also how many layers? And what is P100?"
    signals = compute_signals(query)
    decision = router.decide(query, signals=signals)

    assert decision.provider == "rules"
    assert decision.track == (TRACK_PLANNED if signals.wants_plan else TRACK_DIRECT)


def test_planned_with_all_empty_fields_falls_back_to_rules():
    payload = {
        "track": "planned",
        "subqueries": [],
        "english_query": "",
        "reason": "empty plan",
    }
    router = RetrievalRouter(llm=FakeLLMClient([LLMStep(content=json.dumps(payload))]))
    decision = router.decide("short fact")

    assert decision.provider == "rules"


def test_model_exception_falls_back_to_rules_without_bubbling():
    router = RetrievalRouter(llm=_RaisingLLM())
    decision = router.decide("short fact")

    assert decision.provider == "rules"
    assert decision.track in {TRACK_DIRECT, TRACK_PLANNED}
    # D37: a routing downgrade is reported, not silent.
    assert decision.fallback is True
    assert decision.as_dict()["fallback"] is True


def test_forced_direct_and_planned_modes():
    direct = RetrievalRouter(mode="direct", llm=FakeLLMClient([LLMStep(content="ignored")]))
    direct_decision = direct.decide("anything")
    assert direct_decision.provider == "forced"
    assert direct_decision.track == TRACK_DIRECT
    assert direct_decision.dense_texts("anything") == ["anything"]

    planned = RetrievalRouter(mode="planned", planner=_StubPlanner())
    planned_decision = planned.decide("anything")
    assert planned_decision.provider == "forced"
    assert planned_decision.track == TRACK_PLANNED
    assert planned_decision.subqueries == ("sub-a", "sub-b")
    assert planned_decision.english_query == "english rendering"
    assert "english rendering" in planned_decision.dense_texts("anything")


def test_invalid_mode_raises_value_error():
    with pytest.raises(ValueError):
        RetrievalRouter(mode="bad")


# ---------------------------------------------------------------- R6 · C5: signals actually reach the router


def test_previous_empty_searches_forces_wants_plan():
    """A single-clause / no-number / short query is ``direct`` by default; a
    prior empty search must flip ``wants_plan`` so the rules track escalates."""

    short = "编码器有多少层"
    assert not compute_signals(short, previous_empty_searches=0).wants_plan
    assert compute_signals(short, previous_empty_searches=1).wants_plan


def test_long_question_chars_is_tunable_and_decides_wants_plan():
    """D27: the "long question" threshold used to be a module constant
    (``_LONG_QUESTION_CHARS = 40``), so the only way to move it was a code edit.
    It is data now, and it still gates the rules track."""

    from src.config.settings import RouteSettings

    settings = RouteSettings(long_question_chars=8, min_clause_chars=4)
    signals = compute_signals("编码器有多少层", route_settings=settings)
    assert signals.question_chars == 7
    assert signals.long_question_chars == 8
    # 7 chars < 8 -> still direct; one character more flips the same rule.
    assert not signals.wants_plan
    assert compute_signals("编码器到底有多少层", route_settings=settings).wants_plan

    # The threshold reaches the router through its settings, not through code.
    router = RetrievalRouter(llm=None, settings=settings)
    assert router.decide("编码器有多少层").track == TRACK_DIRECT
    assert router.decide("编码器到底有多少层").track == TRACK_PLANNED


def test_adapter_search_accepts_mode_and_previous_empty_searches():
    """The adapter's public ``search`` surface now accepts both kwargs and
    defaults to the pre-R6 behaviour (mode="deep", streak=0)."""

    from src.adapter.rag_adapter import RagAdapterImpl

    class _EmptyRetriever:
        def retrieve_multi(self, *a, **k):
            return []

    adapter = RagAdapterImpl(
        retriever=_EmptyRetriever(),
        resolve_documents=lambda _cid: ["d1"],
        registry=SimpleNamespace(list_documents=lambda: [], get_document=lambda _d: None),
        router=RetrievalRouter(llm=None),
    )
    # Defaults: current behaviour preserved.
    hits = adapter.search("col", "编码器有多少层")
    assert hits == []
    assert adapter.last_route.track == TRACK_DIRECT
    # Explicit prior empties push into planned.
    adapter.search("col", "编码器有多少层", previous_empty_searches=1)
    assert adapter.last_route.track == TRACK_PLANNED


def test_rules_track_escalates_to_planned_after_first_empty_search(tmp_path):
    """C5 end-to-end via ToolRunner: with the same runner across two search
    calls, the first empty (previous_empty_searches=0 -> direct, 0 hits) must
    bump the streak so the next call's signals push the rule track to planned."""

    from src.adapter.rag_adapter import RagAdapterImpl
    from src.agent.tools import ToolRunner
    from src.config.settings import AgentSettings
    from src.llm.types import ToolCall

    class _EmptyRetriever:
        def retrieve_multi(self, *a, **k):
            return []

    adapter = RagAdapterImpl(
        retriever=_EmptyRetriever(),
        resolve_documents=lambda _cid: ["d1"],
        registry=SimpleNamespace(list_documents=lambda: [], get_document=lambda _d: None),
        router=RetrievalRouter(llm=None),
    )
    runner = ToolRunner(adapter=adapter, collections=None, memory=None, settings=AgentSettings())
    call = ToolCall(id="1", name="search", arguments={"query": "编码器有多少层"})

    first = runner._search(call, "col", "编码器有多少层")
    assert first.ok and not first.hits
    assert adapter.last_route.track == TRACK_DIRECT
    assert adapter.last_route.provider == "rules"

    second = runner._search(call, "col", "编码器有多少层")
    assert second.ok and not second.hits
    assert adapter.last_route.track == TRACK_PLANNED


def test_tool_search_passes_streak_and_mode_to_adapter():
    """The ToolRunner is what owns the counter and the session mode; verify
    both actually arrive at ``adapter.search`` (mock records the kwargs)."""

    from src.adapter.contracts import SearchHit
    from src.adapter.mock_adapter import MockRagAdapter
    from src.agent.tools import ToolRunner
    from src.config.settings import AgentSettings
    from src.llm.types import ToolCall

    adapter = MockRagAdapter(hits=[])
    runner = ToolRunner(adapter=adapter, collections=None, memory=None, settings=AgentSettings())
    call = ToolCall(id="1", name="search", arguments={"query": "anything"})
    for expected in (0, 1, 2, 3):
        runner._search(call, "col", "anything")
        assert adapter.search_calls[-1]["previous_empty_searches"] == expected


def test_tool_search_forwards_session_mode_from_db():
    from src.adapter.contracts import SearchHit
    from src.adapter.mock_adapter import MockRagAdapter
    from src.agent.tools import ToolRunner
    from src.config.settings import AgentSettings
    from src.llm.types import ToolCall

    hit = SearchHit(
        chunk_id="c1", document_id="d1", filename="a.pdf", page=1, score=0.9,
        chunk_type="prose", presentation_content="body", excerpt="body",
    )
    fake_db = SimpleNamespace(get_session=lambda sid: {"mode": "chat"})
    collections = SimpleNamespace(db=fake_db)
    adapter = MockRagAdapter(hits=[hit])
    runner = ToolRunner(adapter=adapter, collections=collections, memory=None, settings=AgentSettings())
    call = ToolCall(id="1", name="search", arguments={"query": "anything"})
    runner._search(call, "col", "anything", session_id="s1")
    assert adapter.search_calls[-1]["mode"] == "chat"
    # Non-empty hits keep the streak at zero.
    assert adapter.search_calls[-1]["previous_empty_searches"] == 0


def test_tool_search_swallows_session_lookup_failures():
    """A missing/unknown session must not derail routing; mode falls through
    to the adapter's ``deep`` default."""

    from src.adapter.mock_adapter import MockRagAdapter
    from src.agent.tools import ToolRunner
    from src.config.settings import AgentSettings
    from src.llm.types import ToolCall

    def _boom(_sid):
        raise KeyError("session missing")

    collections = SimpleNamespace(db=SimpleNamespace(get_session=_boom))
    adapter = MockRagAdapter(hits=[])
    runner = ToolRunner(adapter=adapter, collections=collections, memory=None, settings=AgentSettings())
    call = ToolCall(id="1", name="search", arguments={"query": "anything"})
    observation = runner._search(call, "col", "anything", session_id="s999")
    assert observation.ok
    assert adapter.search_calls[-1]["mode"] is None
