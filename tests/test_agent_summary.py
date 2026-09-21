"""批 D 阶段 2 (D32): the session summary service and its failure budget.

These run against a real :class:`AgentDB` on a temp path -- not a mock -- because the
service's contract *is* its storage side effect (``summary`` +
``summary_upto_message_id`` must move together, and a failed run must leave both
untouched).  A stub would happily "pass" while the real column stayed empty.
"""

from __future__ import annotations

import pytest

from src.agent.summary import (
    MAX_SUMMARY_FAILURES,
    SUMMARY_KEEP_RECENT_MESSAGES,
    SUMMARY_MAX_CHARS,
    SessionSummaryService,
    _clean_summary,
    _format_transcript,
)
from src.storage.agent_db import AgentDB
from src.tasks.summary import run_summary


class _FakeLLM:
    """Stands in for ``LLMClient``: records prompts, optionally fails."""

    def __init__(self, reply: str = "目标：读懂 Transformer 论文", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.reply


def _db(tmp_path) -> AgentDB:
    return AgentDB(tmp_path / "agent.db")


def _session(db: AgentDB, messages: int) -> str:
    session_id = db.create_session(collection_id=None)
    for index in range(messages):
        role = "user" if index % 2 == 0 else "assistant"
        db.append_message(session_id, role, f"第 {index} 条消息")
    return session_id


def _service(db: AgentDB, llm) -> SessionSummaryService:
    return SessionSummaryService(db=db, llm=llm)


# ------------------------------------------------------------------ migration
def test_migration_3_adds_the_failure_counter(tmp_path) -> None:
    """D32's breaker stores its count in the session row, so the column must exist."""

    db = _db(tmp_path)
    session_id = db.create_session(collection_id=None)
    assert db.summary_failures(session_id) == 0
    assert db.bump_summary_failure(session_id) == 1
    assert db.bump_summary_failure(session_id) == 2
    db.reset_summary_failures(session_id)
    assert db.summary_failures(session_id) == 0


# ------------------------------------------------------------------ planning
def test_short_session_is_not_due(tmp_path) -> None:
    db = _db(tmp_path)
    session_id = _session(db, SUMMARY_KEEP_RECENT_MESSAGES)
    assert _service(db, _FakeLLM()).summary_boundary(session_id) is None
    assert _service(db, _FakeLLM()).is_due(session_id) is False


def test_boundary_is_the_message_before_the_kept_tail(tmp_path) -> None:
    db = _db(tmp_path)
    session_id = _session(db, SUMMARY_KEEP_RECENT_MESSAGES + 5)
    rows = db.list_messages(session_id)
    expected = int(rows[-SUMMARY_KEEP_RECENT_MESSAGES - 1]["id"])
    assert _service(db, _FakeLLM()).summary_boundary(session_id) == expected


def test_already_covered_session_is_not_due(tmp_path) -> None:
    """The summary must not be regenerated over turns it already covers."""

    db = _db(tmp_path)
    session_id = _session(db, SUMMARY_KEEP_RECENT_MESSAGES + 5)
    service = _service(db, _FakeLLM())
    boundary = service.summary_boundary(session_id)
    db.set_session_summary(session_id, "旧摘要", boundary)
    assert service.summary_boundary(session_id) is None


# ------------------------------------------------------------------ building
def test_summarize_stores_text_and_pointer_together(tmp_path) -> None:
    db = _db(tmp_path)
    session_id = _session(db, SUMMARY_KEEP_RECENT_MESSAGES + 5)
    llm = _FakeLLM("目标：读懂论文\n已确认：attention-is-all-you-need.pdf 第3页 表2")
    outcome = _service(db, llm).summarize(session_id)
    assert outcome["stored"] is True
    session = db.get_session(session_id)
    assert session["summary"].startswith("目标：读懂论文")
    assert session["summary_upto_message_id"] == outcome["upto_message_id"]
    # The prompt must carry the dropped turns and the pointer rules.
    prompt = llm.prompts[0]
    assert "第 0 条消息" in prompt
    assert "NEVER restate document text" in prompt


def test_incremental_prompt_carries_the_previous_summary(tmp_path) -> None:
    """A second compression extends the old summary instead of re-reading everything."""

    db = _db(tmp_path)
    session_id = _session(db, SUMMARY_KEEP_RECENT_MESSAGES + 5)
    service = _service(db, _FakeLLM())
    first = service.summarize(session_id)
    assert first["stored"] is True

    db.append_message(session_id, "user", "接着问 4.1 节")
    db.append_message(session_id, "assistant", "4.1 节讲的是……")
    for _ in range(SUMMARY_KEEP_RECENT_MESSAGES):
        db.append_message(session_id, "user", "补充")
    llm = _FakeLLM("目标：继续")
    second = _service(db, llm).summarize(session_id)
    assert second["stored"] is True
    assert second["upto_message_id"] > first["upto_message_id"]
    assert "Existing summary to extend" in llm.prompts[0]


def test_empty_model_reply_is_reported_and_changes_nothing(tmp_path) -> None:
    db = _db(tmp_path)
    session_id = _session(db, SUMMARY_KEEP_RECENT_MESSAGES + 5)
    before = db.get_session(session_id)
    outcome = _service(db, _FakeLLM(reply="   ")).summarize(session_id)
    assert outcome["stored"] is False
    assert outcome["reason"] == "empty_summary"
    after = db.get_session(session_id)
    assert after["summary"] == before["summary"]
    assert after["summary_upto_message_id"] == before["summary_upto_message_id"]


def test_generation_error_is_reported_not_raised(tmp_path) -> None:
    db = _db(tmp_path)
    session_id = _session(db, SUMMARY_KEEP_RECENT_MESSAGES + 5)
    outcome = _service(db, _FakeLLM(error=RuntimeError("upstream 503"))).summarize(session_id)
    assert outcome["stored"] is False
    assert outcome["reason"] == "generation_failed"
    assert db.get_session(session_id)["summary"] is None


def test_nothing_to_summarize_on_a_short_session(tmp_path) -> None:
    db = _db(tmp_path)
    session_id = _session(db, 4)
    outcome = _service(db, _FakeLLM()).summarize(session_id)
    assert outcome == {"stored": False, "reason": "nothing_to_summarize", "upto_message_id": None}


# ------------------------------------------------------------------ breaker
def test_breaker_opens_after_the_failure_budget_and_force_bypasses_it(tmp_path) -> None:
    db = _db(tmp_path)
    session_id = _session(db, SUMMARY_KEEP_RECENT_MESSAGES + 5)
    failing = _service(db, _FakeLLM(error=RuntimeError("boom")))

    for _ in range(MAX_SUMMARY_FAILURES):
        outcome = run_summary(session_id, service=failing, db=db)
        assert outcome["stored"] is False
    assert db.summary_failures(session_id) == MAX_SUMMARY_FAILURES

    # The budget is spent: no further attempt, and no further model call.
    skipped = run_summary(session_id, service=failing, db=db)
    assert skipped["reason"] == "breaker_open"
    assert len(failing.llm.prompts) == MAX_SUMMARY_FAILURES

    # A human asking explicitly is not bound by the counter.
    forced = run_summary(session_id, service=_service(db, _FakeLLM("目标：手动")), db=db, force=True)
    assert forced["stored"] is True
    assert db.summary_failures(session_id) == 0


def test_nothing_to_summarize_does_not_count_as_a_failure(tmp_path) -> None:
    db = _db(tmp_path)
    session_id = _session(db, 4)
    outcome = run_summary(session_id, service=_service(db, _FakeLLM()), db=db)
    assert outcome["reason"] == "nothing_to_summarize"
    assert db.summary_failures(session_id) == 0


# ------------------------------------------------------------------ helpers
def test_clean_summary_strips_a_code_fence_and_caps_the_length() -> None:
    assert _clean_summary("```\n目标：X\n```") == "目标：X"
    long = "目" * (SUMMARY_MAX_CHARS + 50)
    cleaned = _clean_summary(long)
    assert len(cleaned) == SUMMARY_MAX_CHARS + 1  # the ellipsis replaces the tail
    assert cleaned.endswith("…")


def test_format_transcript_is_oldest_first_and_capped() -> None:
    rows = [
        {"role": "system", "content": "忽略我"},
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "  第一答  "},
    ]
    assert _format_transcript(rows) == "user: 第一问\nassistant: 第一答"
    assert len(_format_transcript([{"role": "user", "content": "x" * 9000}])) <= 6001
