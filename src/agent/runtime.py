"""ReadMate ReAct runtime: mode gate, tool loop, answer assembly.

The gate is the point of the whole module: in deep-read mode a turn may not be
finished without a successful search or read, and the check is on tool success
(not on how politely the model words its answer).

Instrumentation note: the tool loop records one entry per round (``RunState.rounds``)
and a timestamp per tool call, because a flat call list cannot answer the two
questions an evaluation actually asks -- "9 calls: was that 3 rounds or 9?" and
"which round was closable before the budget ran out?".  Nothing here changes
behaviour: same calls, same order, same gate, same answer validation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from src.agent.answer_protocol import parse_agent_answer, validate_citations
from src.agent.relevance import is_related
from src.agent.token_budget import (
    PROMPT_PRESSURE_ALERT_RATIO,
    estimate_prompt_tokens,
    history_budget_tokens,
    select_history,
)
from src.agent.tools import Observation, ToolRunner
from src.llm.types import ToolCall, ToolLLM
from src.storage.agent_db import session_scope

MODE_DEEP = "deep"
MODE_CHAT = "chat"

# The frozen per-session system prompt carries this version; bumping it makes the
# next turn rebuild every snapshot (see ``_prompt_snapshot``).
#   1 - initial snapshot (rules then material)
#   2 - material before rules: the JSON envelope must stay the last thing read,
#       otherwise chat-mode greetings drift to plain prose (``answer_not_json``)
#   4 - memory contract added to the frozen rules (explicit request -> write now,
#       inferred -> ask once); 3 was the history note
PROMPT_SNAPSHOT_VERSION = 4

# 2026-09-19 用户口径（M1=c / M2=a）：记忆的写入语义要跟"谁发起"绑定，写进冻结规则段。
MEMORY_CONTRACT = (
    "Memory rules. (a) When the user explicitly asks you to remember something "
    "(\"记住…\", \"记下…\", \"以后都…\"), call memory_note with explicit=true: it is written to "
    "long-term memory immediately, and your answer must say it is saved -- never that it "
    "awaits confirmation. (b) When you merely infer a stable fact about the user (preference, "
    "profile, learning progress), call memory_note WITHOUT explicit -- it stays pending -- and "
    "ask the user once, in the same answer, whether to save it. If they say yes, call memory_note "
    "again with explicit=true and the same key so it is written straight away. Never ask about "
    "the same fact twice, and never ask about something the user did not raise unless this turn "
    "carries a memory-review note."
)

# 每 N 轮（``memory_review_every_turns``）发一次的回顾提示。**不进快照**：它是可变的，
# 混进冻结前缀会让前缀漂移（第 7 条的教训）。
MEMORY_REVIEW_INSTRUCTION = (
    "Memory review (periodic). Look back over this conversation: if you have picked up a stable, "
    "durable fact about the user (profile / preference / learning progress) that is not recorded "
    "yet, call memory_note once for it and ask the user whether to save it. If there is nothing "
    "worth recording, ignore this note and answer the question normally."
)

# 明确记忆指令的正则（**只看用户消息**，不看模型输出，避免"我记得论文里…"误命中）。
# 只收祈使/宣告式说法；"记得"单独出现不算（"我记得…"太常见），"记不住"也不会命中。
_EXPLICIT_MEMORY_RE = re.compile(
    r"(请?记住|帮我记住|给我记住|要记住|记一下|记下来|记下|记牢|记着|别忘了|别忘记|"
    r"以后都|之后都|每次都|下次都|永远都|"
    r"remember (this|that)|keep (this|that) in mind|always (use|answer|reply))",
    re.IGNORECASE,
)


def _is_explicit_memory_request(text: str) -> bool:
    """True when the *user's own message* explicitly asks for something to be remembered.

    This is the authoritative half of M1=c: deterministic, testable, and immune to the
    model forgetting to set ``explicit=true``.  The model's flag stays as a supplement
    for phrasings this regex does not know.
    """

    return bool(_EXPLICIT_MEMORY_RE.search(text or ""))

# Replayed history carries the *answer text* of earlier turns, so the model has to be
# told that this display form does not change the reply contract.  Frozen into the
# snapshot: costs nothing per turn and cannot make the prefix drift.
SNAPSHOT_HISTORY_NOTE = (
    "Earlier turns are replayed as plain text; that is a display format, not a style "
    "guide. Every reply of yours must still be the JSON object described above, "
    "follow-ups included."
)

# Sent immediately before the live question on every turn (see ``_build_messages``).
REPLY_CONTRACT_REMINDER = (
    'Reply with the JSON object only: {"answer": "...", "citations": [{"chunk_id": "...", '
    '"page": 1, "quote": "..."}], "non_source": [...], "refused": false}. '
    "No prose outside the JSON."
)

GATE_BLOCK_DEEP = "deep_mode_requires_evidence"
GATE_BLOCK_CHAT = "chat_related_requires_search"


def _noop_event(_name: str, _payload: dict) -> None:
    """Default on_event: streaming off costs nothing."""

NO_EVIDENCE_MESSAGE = "本轮没有取到资料证据，无法给出有出处的回答。"
_EVIDENCE_UNCONVERGED_MESSAGE = "本轮已取到资料证据，但未能在步数预算内收敛出答案；可重试或缩小问题范围。"
_TOOL_ERROR_MESSAGE = "工具连续返回错误，本轮未能取到可用的资料证据。"


@dataclass
class RunState:
    searched: bool = False
    read: bool = False
    blocks: int = 0
    # Deep mode let a turn finish without evidence, but only because the question
    # was judged unrelated to the material (see ``_gate``).  Kept so the answer
    # can carry an honest "unverified" mark instead of looking sourced.
    nonmaterial_pass: bool = False
    # 问题.md 第二轮第 8 条: the request asked for different material than the
    # session is locked to.  The session wins; the mismatch is reported, not hidden.
    scope_overridden: bool = False
    # 2026-09-19 (M3=a): a memory_note written straight to the long-term tables
    # because the user explicitly asked for it.  Surfaced as a warning so the turn
    # shows *why* nothing is waiting for confirmation.
    memory_auto_written: bool = False
    observations: list[dict] = field(default_factory=list)
    observed_chunks: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return self.searched or self.read


@dataclass
class AgentRunResult:
    answer: str
    citations: list[dict]
    non_source: list[str]
    warnings: list[str]
    tool_trace: list[dict]
    mode: str
    gate: dict
    refused: bool = False
    failure: str | None = None
    # Additive fields.  ``dropped_citations`` surfaces what used to be a bare
    # boolean warning ("citations_dropped_unobserved"): the model proposing
    # citations for chunks no tool observed is the hallucination signal.
    rounds: list[dict] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)
    # D57: which chunks a tool actually put in front of the model.  The eval needs
    # this to tell a *closed* page-read hole (ids present -> citable) from a real one
    # (a page read whose ids never became observable).  Same list the persisted
    # payload already carried.
    observed_chunk_ids: list[str] = field(default_factory=list)


class AgentRuntime:
    def __init__(self, *, llm: ToolLLM, tools: ToolRunner, db, settings) -> None:
        self.llm = llm
        self.tools = tools
        self.db = db
        self.settings = settings

    # ------------------------------------------------------------------ public
    def run(
        self,
        session_id: str,
        user_text: str,
        mode: str | None = None,
        *,
        on_event=None,
        stream: bool = False,
        collection_ids: list[str] | None = None,
    ) -> AgentRunResult:
        """One ReAct turn.  ``stream=True`` + an LLM that supports
        ``stream_complete`` additionally emits events through ``on_event``
        (SSE protocol, frontend plan §5).  Default arguments keep every
        existing caller and test byte-for-byte on the old path.

        ``collection_ids`` is the R7 (D-5b) per-message scope: a multi-collection
        turn searches the union of those collections and never persists them to
        the session.  When omitted, the session's own ``collection_id`` still
        governs, so single-collection behaviour is unchanged.
        """

        emit = on_event if on_event is not None else _noop_event
        session = self.db.get_session(session_id)
        mode = mode or session.get("mode") or self.settings.default_mode
        session_collection = session.get("collection_id")
        requested = [cid for cid in (collection_ids or []) if cid] or ([session_collection] if session_collection else [])
        state = RunState()
        # 问题.md 第二轮第 8 条: the material scope belongs to the *conversation*,
        # not to a single message.  The first turn decides it and locks it; later
        # turns use the session's scope even when the request asks for another one
        # (recorded as ``scope_overridden`` so the mismatch is visible, never silent).
        locked_scope = session_scope(session)
        if locked_scope is None:
            scope_ids = requested
            self.db.set_session_scope(session_id, scope_ids)
        else:
            scope_ids = locked_scope
            if requested and set(requested) != set(locked_scope):
                state.scope_overridden = True
                state.warnings.append("scope_overridden")
        collection_id = session_collection  # representative: legacy gate/message plumbing
        self.db.append_message(session_id, "user", user_text)

        # 2026-09-19 记忆写入语义（M1=c / M2=a）：显式要求 → 直写；模型自己推断的 →
        # 每 N 轮回顾一次（``memory_review_every_turns``），不再每轮试探。
        explicit_memory = _is_explicit_memory_request(user_text)
        memory_review = self._memory_review_due(session_id)

        messages = self._build_messages(session, scope_ids, mode, user_text, memory_review=memory_review)
        clock = time.perf_counter()

        for round_index in range(1, self.settings.max_steps + 1):
            round_started = time.perf_counter()
            streamed_this_round = False
            last_round = round_index >= self.settings.max_steps
            if last_round:
                # A2: reserve the final step for answering.  With no tool specs
                # the model cannot burn the last round on a search whose result
                # it would never see (the Q17 dead-loop); the evidence gate
                # still runs on whatever answer it produces below.
                messages.append({"role": "system", "content": _answer_now_instruction()})
            specs = [] if last_round else self.tools.specs()
            if stream and hasattr(self.llm, "stream_complete"):
                def _on_delta(text: str, _round: int = round_index) -> None:
                    nonlocal streamed_this_round
                    streamed_this_round = True
                    emit("answer_delta", {"round": _round, "text": text})

                step = self.llm.stream_complete(messages, tools=specs, on_delta=_on_delta)
            else:
                step = self.llm.complete(messages, tools=specs)

            if step.tool_calls:
                # Optimistic streaming: text emitted before the round turned
                # out to be a tool round belongs to no answer (plan §5).
                if streamed_this_round:
                    emit("answer_reset", {"round": round_index})
                messages.append(_assistant_tool_message(step.tool_calls, step.content))
                for call_index, call in enumerate(step.tool_calls):
                    emit("tool_call", {"round": round_index, "call_index": call_index, "tool": call.name, "arguments": call.arguments})
                    call_started = time.perf_counter()
                    observation = self.tools.run(call, collection_id=collection_id, session_id=session_id, user_text=user_text, collection_ids=scope_ids, explicit_memory=explicit_memory)
                    call_duration = round(time.perf_counter() - call_started, 3)
                    self._absorb(
                        call,
                        observation,
                        state,
                        round_index=round_index,
                        offset_s=round(call_started - clock, 3),
                        duration_s=call_duration,
                    )
                    emit(
                        "tool_result",
                        {
                            "round": round_index,
                            "call_index": call_index,
                            "tool": call.name,
                            "ok": observation.ok,
                            "hits": len(observation.hits) if observation.hits else None,
                            "error": observation.error_code,
                            "duration_s": call_duration,
                            "route": observation.route,
                        },
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": observation.content,
                        }
                    )
                state.rounds.append(
                    _round_record(
                        round_index,
                        calls=[call.name for call in step.tool_calls],
                        started_offset_s=round(round_started - clock, 3),
                        duration_s=round(time.perf_counter() - round_started, 3),
                        final_answer=False,
                    )
                )
                continue

            block = self._gate(mode, scope_ids, user_text, state)
            if block:
                # A9: whatever this round streamed is not an answer -- the gate is
                # about to refuse it.  Reset it exactly like a tool round does;
                # otherwise the UI keeps showing a confident-looking reply and
                # only ``turn_end`` replaces it with the refusal, which reads as
                # the answer being silently taken away.
                if streamed_this_round:
                    emit("answer_reset", {"round": round_index})
                state.blocks += 1
                state.warnings.append(block)
                emit("gate_block", {"round": round_index, "block": block, "blocks": state.blocks})
                state.rounds.append(
                    _round_record(
                        round_index,
                        calls=[],
                        started_offset_s=round(round_started - clock, 3),
                        duration_s=round(time.perf_counter() - round_started, 3),
                        final_answer=False,
                        gate_block=block,
                    )
                )
                if mode == MODE_DEEP and state.blocks >= self.settings.max_gate_blocks:
                    return self._failure(session_id, mode, state, "no_evidence_after_gate")
                messages.append({"role": "system", "content": _retry_instruction(block, mode)})
                continue

            state.rounds.append(
                _round_record(
                    round_index,
                    calls=[],
                    started_offset_s=round(round_started - clock, 3),
                    duration_s=round(time.perf_counter() - round_started, 3),
                    final_answer=True,
                )
            )
            return self._finish(session_id, mode, state, step.content)

        return self._failure(session_id, mode, state, "max_steps_without_answer")

    # --------------------------------------------------------------- internals
    def _memory_review_due(self, session_id: str) -> bool:
        """每 N 个**用户轮**发一次记忆回顾提示（``memory_review_every_turns``）。

        Counts user turns, not messages: a turn with three tool rounds would otherwise
        advance the cadence three times and the review would fire far too often.  Called
        after the user message is appended, so the current turn is included.  ``0``
        disables the periodic review entirely.
        """

        every = int(getattr(self.settings, "memory_review_every_turns", 0) or 0)
        if every <= 0:
            return False
        try:
            turns = self.db.count_messages(session_id, role="user")
        except Exception:  # pragma: no cover - defensive, cadence must never break a turn
            return False
        return turns > 0 and turns % every == 0

    def _history_budget_tokens(self) -> int:
        """D32: the history budget for this turn's model (context_length * ratio)."""

        context = int(getattr(getattr(self.llm, "profile", None), "context_length", 0) or 0)
        return history_budget_tokens(context, self.settings.history_context_ratio)

    def _note_prompt_pressure(self, messages: list[dict], session_id: str) -> None:
        """D32 阶段 3 (partial): shout when the assembled prompt nears the window.

        The ratio deliberately leaves room for evidence and the answer; this is the
        tripwire that says the assumption stopped holding (a table-heavy session, a
        much larger context_length than declared).
        """

        context = int(getattr(getattr(self.llm, "profile", None), "context_length", 0) or 0)
        if context <= 0:
            return
        used = sum(estimate_prompt_tokens(str(m.get("content") or "")) for m in messages)
        if used > context * PROMPT_PRESSURE_ALERT_RATIO:
            logger.warning(
                "prompt pressure: ~%d tokens of a %d-token window (%.0f%%) for session %s",
                used,
                context,
                100.0 * used / context,
                session_id,
            )

    def _build_messages(self, session: dict, scope_ids: list[str], mode: str, user_text: str, *, memory_review: bool = False) -> list[dict]:
        """Frozen snapshot first, then a real multi-turn history.

        问题.md 第二轮第 7 条.  The old assembly spliced the rules, the memory
        digest, the collection listing, the summary *and* the whole transcript into
        one system string, so the prefix changed on every turn and the model read
        the transcript as instructions rather than as turns.  Now:

          * the system prompt is a per-session snapshot (``_prompt_snapshot``),
            written once and reused byte-for-byte -- stable prefix, reproducible;
          * the session's material is *declared* in that snapshot the way a tool
            schema is declared, so the model knows its scope before searching;
          * only genuinely volatile context (memory digest, conversation summary)
            rides in a second system block *after* the snapshot;
          * history goes back to being real ``user``/``assistant`` messages.
        """

        messages: list[dict] = [
            {"role": "system", "content": self._prompt_snapshot(session, scope_ids, mode)}
        ]
        volatile: list[str] = []
        digest = self.tools.memory.digest(None, collection_ids=scope_ids)
        if digest:
            volatile.append(f"Learner context:\n{digest}")
        summary = session.get("summary")
        if summary:
            volatile.append(f"Earlier conversation summary:\n{summary}")
        if volatile:
            messages.append({"role": "system", "content": "\n\n".join(volatile)})
        # D32 阶段 1 (2026-09-20): admit history by token budget, not by turn count.
        # The fixed 12-message window cut conversations at ~8% of a 32k window and
        # had no compensation for what it dropped; ``history_turns`` is only a floor
        # now (at least that many turns survive even if they exceed the budget).
        floor_messages = max(1, self.settings.history_turns * 2)
        # Read a generous window from the DB and let the budget decide: the extra
        # rows are cheap to load and are discarded without ever reaching the model.
        history = self.db.recent_messages(str(session["id"]), max(floor_messages, 200))
        history = select_history(
            history,
            budget_tokens=self._history_budget_tokens(),
            floor_messages=floor_messages,
        )
        for row in history[:-1]:
            if row["role"] in {"user", "assistant"} and row.get("content"):
                messages.append({"role": row["role"], "content": row["content"]})
        # The reply contract goes last, right before the live question.  Real turns in
        # the history make the model imitate its own earlier *prose*; the frozen note
        # in the snapshot is not enough on its own (measured 2026-09-19: the follow-up
        # still came back as ``answer_not_json``/``answer_recovered_partial``).  A
        # reminder here does not disturb the prefix -- it sits after the history.
        # The periodic memory review rides in the same slot (before the contract, so
        # the JSON reminder still has the last word).
        if memory_review:
            messages.append({"role": "system", "content": MEMORY_REVIEW_INSTRUCTION})
        messages.append({"role": "system", "content": REPLY_CONTRACT_REMINDER})
        messages.append({"role": "user", "content": user_text})
        self._note_prompt_pressure(messages, str(session.get("id") or ""))
        return messages

    def summary_due(self, session_id: str) -> bool:
        """批 D (D32): whether older turns fell out of the window uncovered.

        The window keeps the newest ``history_turns * 2`` messages (or as many as
        the token budget allows); anything older than that is invisible to the
        model unless a summary covers it.  This is the cheap gate the caller uses
        before paying for a summary job -- the authoritative boundary check lives
        in :meth:`SessionSummaryService.summary_boundary`.

        Returns ``False`` on any error: a summary is an optimisation, never a
        reason to fail a turn.
        """

        try:
            session = self.db.get_session(session_id)
            if not session:
                return False
            covered = int(session.get("summary_upto_message_id") or 0)
            floor_messages = max(1, self.settings.history_turns * 2)
            recent = self.db.recent_messages(session_id, max(floor_messages, 200))
            if len(recent) <= floor_messages:
                return False
            oldest = int(recent[0]["id"])
            return oldest - 1 > covered
        except Exception as error:  # noqa: BLE001 - optimisation only
            logger.warning("summary_due check failed for %s: %s", session_id, error)
            return False

    def _prompt_snapshot(self, session: dict, scope_ids: list[str], mode: str) -> str:
        """The session's frozen system prompt: built on the first turn, reused after.

        A session's mode never changes after creation (nothing calls
        ``set_session_mode``), so freezing the mode-dependent rules is safe; bumping
        ``PROMPT_SNAPSHOT_VERSION`` rebuilds every stale snapshot.
        """

        existing = session.get("system_prompt")
        stored_version = int(session.get("prompt_snapshot_version") or 0)
        if isinstance(existing, str) and existing.strip() and stored_version == PROMPT_SNAPSHOT_VERSION:
            return existing
        snapshot = self._build_prompt_snapshot(session, scope_ids, mode)
        if hasattr(self.db, "set_session_prompt_snapshot"):
            self.db.set_session_prompt_snapshot(str(session["id"]), snapshot, PROMPT_SNAPSHOT_VERSION)
        return snapshot

    def _build_prompt_snapshot(self, session: dict, scope_ids: list[str], mode: str) -> str:
        """Material first, rules last.

        The JSON envelope is a contract the model must obey, so it stays at the very
        end of the prompt exactly as it was before this change.  Putting the material
        block after it measurably drifted chat-mode greetings into plain prose
        (``answer_not_json`` -- measured on 2026-09-19), i.e. out of the envelope.
        """

        parts = [self._scope_section(scope_ids), _system_prompt(mode), MEMORY_CONTRACT, SNAPSHOT_HISTORY_NOTE]
        return "\n\n".join(part for part in parts if part)

    def _scope_section(self, scope_ids: list[str]) -> str:
        """Declare the session's material up front (第 7 条).

        An empty ``scope_ids`` means "the whole library", which the snapshot states
        explicitly -- the model should never have to discover its own scope by
        searching, and it must never be surprised by a scope change mid-dialogue.
        """

        names: list[str] = []
        try:
            known = {str(c["id"]): str(c.get("name") or c["id"]) for c in self.db.list_collections()}
        except Exception:  # noqa: BLE001 - a listing failure must not block a turn
            known = {}
        for collection_id in scope_ids:
            names.append(f"{known.get(collection_id, collection_id)} ({collection_id})")
        lines = ["Session material (fixed when this conversation started; never changed mid-dialogue):"]
        lines.append(f"collections: {', '.join(names)}" if names else "collections: the whole library")
        digest_fn = getattr(self.tools.collections, "digest", None)
        digest = digest_fn(scope_ids) if callable(digest_fn) else ""
        if digest:
            lines.append(digest)
        return "\n".join(lines)

    def _absorb(
        self,
        call: ToolCall,
        observation: Observation,
        state: RunState,
        *,
        round_index: int | None = None,
        offset_s: float | None = None,
        duration_s: float | None = None,
    ) -> None:
        state.observations.append(
            {
                "tool": call.name,
                "arguments": call.arguments,
                "ok": observation.ok,
                "error": observation.error_code,
                "route": observation.route,
                "round": round_index,
                "offset_s": offset_s,
                "duration_s": duration_s,
            }
        )
        if not observation.ok:
            return
        if call.name == "search":
            state.searched = state.searched or bool(observation.hits)
        if call.name == "read":
            state.read = state.read or bool(observation.read and str(observation.read.get("text") or "").strip())
        # M3=a: an explicit "记住…" turn writes straight through; mark it so the turn
        # reports the write instead of looking like it silently skipped confirmation.
        # Deliberately *not* pushed into ``state.warnings``: that list also feeds the
        # chat-mode ``unverified_no_search`` check, and a memory-only turn must not
        # manufacture an evidence warning.
        if observation.memory_auto_written:
            state.memory_auto_written = True
        # D24/D25/D31/D37: a degraded retrieval step is a fact about this turn, so
        # it rides into the answer envelope as a warning the UI can show.
        signals = getattr(observation, "signals", None) or {}
        if signals.get("low_confidence_applied") and "low_confidence_applied" not in state.warnings:
            state.warnings.append("low_confidence_applied")
        if (
            signals.get("rerank_enabled")
            and not signals.get("rerank_applied")
            and "rerank_fallback" not in state.warnings
        ):
            state.warnings.append("rerank_fallback")
        if signals.get("route_fallback") and "route_fallback" not in state.warnings:
            state.warnings.append("route_fallback")
        contributed: list[str] = []
        for hit in observation.hits:
            chunk_id = str(hit.get("chunk_id") or "")
            if chunk_id:
                state.observed_chunks[chunk_id] = hit
                contributed.append(chunk_id)
        # D57: record what this step actually made citable.  The eval could previously
        # only see *that* a page read happened, never whether it left the model holding
        # evidence it cannot cite -- which is the real hole (the attempt itself is fine:
        # A3 made a page read return its rows' chunk_ids).  ``state.observations[-1]``
        # is this step: ``_record`` appends before running this block.
        if state.observations:
            state.observations[-1]["observed"] = contributed

    def _gate(self, mode: str, scope_ids: list[str], user_text: str, state: RunState) -> str | None:
        if mode == MODE_DEEP:
            if state.has_evidence:
                return None
            # Q2 (2026-09-19): deep mode means "search when the question is about
            # the material", not "refuse everything you did not search for".  A
            # greeting or an off-topic aside was structurally refused (measured:
            # "你好" -> no_evidence_after_gate), which is both wrong and rude.  The
            # same rule chat mode already uses decides materiality; a material
            # question still cannot be answered without evidence.
            if not self._is_material(scope_ids, user_text):
                state.nonmaterial_pass = True
                state.warnings.append("deep_nonmaterial_pass")
                return None
            return GATE_BLOCK_DEEP
        if state.blocks >= 1:
            # Soft constraint: warn once, then let the turn finish (marked).
            return None
        filenames: list[str] = []
        if scope_ids:
            docs = (
                self.tools.collections.list_docs_multi(scope_ids)
                if len(scope_ids) > 1
                else self.tools.collections.list_docs(scope_ids[0])
            )
            filenames = [doc.filename for doc in docs]
        if is_related(user_text, has_collection=bool(scope_ids), filenames=filenames) and not state.searched:
            return GATE_BLOCK_CHAT
        return None

    def _is_material(self, scope_ids: list[str], user_text: str) -> bool:
        """Whether the question is about the collection at all (reuses ``is_related``)."""

        filenames: list[str] = []
        if scope_ids:
            try:
                docs = (
                    self.tools.collections.list_docs_multi(scope_ids)
                    if len(scope_ids) > 1
                    else self.tools.collections.list_docs(scope_ids[0])
                )
                filenames = [doc.filename for doc in docs]
            except Exception:  # noqa: BLE001 - a listing failure must not block the turn
                filenames = []
        return is_related(user_text, has_collection=bool(scope_ids), filenames=filenames)

    def _finish(self, session_id: str, mode: str, state: RunState, raw: str) -> AgentRunResult:
        answer = validate_citations(parse_agent_answer(raw), state.observed_chunks)
        if not answer.citations and answer.text and "no_citation" not in answer.warnings:
            answer.warnings.append("no_citation")
        if answer.refused and "model_refused" not in answer.warnings:
            # A12: the model declined on its own.  Recorded as a warning so the
            # trace shows *why* a turn without citations is not a failure.
            answer.warnings.append("model_refused")
        if mode == MODE_CHAT and not state.searched and state.warnings:
            answer.warnings.append("unverified_no_search")
        if state.scope_overridden and "scope_overridden" not in answer.warnings:
            # 第 8 条: this turn asked for material the session is not scoped to.  The
            # session's scope was used; say so in the answer's own record.
            answer.warnings.append("scope_overridden")
        if mode == MODE_DEEP and state.nonmaterial_pass and not state.has_evidence:
            # Same meaning as the chat-mode mark: answered without any evidence,
            # so the UI must not present it as sourced.
            answer.warnings.append("unverified_nonmaterial")
        if state.memory_auto_written and "memory_auto_written" not in answer.warnings:
            # M3=a: the user explicitly asked for this to be remembered, so it is
            # already in the long-term tables -- nothing is waiting for confirmation.
            answer.warnings.append("memory_auto_written")
        gate = {"blocks": state.blocks, "searched": state.searched, "read": state.read}
        self.db.append_message(
            session_id,
            "assistant",
            answer.text,
            payload_json=json.dumps(
                {
                    "citations": [c.__dict__ for c in answer.citations],
                    "warnings": answer.warnings,
                    "failure": None,
                    "refused": answer.refused,
                    "gate": gate,
                    "rounds": state.rounds,
                    "tool_trace_digest": _digest_tool_trace(state.observations),
                    "observed_chunk_ids": sorted(state.observed_chunks.keys()),
                },
                ensure_ascii=False,
            ),
        )
        return AgentRunResult(
            answer=answer.text,
            citations=[c.__dict__ for c in answer.citations],
            non_source=answer.non_source,
            warnings=answer.warnings,
            tool_trace=state.observations,
            mode=mode,
            gate=gate,
            refused=answer.refused,
            rounds=state.rounds,
            dropped_citations=list(answer.dropped_citations),
            observed_chunk_ids=sorted(state.observed_chunks.keys()),
        )

    def _failure(self, session_id: str, mode: str, state: RunState, reason: str) -> AgentRunResult:
        gate = {"blocks": state.blocks, "searched": state.searched, "read": state.read}
        warnings = [*state.warnings, reason]
        answer = _failure_message(state, reason)
        self.db.append_message(
            session_id,
            "assistant",
            answer,
            payload_json=json.dumps(
                {
                    "failure": reason,
                    "warnings": warnings,
                    "gate": gate,
                    "rounds": state.rounds,
                    "tool_trace_digest": _digest_tool_trace(state.observations),
                    "observed_chunk_ids": sorted(state.observed_chunks.keys()),
                },
                ensure_ascii=False,
            ),
        )
        return AgentRunResult(
            answer=answer,
            citations=[],
            non_source=[],
            warnings=warnings,
            tool_trace=state.observations,
            mode=mode,
            gate=gate,
            refused=True,
            failure=reason,
            rounds=state.rounds,
            observed_chunk_ids=sorted(state.observed_chunks.keys()),
        )


def _digest_tool_trace(
    observations: list[dict],
    *,
    limit: int = 20,
    arguments_max_chars: int = 500,
) -> list[dict]:
    """Keep the payload bounded: last ``limit`` calls, each ``arguments`` capped
    at ``arguments_max_chars`` of serialized JSON.  The live ``tool_trace`` in the
    HTTP response is untouched -- this is the audit copy that lands in the DB."""

    digested = []
    for entry in observations[-limit:]:
        item = dict(entry)
        serialized = json.dumps(item.get("arguments") or {}, ensure_ascii=False)
        if len(serialized) > arguments_max_chars:
            serialized = serialized[:arguments_max_chars]
        item["arguments"] = serialized
        digested.append(item)
    return digested


def _round_record(
    round_index: int,
    *,
    calls: list[str],
    started_offset_s: float,
    duration_s: float,
    final_answer: bool,
    gate_block: str | None = None,
) -> dict:
    """One round of the loop: what it called, how long it took, was it closable."""

    return {
        "round": round_index,
        "tool_calls": list(calls),
        "tool_call_count": len(calls),
        "started_offset_s": started_offset_s,
        "duration_s": duration_s,
        "final_answer": final_answer,
        "gate_block": gate_block,
    }


def _system_prompt(mode: str) -> str:
    common = (
        "You are ReadMate, a local reading assistant for the user's own PDF collection.\n"
        "Answer with evidence from tools whenever the question is about the material.\n"
        "Return your final answer as JSON only:\n"
        '{"answer": "...", "citations": [{"chunk_id": "...", "page": 1, "quote": "..."}], '
        '"non_source": ["..."], "refused": false}\n'
        "Every cited chunk_id must come from a search or read observation in this turn.\n"
        "Anything you add from general knowledge goes into non_source, never into citations.\n"
        "If the tools do not support an answer, say so plainly; never invent page numbers or quotes.\n"
        # A12: the refusal has to be machine-readable, not just polite prose.  A
        # Chinese "资料库中没有…" answer used to arrive indistinguishable from a
        # sourced one, so the evaluation could not tell a correct decline from a
        # hallucinated answer.
        'Set "refused": true when you decline to answer because the material does not '
        "contain what the question asks for (and say so in \"answer\"); keep it false "
        "when you do answer."
    )
    if mode == MODE_DEEP:
        return common + "\nDeep-read mode: you must call search (or read) and obtain evidence before answering."
    return common + "\nChat mode: search when the question touches the current collection; small talk may skip it."


def _retry_instruction(block: str, mode: str) -> str:
    if block == GATE_BLOCK_DEEP:
        return "Deep-read mode: you have no evidence yet. Call search or read first, then answer."
    return "This question looks related to the collection: call search first, then answer."


def _answer_now_instruction() -> str:
    """Injected on the last allowed round: no tools remain, so the only legal
    move is to answer with the evidence already on the table (A2)."""

    return (
        "Final step: no more tools are available. "
        "Answer now with the evidence you already gathered this turn "
        "(or admit plainly that the evidence is insufficient). "
        "Do not call search or read."
    )


def _failure_message(state: RunState, reason: str) -> str:
    """A1: stop printing "no evidence" when the loop actually gathered it.

    Three distinct branches the caller can only see from ``state``:

    * evidence was found but the loop still did not converge (usually a
      model that ignored the wrap-up instruction) -- do NOT say we found
      nothing;
    * no observations succeeded, but every tool round returned an error --
      say the tool path is broken, not that the collection lacks evidence;
    * truly no evidence (the deep-mode gate refused the answer).
    """

    if state.has_evidence:
        return _EVIDENCE_UNCONVERGED_MESSAGE
    observations = state.observations
    if observations and all(not entry.get("ok") for entry in observations):
        return _TOOL_ERROR_MESSAGE
    return NO_EVIDENCE_MESSAGE


def _assistant_tool_message(calls: list[ToolCall], content: str) -> dict:
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
            }
            for call in calls
        ],
    }