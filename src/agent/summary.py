"""Session summaries: pointer-style, incremental, failure-tolerant (批 D / D32).

Why pointer-style
-----------------
ReadMate answers questions *about documents*, and every document is still on disk
and reachable through ``search`` / ``read``.  A coding agent has to summarise the
code it read because the file may change; ReadMate only needs to remember **what
was discussed and where the answer lives**.  So the summary carries pointers --
document names, page numbers, section numbers, table labels, user constraints --
and never paraphrases document text.  That keeps it short (the budget is ~600
characters), cheap, and hard to get wrong.

Why incremental
----------------
Each run feeds the previous summary plus only the turns that have newly fallen
out of the window, so a long session costs one small call per compression instead
of re-reading the whole transcript.

Why the failure paths matter
----------------------------
A summary that cannot be generated must leave the session *exactly as it was* --
no half-written text, no deleted rows.  The caller counts failures and stops
after ``MAX_SUMMARY_FAILURES`` so a broken endpoint cannot burn a call per turn.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from src.agent.token_budget import estimate_prompt_tokens

logger = logging.getLogger(__name__)

# D32 阶段 3: stop trying after this many consecutive failures.
MAX_SUMMARY_FAILURES = 3
# The design budget: ~600 characters.  Roughly 900 tokens, far less than the
# history it replaces.
SUMMARY_MAX_CHARS = 600
# How many of the newest messages must stay verbatim in the window; the summary
# covers everything older than these.
SUMMARY_KEEP_RECENT_MESSAGES = 12

POINTER_SUMMARY_PROMPT = """You compress the earlier part of a document-reading session.

Rules:
- Keep ONLY pointers and decisions: which document, which page / section / table,
  what the user asked for, what was concluded, what is still open.
- NEVER restate document text.  A later turn can search or read the document again.
- Keep the user's standing constraints verbatim (language, output shape, scope).
- Write in the same language the user writes in.  At most {max_chars} characters.
- Output the summary only: no preamble, no headings, no bullet markers beyond "- ".

Reply with exactly these five lines, omitting a line when it has no content:
目标：<what the user is trying to get out of this session>
已确认：<conclusions reached, each with document name + page/section/table pointer>
约束：<standing user constraints>
未决：<open questions or promised follow-ups>

{previous_block}Earlier turns to fold in (oldest first):
{transcript}
"""


def _previous_block(previous: str) -> str:
    text = str(previous or "").strip()
    if not text:
        return ""
    return f"Existing summary to extend (do not repeat it verbatim):\n{text}\n\n"


def _format_transcript(rows: Iterable[Mapping[str, object]], *, max_chars: int = 6000) -> str:
    """Oldest-first ``role: content`` lines, capped so one huge turn cannot dominate."""

    lines: list[str] = []
    used = 0
    for row in rows:
        role = str(row.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        content = " ".join(str(row.get("content") or "").split())
        if not content:
            continue
        line = f"{role}: {content}"
        if used + len(line) > max_chars:
            remaining = max_chars - used
            if remaining < 80:
                break
            line = line[:remaining] + "…"
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def _clean_summary(raw: str, *, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    # Strip the fenced-block habit some models have.
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


class SessionSummaryService:
    """Builds and stores one session's pointer summary.  Storage-agnostic on input."""

    def __init__(self, *, db, llm, keep_recent: int = SUMMARY_KEEP_RECENT_MESSAGES) -> None:
        self.db = db
        self.llm = llm
        self.keep_recent = int(keep_recent)

    # ------------------------------------------------------------------ planning
    def summary_boundary(self, session_id: str) -> int | None:
        """The newest message id that should be folded into the summary.

        Everything older than the newest ``keep_recent`` messages qualifies; the
        return value is the id of the last such message, or ``None`` when the
        session is still short enough to fit verbatim.
        """

        rows = self.db.list_messages(session_id)
        if len(rows) <= self.keep_recent:
            return None
        boundary = int(rows[-self.keep_recent - 1]["id"])
        session = self._session(session_id)
        covered = int((session or {}).get("summary_upto_message_id") or 0)
        return boundary if boundary > covered else None

    def is_due(self, session_id: str) -> bool:
        """Whether a summary run would fold in anything new."""

        try:
            return self.summary_boundary(session_id) is not None
        except Exception as error:  # noqa: BLE001 - never let planning break a turn
            logger.warning("summary planning failed for %s: %s", session_id, error)
            return False

    # ------------------------------------------------------------------ building
    def build(self, session_id: str, *, upto_message_id: int) -> str | None:
        """Generate the new summary text, or ``None`` when the model produced nothing."""

        session = self._session(session_id) or {}
        previous = str(session.get("summary") or "")
        covered = int(session.get("summary_upto_message_id") or 0)
        rows = [
            row
            for row in self.db.list_messages(session_id)
            if covered < int(row.get("id") or 0) <= int(upto_message_id)
        ]
        transcript = _format_transcript(rows)
        if not transcript.strip():
            return None
        prompt = POINTER_SUMMARY_PROMPT.format(
            max_chars=SUMMARY_MAX_CHARS,
            previous_block=_previous_block(previous),
            transcript=transcript,
        )
        raw = self.llm.generate(prompt)
        return _clean_summary(raw) or None

    def summarize(self, session_id: str, *, upto_message_id: int | None = None) -> dict:
        """Plan, generate, store.  Never raises; reports what happened instead.

        ``stored: False`` with a ``reason`` is the honest answer for every failure
        path -- the session is left untouched, and the caller decides whether to
        count it against the failure budget.
        """

        boundary = int(upto_message_id or 0) or self.summary_boundary(session_id)
        if not boundary:
            return {"stored": False, "reason": "nothing_to_summarize", "upto_message_id": None}
        try:
            summary = self.build(session_id, upto_message_id=boundary)
        except Exception as error:  # noqa: BLE001 - the caller counts failures
            logger.warning("summary generation failed for %s: %s", session_id, error)
            return {"stored": False, "reason": "generation_failed", "upto_message_id": boundary}
        if not summary:
            return {"stored": False, "reason": "empty_summary", "upto_message_id": boundary}
        self.db.set_session_summary(session_id, summary, boundary)
        logger.info(
            "session summary stored: %s up to message %d (%d chars, ~%d tokens)",
            session_id,
            boundary,
            len(summary),
            estimate_prompt_tokens(summary),
        )
        return {
            "stored": True,
            "reason": "ok",
            "upto_message_id": boundary,
            "chars": len(summary),
            "summary": summary,
        }

    # ------------------------------------------------------------------- helpers
    def _session(self, session_id: str) -> dict | None:
        try:
            return self.db.get_session(session_id)
        except Exception:  # noqa: BLE001 - a missing session is a normal race
            return None
