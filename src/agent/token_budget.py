"""Token budgeting for the conversation window (批 D / D32, 2026-09-20).

Why this module exists
----------------------
ReadMate used to hand the model ``history_turns * 2`` messages -- a fixed count
of 12 -- while the prompt it assembled measured **under 8% of a 32k window**
(fixed overhead 1,138 + history 1,484 + evidence up to 10,800 tokens).  The
window was not a budget at all: it cut the conversation by count, long before
the context was anywhere near full, and nothing compensated for the turns it
dropped.

The fix is a *token* budget: history is admitted newest-first until the budget
is spent, ``history_turns`` survives only as a floor, and the budget itself is a
fraction of the model's real window (``ModelProfile.context_length``).

The estimator is deliberately conservative and CJK-aware
--------------------------------------------------------
``ordered_chunker.estimate_tokens`` divides by 4, which is an English ratio: for
Chinese text it under-counts by roughly 1.5-4x (measured on a real prompt:
naive 2,127 vs CJK-aware 2,635).  A budget that under-counts silently overruns
the window, so CJK characters are counted as one token each here -- the safe
direction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# D32: the ratio and the clamps are settings-driven (AGENT_HISTORY_CONTEXT_RATIO),
# these are the outer bounds the ratio is clamped into.
HISTORY_BUDGET_FLOOR_TOKENS = 4_000
HISTORY_BUDGET_CAP_TOKENS = 24_000
# Above this share of the window the assembled prompt is logged as oversized.
PROMPT_PRESSURE_ALERT_RATIO = 0.85


def estimate_prompt_tokens(text: str) -> int:
    """Conservative token estimate: CJK 1 token/char, everything else 1/4."""

    if not text:
        return 0
    cjk = 0
    for char in text:
        code = ord(char)
        if (
            0x3400 <= code <= 0x4DBF        # CJK ext A
            or 0x4E00 <= code <= 0x9FFF     # CJK unified
            or 0xF900 <= code <= 0xFAFF     # compatibility ideographs
            or 0x3000 <= code <= 0x303F     # CJK punctuation
            or 0xFF00 <= code <= 0xFFEF     # full-width forms
        ):
            cjk += 1
    return cjk + max(0, len(text) - cjk) // 4


def history_budget_tokens(context_length: int, ratio: float) -> int:
    """``context_length * ratio``, clamped so a tiny or huge window stays sane."""

    try:
        context = int(context_length or 0)
    except (TypeError, ValueError):
        context = 0
    if context <= 0:
        context = 32_768
    budget = int(context * float(ratio))
    return max(HISTORY_BUDGET_FLOOR_TOKENS, min(HISTORY_BUDGET_CAP_TOKENS, budget))


def select_history(
    rows: Iterable[Mapping[str, object]],
    *,
    budget_tokens: int,
    floor_messages: int,
) -> list[Mapping[str, object]]:
    """Newest-first selection until the budget is spent; the floor always wins.

    ``rows`` arrive oldest-first (the shape ``AgentDB.recent_messages`` returns).
    A single oversized message is never split -- truncating a turn mid-sentence
    would corrupt the history -- so the newest message always gets in.
    """

    ordered = list(rows)
    if not ordered:
        return []
    floor = max(0, int(floor_messages))
    selected: list[Mapping[str, object]] = []
    spent = 0
    for row in reversed(ordered):
        if len(selected) >= floor:
            text = str(row.get("content") or "")
            cost = estimate_prompt_tokens(text)
            if selected and spent + cost > budget_tokens:
                break
            spent += cost
        selected.append(row)
    selected.reverse()
    return selected
