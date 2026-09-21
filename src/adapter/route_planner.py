"""Dual-track retrieval routing: the model decides, rules keep a safety net.

Track direct   one dense query -- the behaviour that existed before planning,
               cheapest for a single-fact question.
Track planned  sub-queries plus an English rendering -- the configuration the
               82% source-hit figure was measured in.

The model makes the choice, and the same request also produces the plan when it
chooses the planned track, so routing costs no extra call over always-planning.
If that request fails or returns something unparsable, deterministic signals
decide: routing must not become a new failure mode.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace

from src.config.settings import RouteSettings, get_route_settings
from src.rag.query_planner import QueryPlanner

logger = logging.getLogger(__name__)

TRACK_DIRECT = "direct"
TRACK_PLANNED = "planned"

ROUTE_AUTO = "auto"
ROUTE_MODES = frozenset({ROUTE_AUTO, TRACK_DIRECT, TRACK_PLANNED})

# Same clause boundaries the planner uses; kept here so the router can run its
# rule fallback without depending on planner internals.
_CLAUSE_SPLIT_RE = re.compile(r"[\uFF1F?\uFF1B;\u3002!\uFF01]|\uFF0C|\u3001|\u5E76\u4E14|\u4EE5\u53CA|\u8FD8\u6709|\u540C\u65F6|\u5206\u522B")
_NUMBER_RE = re.compile(r"\d")

# A8 (2026-09-19): a CJK question over the English corpus must not take the
# dense-only track.  Punctuation ranges are included on purpose -- a section
# number sits inside a Chinese sentence, and the question's script is what
# decides here.
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")
# A verbatim anchor (a section number, a model name, a metric) is what the
# dense-only track cannot carry across languages; pure-Chinese wording does
# not need the extra planning call.
_LITERAL_RE = re.compile(r"[0-9A-Za-z]")


def _cjk_ratio(text: str) -> float:
    """Share of non-space characters that are CJK."""

    chars = [ch for ch in str(text) if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if _CJK_RE.match(ch)) / len(chars)

ROUTE_PROMPT = """You route one question about the user's own documents to a retrieval track. You never answer.

Question:
{question}

Cheap signals already computed for you (use them, do not recompute):
- parallel clauses detected: {clauses}
- contains numbers or metric values: {has_numbers}
- characters: {question_chars}
- mode: {mode}
- documents in scope: {collection_size}

tracks:
- "direct": one similarity search on the question as written. Use it for a single
  short, unambiguous request, usually one fact, name or definition.
- "planned": split the question into sub-questions, plus one English search
  sentence. Use it when the question
  asks several things at once, when the question language differs from the
  documents, or when exact numbers matter.

Reply with ONLY this JSON:
{{"track": "direct" | "planned", "reason": "at most 12 words", "subqueries": ["..."], "english_query": "..."}}
When the track is "direct", leave the other two fields empty.

Question again:
{question}
"""


@dataclass(frozen=True)
class RouteSignals:
    """Cheap, deterministic facts the model gets to see before deciding."""

    clauses: int = 1
    has_numbers: bool = False
    question_chars: int = 0
    mode: str = "deep"
    collection_size: int = 0
    previous_empty_searches: int = 0
    # D27: the threshold is data, not a module constant, so it stays tunable.
    long_question_chars: int = 40

    @property
    def wants_plan(self) -> bool:
        return (
            self.clauses >= 2
            or self.has_numbers
            or self.question_chars >= self.long_question_chars
            or self.previous_empty_searches > 0
        )


@dataclass(frozen=True)
class RouteDecision:
    track: str = TRACK_DIRECT
    reason: str = ""
    provider: str = "rules"        # llm | rules | forced
    subqueries: tuple[str, ...] = ()
    english_query: str = ""
    # D37 (2026-09-20): True when the model path failed and the rules decided.
    # Surfaced through diagnostics so a silent routing downgrade is visible.
    fallback: bool = False

    def dense_texts(self, original: str) -> list[str]:
        if self.track != TRACK_PLANNED:
            return [original]
        texts: list[str] = []
        for text in (original, *self.subqueries, self.english_query):
            value = str(text).strip()
            if value and value not in texts:
                texts.append(value)
        return texts

    def as_dict(self) -> dict[str, object]:
        return {
            "track": self.track,
            "provider": self.provider,
            "reason": self.reason,
            "fallback": self.fallback,
        }


def split_clauses(query: str, *, limit: int | None = None, min_chars: int | None = None) -> list[str]:
    """Split on the same boundaries the planner prompt is told to use.

    D27: the limits default to ``RouteSettings`` so tuning them needs no code
    change; both stay overridable for callers that measure.
    """

    route = get_route_settings()
    limit = route.max_subqueries if limit is None else int(limit)
    min_chars = route.min_clause_chars if min_chars is None else int(min_chars)

    clauses = [clause.strip(" \u3000\uFF0C\u3002\uFF1F?") for clause in _CLAUSE_SPLIT_RE.split(query)]
    selected = [c for c in clauses if len(c) >= min_chars and c != query]
    return selected[:limit] if len(selected) >= 2 else []


def compute_signals(
    query: str,
    *,
    mode: str = "deep",
    collection_size: int = 0,
    previous_empty_searches: int = 0,
    route_settings: RouteSettings | None = None,
) -> RouteSignals:
    route = route_settings or get_route_settings()
    return RouteSignals(
        clauses=len(split_clauses(query, limit=route.max_subqueries, min_chars=route.min_clause_chars)) or 1,
        has_numbers=bool(_NUMBER_RE.search(query)),
        question_chars=len(str(query).strip()),
        mode=mode,
        collection_size=collection_size,
        previous_empty_searches=previous_empty_searches,
        long_question_chars=route.long_question_chars,
    )


class RetrievalRouter:
    """Decide which track a query takes; ``llm=None`` means rules only."""

    def __init__(
        self,
        *,
        llm=None,
        planner: QueryPlanner | None = None,
        mode: str = ROUTE_AUTO,
        settings: RouteSettings | None = None,
    ) -> None:
        if mode not in ROUTE_MODES:
            raise ValueError(f"unknown route mode: {mode}")
        self.llm = llm
        self.planner = planner
        self.mode = mode
        self.route_settings = settings or get_route_settings()

    def decide(self, query: str, signals: RouteSignals | None = None) -> RouteDecision:
        signals = signals or compute_signals(query, route_settings=self.route_settings)
        if self.mode == TRACK_DIRECT:
            return RouteDecision(track=TRACK_DIRECT, provider="forced", reason="forced direct")
        if self.mode == TRACK_PLANNED:
            return self._planned_with_planner(query)
        if self.llm is None:
            decision = self._rule_decision(query, signals, reason="no router model")
        else:
            try:
                decision = self._llm_decision(query, signals)
            except Exception as error:  # pragma: no cover - network dependent
                logger.warning("route model failed; falling back to rules: %s", error)
                decision = replace(
                    self._rule_decision(query, signals, reason="router model failed"), fallback=True
                )
            if decision is None:
                decision = replace(
                    self._rule_decision(query, signals, reason="router output unusable"), fallback=True
                )
        # Only ``auto`` mode is guarded: the forced modes above are an explicit
        # operator choice and must keep behaving exactly as pinned.
        return self._cross_language_guard(query, decision)

    def _cross_language_guard(self, query: str, decision: RouteDecision) -> RouteDecision:
        """Route a CJK question to the planned track (A8).

        ``direct`` carries no keyword channel, so a question whose only concrete
        anchor is a literal -- a section number inside a Chinese sentence -- had
        nothing to match on: measured dense cosine 0.35-0.41 for the four chunks
        that *do* contain it, under the 0.45 gate, and reported to the user as
        "not in the corpus".  The planned track feeds the literal to the keyword
        channel as well.
        """

        if decision.track != TRACK_DIRECT or _cjk_ratio(query) < self.route_settings.cross_language_ratio:
            # English (or a pinned track) keeps the original contract: the dense
            # ranking alone already serves it, so nothing here may change.
            return decision
        if not _LITERAL_RE.search(query):
            # D38 (2026-09-20): the keyword channel is gone, so the branch that
            # used to hand this question's literal anchors to it is a no-op and
            # was deleted.  A pure-Chinese question with no literal stays on the
            # dense track -- the configuration the source-hit baseline was
            # measured in.
            return decision
        return self._planned_with_planner(query, reason="forced planned: cjk question with literal")

    # ------------------------------------------------------------------- paths
    def _llm_decision(self, query: str, signals: RouteSignals) -> RouteDecision | None:
        prompt = ROUTE_PROMPT.format(
            question=query,
            clauses=signals.clauses,
            has_numbers=signals.has_numbers,
            question_chars=signals.question_chars,
            mode=signals.mode,
            collection_size=signals.collection_size,
        )
        step = self.llm.complete([{"role": "user", "content": prompt}])
        payload = _json_object(getattr(step, "content", ""))
        if payload is None:
            return None
        track = str(payload.get("track") or "").strip().lower()
        if track not in {TRACK_DIRECT, TRACK_PLANNED}:
            return None
        subqueries = tuple(
            str(item).strip() for item in (payload.get("subqueries") or []) if str(item).strip()
        )
        english = str(payload.get("english_query") or "").strip()
        if track == TRACK_PLANNED and not (subqueries or english):
            return None
        return RouteDecision(
            track=track,
            provider="llm",
            reason=str(payload.get("reason") or "")[:120],
            subqueries=subqueries[: self.route_settings.max_subqueries],
            english_query=english,
        )

    def _planned_with_planner(self, query: str, *, reason: str = "forced planned") -> RouteDecision:
        if self.planner is None:
            return self._rule_decision(query, compute_signals(query, route_settings=self.route_settings), reason=reason)
        plan = self.planner.plan(query)
        return RouteDecision(
            track=TRACK_PLANNED,
            provider="forced",
            reason=reason,
            subqueries=tuple(plan.subqueries),
            english_query=plan.english_query,
        )

    def _rule_decision(self, query: str, signals: RouteSignals, *, reason: str) -> RouteDecision:
        if not signals.wants_plan:
            return RouteDecision(track=TRACK_DIRECT, provider="rules", reason=reason)
        return RouteDecision(
            track=TRACK_PLANNED,
            provider="rules",
            reason=reason,
            subqueries=tuple(split_clauses(query, limit=self.route_settings.max_subqueries, min_chars=self.route_settings.min_clause_chars)),
            english_query=query if str(query).isascii() else "",
        )


def _json_object(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*", "", str(text or "").strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
