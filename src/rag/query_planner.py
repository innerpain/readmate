"""Query planning: one Chinese question in, several retrieval queries out.

Two failure modes motivated this module, both measured on the fixed question
set: 15 of 17 answerable questions ask for several things at once (so a single
query vector averages them away), and the documents are English while the
questions are Chinese (so the question has to be rendered in English too).

The plan therefore carries two things:

    subqueries      one per stated requirement, so each can be retrieved for
    english_query   a retrieval-side English rendering of the question

Nothing here is allowed to answer the question, and a failure at any point
degrades to the deterministic term-table plan instead of raising: retrieval
quality must not depend on an LLM being reachable.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from src.config.settings import QueryPlanningSettings, get_query_planning_settings
from src.rag.terms import matched_terms, term_pairs_for

logger = logging.getLogger(__name__)

QUERY_PLANNER_VERSION = "query-planner-1"
# Clause boundaries a multi-requirement Chinese question uses.
_CLAUSE_SPLIT_RE = re.compile(r"[？?；;。!！]|，|、|并且|以及|还有|同时|分别")
_MIN_CLAUSE_CHARS = 4

PLANNER_PROMPT = """You prepare retrieval queries for a search engine. You never answer.

Question (may be Chinese, documents are English):
{question}

Known term mappings you may reuse (Chinese -> English):
{terms}

Rules:
1. Split the question into at most {max_subqueries} sub-questions, one per thing
   being asked. Keep each sub-question short and self-contained.
2. Write "english_query": one English search sentence that captures the whole
   question in the wording an English paper would use.
3. "keywords": names, metrics, dataset names and numbers that must be matched
   exactly. Copy numbers and proper nouns character for character.
4. Never answer the question, never add facts that are not in it.
5. Reply with ONLY this JSON:
{{"subqueries": ["..."], "english_query": "...", "keywords": ["..."]}}

Question again:
{question}
"""


class PromptLLM(Protocol):
    """Minimal text-completion boundary (matches the project LLM adapters)."""

    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class QueryPlan:
    """Retrieval-side view of one user question."""

    original: str
    subqueries: tuple[str, ...] = ()
    english_query: str = ""
    provider: str = "identity"

    def dense_texts(self) -> list[str]:
        """Texts to embed and search with (deduplicated, original kept first)."""

        texts: list[str] = []
        for text in (self.original, *self.subqueries, self.english_query):
            value = str(text).strip()
            if value and value not in texts:
                texts.append(value)
        return texts

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "subqueries": list(self.subqueries),
            "english_query": self.english_query,
            "keywords": list(self.keywords),
        }


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text)))


class QueryPlanner:
    """Build a retrieval plan, falling back to deterministic keywords."""

    def __init__(
        self,
        *,
        llm: PromptLLM | None = None,
        settings: QueryPlanningSettings | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_query_planning_settings()

    @property
    def version(self) -> str:
        mode = "llm" if self.llm is not None and self.settings.llm_enabled else "terms"
        return f"{QUERY_PLANNER_VERSION}:{mode}"

    def plan(self, question: str) -> QueryPlan:
        original = str(question).strip()
        if not original or not self.settings.enabled:
            return QueryPlan(original=original, provider="identity")

        deterministic = self._deterministic_plan(original)

        if not self.settings.llm_enabled or self.llm is None:
            return deterministic
        llm_plan = self._llm_plan(original)
        if llm_plan is None:
            return deterministic
        return self._merge(llm_plan, deterministic)

    # -------------------------------------------------------------- fallbacks
    def _deterministic_plan(self, question: str) -> QueryPlan:
        """No-LLM plan: clause-level sub-queries.

        The English query is left empty on purpose -- a machine-built English
        sentence would be worse than the original for a multilingual embedder.
        """

        subqueries = self._clauses(question)
        return QueryPlan(
            original=question,
            subqueries=tuple(subqueries),
            english_query="" if _has_cjk(question) else question,
            provider="terms",
        )

    def _clauses(self, question: str) -> list[str]:
        if self.settings.max_subqueries <= 1:
            return []
        raw_clauses = _CLAUSE_SPLIT_RE.split(question)
        clauses = [clause.strip(" 　，。？?") for clause in raw_clauses]
        selected = [
            clause for clause in clauses
            if len(clause) >= _MIN_CLAUSE_CHARS and (clause != question)
        ]
        # A single surviving clause is the question itself; no split happened.
        if len(selected) < 2:
            return []
        return selected[: self.settings.max_subqueries]

    # -------------------------------------------------------------------- llm
    def _llm_plan(self, question: str) -> QueryPlan | None:
        terms = matched_terms(question)
        pairs = [pair for pair in term_pairs_for(question) if pair[1] in terms]
        term_hint = "\n".join(f"- {zh} -> {en}" for zh, en in pairs) or "(none)"
        prompt = PLANNER_PROMPT.format(
            question=question,
            terms=term_hint,
            max_subqueries=self.settings.max_subqueries,
        )
        try:
            raw = self.llm.generate(prompt) if self.llm is not None else ""
        except Exception as error:  # pragma: no cover - network/model dependent
            logger.warning("Query planning fell back to terms: %s", error)
            return None
        payload = _parse_json(raw)
        if payload is None:
            return None
        subqueries = [
            str(item).strip() for item in (payload.get("subqueries") or [])
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]
        english_query = str(payload.get("english_query") or "").strip()
        if not english_query and not subqueries:
            return None
        return QueryPlan(
            original=question,
            subqueries=tuple(subqueries[: self.settings.max_subqueries]),
            english_query=english_query,
            provider="llm+terms",
        )

    @staticmethod
    def _merge(llm_plan: QueryPlan, deterministic: QueryPlan) -> QueryPlan:
        """Keep the LLM's sub-queries, fall back to the deterministic split."""

        subqueries = list(llm_plan.subqueries) or list(deterministic.subqueries)
        return QueryPlan(
            original=llm_plan.original,
            subqueries=tuple(subqueries),
            english_query=llm_plan.english_query or deterministic.english_query,
            provider=llm_plan.provider,
        )


def _term_pairs(question: str, english_terms: list[str]) -> list[tuple[str, str]]:
    """Chinese/English pairs actually relevant to this question, for the prompt."""

    pairs: list[tuple[str, str]] = []
    for chinese, english in _TERM_LOOKUP(question):
        if english in english_terms:
            pairs.append((chinese, english))
    return pairs


def _TERM_LOOKUP(question: str) -> list[tuple[str, str]]:
    from src.rag.terms import _ORDERED_TERMS  # local import keeps the table private

    return [(zh, en) for zh, en in _ORDERED_TERMS if zh in question]


def _parse_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
