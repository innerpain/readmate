"""Query planner: term table, clause split, LLM merge.

D19/D38 (2026-09-20): the lexical channel and the keyword chain were deleted, so
``QueryPlan`` now carries only the original question, its sub-queries and one
English rendering.  The term table survives because the planner prompt still
shows the model the domain vocabulary it is expected to translate.
"""

import json

from src.config.settings import QueryPlanningSettings
from src.rag.query_planner import QueryPlanner
from src.rag.terms import matched_terms, term_pairs_for


class _StubLLM:
    """Text-completion stub: returns whatever the test wants the model to say."""

    def __init__(self, payload):
        self.payload = payload
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload, ensure_ascii=False)


def test_term_table_maps_the_domain_vocabulary_to_english():
    terms = matched_terms("多头注意力和位置编码的关系是什么？")
    assert "Multi-Head Attention" in terms
    assert "Positional Encoding" in terms
    pairs = dict(term_pairs_for("多头注意力"))
    assert pairs["多头注意力"] == "Multi-Head Attention"


def test_planner_without_an_llm_splits_clauses_and_keeps_terms():
    planner = QueryPlanner(settings=QueryPlanningSettings(llm_enabled=False))
    plan = planner.plan("Transformer 的训练数据是什么？硬件用的是什么？训练时间是多少？")
    assert plan.provider == "terms"
    assert len(plan.subqueries) >= 2
    # D38: there is no keyword channel to feed any more, so the plan exposes the
    # sub-queries (which go to dense) and nothing else.
    assert not hasattr(plan, "keywords")
    # The original question always leads the dense channels.
    assert plan.dense_texts()[0].startswith("Transformer 的训练数据")


def test_planner_uses_llm_subqueries_and_merges_them():
    llm = _StubLLM({
        "subqueries": ["What training data was used?", "How long was the base model trained?"],
        "english_query": "Transformer training data hardware and training time for base and big models",
    })
    planner = QueryPlanner(llm=llm, settings=QueryPlanningSettings())
    plan = planner.plan("论文中 Transformer 的训练数据、硬件和训练时间分别是什么？区分 Base 和 Big 模型")
    assert plan.provider == "llm+terms"
    assert plan.subqueries[0] == "What training data was used?"
    assert plan.english_query.startswith("Transformer training data")
    # Everything the plan exposes is a dense text, and the original leads.
    assert plan.dense_texts()[:2] == [plan.original, "What training data was used?"]
    assert llm.prompts and "never answer" in llm.prompts[0].lower()


def test_planner_falls_back_when_the_model_returns_junk():
    planner = QueryPlanner(llm=_StubLLM("I cannot help with that."), settings=QueryPlanningSettings())
    plan = planner.plan("多头注意力是什么？位置编码呢？")
    assert plan.provider == "terms"
    assert len(plan.subqueries) >= 2


def test_planner_is_inert_when_disabled():
    planner = QueryPlanner(settings=QueryPlanningSettings(enabled=False))
    plan = planner.plan("多头注意力是什么？")
    assert plan.provider == "identity"
    assert plan.dense_texts() == ["多头注意力是什么？"]
