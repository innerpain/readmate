"""C9: the two weight-bearing retrieval objects are shared across requests.

Rebuilding them per request is what made the first ``search`` of every turn cost
58-80s (the loader ran again on every HTTP call).  These tests pin the sharing
contract, the escape hatch, and the rerank gate -- the three ways the cache can
regress.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval import model_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    model_cache.reset_model_cache()
    yield
    model_cache.reset_model_cache()


def _fake_embedder_cls(built: list):
    class _FakeGenerator:
        @classmethod
        def from_settings(cls, settings=None):
            built.append(object())
            return built[-1]

    return _FakeGenerator


def test_embedder_is_shared_until_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list = []
    # Patch where production *reads* it: ``model_cache`` imports the class at
    # module level (the in-function import shadowed the name and broke the
    # disabled-cache branch), so patching the defining module would be a
    # silent no-op.
    monkeypatch.setattr(model_cache, "EmbeddingGenerator", _fake_embedder_cls(built))
    monkeypatch.delenv("READMATE_MODEL_CACHE", raising=False)

    first = model_cache.get_embedder()
    assert model_cache.get_embedder() is first
    assert len(built) == 1, "second call must not reload the model"

    model_cache.reset_model_cache()
    assert model_cache.get_embedder() is not first
    assert len(built) == 2, "reset drops the singleton so tests start clean"


def test_cache_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """READMATE_MODEL_CACHE=0 keeps the old per-call construction for call sites
    that need an isolated object."""

    built: list = []
    # Patch where production *reads* it: ``model_cache`` imports the class at
    # module level (the in-function import shadowed the name and broke the
    # disabled-cache branch), so patching the defining module would be a
    # silent no-op.
    monkeypatch.setattr(model_cache, "EmbeddingGenerator", _fake_embedder_cls(built))
    monkeypatch.setenv("READMATE_MODEL_CACHE", "0")

    assert model_cache.get_embedder() is not model_cache.get_embedder()
    assert len(built) == 2


_AGENT = SimpleNamespace(
    tool_text_chars=1200,
    tool_table_chars=3000,
    tool_primary_min_ratio=0.6,
    tool_neighbor_min_chars=120,
)


def _stub_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_cache, "get_agent_settings", lambda: _AGENT)


def test_reranker_is_not_built_when_rerank_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kwargs helper is the single place the flag is honoured: with rerank
    disabled the retriever must receive no reranker at all (it never consults one),
    so the ~2.3GB model is never loaded."""

    monkeypatch.setattr(model_cache, "get_retrieval_settings", lambda: SimpleNamespace(rerank_enabled=False))
    monkeypatch.setattr(model_cache, "get_embedder", lambda: "embedder")
    _stub_agent(monkeypatch)
    called = {"n": 0}

    def _should_not_run():
        called["n"] += 1
        return "reranker"

    monkeypatch.setattr(model_cache, "get_reranker", _should_not_run)

    kwargs = model_cache.retrieval_model_kwargs()
    assert kwargs["embedder"] == "embedder"
    assert "reranker" not in kwargs
    assert called["n"] == 0


def test_reranker_is_passed_when_rerank_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_cache, "get_retrieval_settings", lambda: SimpleNamespace(rerank_enabled=True))
    monkeypatch.setattr(model_cache, "get_embedder", lambda: "embedder")
    monkeypatch.setattr(model_cache, "get_reranker", lambda: "reranker")
    _stub_agent(monkeypatch)

    kwargs = model_cache.retrieval_model_kwargs()
    assert kwargs["embedder"] == "embedder"
    assert kwargs["reranker"] == "reranker"


def test_the_agent_presentation_budget_is_handed_over_as_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """D26 (2026-09-20): the retriever used to call ``get_agent_settings()``
    mid-retrieval to size the presentation window, coupling the retrieval layer
    to the agent layer.  The composition root passes the agent's numbers in now."""

    monkeypatch.setattr(model_cache, "get_retrieval_settings", lambda: SimpleNamespace(rerank_enabled=False))
    monkeypatch.setattr(model_cache, "get_embedder", lambda: "embedder")
    _stub_agent(monkeypatch)

    budget = model_cache.retrieval_model_kwargs()["presentation"]
    assert (budget.budget, budget.table_budget) == (1200, 3000)
    assert (budget.primary_min_ratio, budget.neighbor_min_chars) == (0.6, 120)
