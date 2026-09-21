"""Process-wide caches for the two *weight-bearing* retrieval objects.

Why this module exists
----------------------
Every ``/agent/*`` request used to construct a fresh ``PersistentRetriever``
through ``agent_routes.build_runtime()``, which reloaded the embedding model and
the cross-encoder reranker.  Measured on 2026-09-19: the first ``search`` of
*every* turn cost 58-80s while later searches in the same turn cost 2-9s, and a
single test session produced 48 "Loading weights" passes in the api container.

Why only these two objects
--------------------------
The agent objects are cheap and stateful -- ``AgentDB`` opens a short-lived
sqlite connection per call, ``PersistentRetriever`` keeps per-call
``last_diagnostics``.  Sharing them across requests would buy milliseconds and
risk cross-request state bugs.  The models are the expensive, stateless part, so
they are shared and everything else stays per-request exactly as before.

Thread safety
-------------
FastAPI runs these sync routes in a worker thread pool, so the singletons are
double-checked under a lock.  ``READMATE_MODEL_CACHE=0`` turns sharing off (for a
call site that wants a fresh object), and ``reset_model_cache()`` drops them for
test isolation.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from src.config.settings import get_agent_settings, get_model_settings, get_retrieval_settings
from src.retrieval.embedding_generator import EmbeddingGenerator
from src.retrieval.persistent_retriever import PresentationBudget

_EMBEDDER: Any | None = None
_RERANKER: Any | None = None
_LOCK = threading.Lock()

_FALSY = {"0", "false", "no", "off"}


def _cache_enabled() -> bool:
    """Sharing is on unless explicitly disabled (tests, one-off scripts)."""

    return os.getenv("READMATE_MODEL_CACHE", "1").strip().lower() not in _FALSY


def get_embedder() -> Any:
    """The shared ``EmbeddingGenerator`` (query-side encoding for retrieval)."""

    global _EMBEDDER
    if not _cache_enabled():
        return EmbeddingGenerator.from_settings(get_model_settings())
    if _EMBEDDER is None:
        with _LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = EmbeddingGenerator.from_settings(get_model_settings())
    return _EMBEDDER


def get_reranker() -> Any:
    """The shared cross-encoder reranker, built from current retrieval settings.

    Callers must gate on ``get_retrieval_settings().rerank_enabled`` -- building
    this loads a ~2.3GB model, and the retriever never consults it when rerank is
    off, so the disabled configuration must not pay for it.
    """

    global _RERANKER
    settings = get_retrieval_settings()
    if not _cache_enabled():
        from src.retrieval.reranker import LocalCrossEncoderReranker

        return LocalCrossEncoderReranker(settings.rerank_model, max_length=settings.rerank_max_length)
    if _RERANKER is None:
        with _LOCK:
            if _RERANKER is None:
                from src.retrieval.reranker import LocalCrossEncoderReranker

                _RERANKER = LocalCrossEncoderReranker(settings.rerank_model, max_length=settings.rerank_max_length)
    return _RERANKER


def retrieval_model_kwargs() -> dict[str, Any]:
    """``PersistentRetriever`` keyword arguments backed by the shared models.

    Single call site so ``rerank_enabled`` is honoured in exactly one place.
    """

    settings = get_retrieval_settings()
    kwargs: dict[str, Any] = {"embedder": get_embedder()}
    if settings.rerank_enabled:
        kwargs["reranker"] = get_reranker()
    # D26 (2026-09-20): the presentation window belongs to the agent layer, so the
    # composition root hands it over as data -- the retriever no longer imports
    # agent settings mid-retrieval.
    agent = get_agent_settings()
    kwargs["presentation"] = PresentationBudget(
        budget=agent.tool_text_chars,
        table_budget=agent.tool_table_chars,
        primary_min_ratio=agent.tool_primary_min_ratio,
        neighbor_min_chars=agent.tool_neighbor_min_chars,
    )
    return kwargs


def reset_model_cache() -> None:
    """Drop the singletons (test isolation; the next call reloads them)."""

    global _EMBEDDER, _RERANKER
    with _LOCK:
        _EMBEDDER = None
        _RERANKER = None