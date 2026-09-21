"""Retriever-exit tests for A4 (global min_score) and D5 (gate_query).

The fake store and bag-of-words embedder mirror ``test_multi_channel_retrieval``
so the behaviour is measured end-to-end without an index or an embedding model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.config.settings import ModelSettings, RetrievalSettings
from src.retrieval.embedding_generator import EmbeddingGenerator
from src.retrieval.persistent_retriever import PersistentRetriever

VOCAB = ["encoder", "attention", "table", "score", "bleu", "unrelated"]


def _vector(text: str) -> np.ndarray:
    lowered = str(text).lower()
    values = np.array([lowered.count(term) for term in VOCAB], dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


class _FakeBackend:
    def encode(self, texts, **kwargs):
        return np.vstack([_vector(text) for text in texts])


class _FakeStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def all_chunks(self):
        ids = list(self._chunks)
        return ids, [self._chunks[cid]["text"] for cid in ids]

    def query(self, embedding, k):
        query = np.asarray(embedding, dtype=np.float32)
        scored = []
        for chunk_id, payload in self._chunks.items():
            vec = _vector(payload["text"])
            sim = float(np.dot(query, vec)) if vec.any() and query.any() else 0.0
            scored.append((chunk_id, sim))
        scored.sort(key=lambda item: -item[1])
        top = scored[:k]
        return {
            "documents": [[self._chunks[cid]["text"] for cid, _ in top]],
            "metadatas": [[{**self._chunks[cid]["metadata"], "chunk_id": cid} for cid, _ in top]],
            "distances": [[1.0 - sim for _, sim in top]],
        }

    def get_by_ids(self, chunk_ids):
        return {
            cid: {"document": self._chunks[cid]["text"],
                  "metadata": {**self._chunks[cid]["metadata"], "chunk_id": cid}}
            for cid in chunk_ids if cid in self._chunks
        }


CHUNKS = {
    "strong": {
        "text": "encoder attention encoder attention",
        "metadata": {
            "document_id": "d1", "filename": "a.pdf", "page": 1, "section": "3",
            "chunk_type": "prose", "content_kind": "text", "heading_path": ["3 Model"],
        },
    },
    "weak": {
        "text": "bleu bleu",
        "metadata": {
            "document_id": "d1", "filename": "a.pdf", "page": 2, "section": "6",
            "chunk_type": "prose", "content_kind": "text", "heading_path": ["6 Results"],
        },
    },
    "table": {
        "text": "attention table score",
        "metadata": {
            "document_id": "d1", "filename": "a.pdf", "page": 3, "section": "6.2",
            "chunk_type": "table_pack", "content_kind": "table", "table_id": "t6",
            "heading_path": ["6 Results", "6.2 Model Variations"],
            "neighbor_prev_chunk_ids": [], "neighbor_next_chunk_ids": [],
        },
    },
}


def _retriever(**overrides):
    index_dir = Path(tempfile.mkdtemp()) / "chroma"
    settings = RetrievalSettings(**overrides)
    retriever = PersistentRetriever(
        embedder=EmbeddingGenerator(backend=_FakeBackend()),
        model_settings=ModelSettings(embedding_query_instruction=""),
        retrieval_settings=settings,
        index_dir=index_dir,
    )
    retriever._load_snapshot = lambda: (  # type: ignore[assignment]
        _FakeStore(CHUNKS),
        {
            "chunk_count": len(CHUNKS),
            "embedding_model": retriever.model_settings.embedding_model,
            "embedding_dimension": len(VOCAB),
            "normalize_embeddings": True,
        },
    )
    return retriever


def test_min_score_zero_keeps_everything():
    retriever = _retriever(min_score=0.0, score_floor_ratio=0.0, rerank_enabled=False)
    results = retriever.retrieve_multi(["encoder attention"], final_k=5, gate_query="encoder attention Table")
    ids = {r.chunk_id for r in results}
    assert {"strong", "weak", "table"} <= ids
    assert retriever.last_diagnostics["dropped_by_min_score"] == 0


def test_min_score_filter_drops_weak_and_reports_count():
    # Force the "weak" chunk to fall below the threshold while the "table"
    # chunk (which shares "attention") stays above it.
    retriever = _retriever(min_score=0.5, score_floor_ratio=0.0, rerank_enabled=False)
    results = retriever.retrieve_multi(
        ["encoder attention"], final_k=5, gate_query="encoder attention Table",
    )
    ids = {r.chunk_id for r in results}
    assert "weak" not in ids
    assert retriever.last_diagnostics["dropped_by_min_score"] >= 1


def test_gate_query_opens_table_gate_independently_of_dense_texts():
    """D5: an English model-rewritten query has no table cue, but the user's
    original question does.  The gate must read the user wording."""

    retriever_off = _retriever(min_score=0.0, score_floor_ratio=0.0, rerank_enabled=False)
    results_off = retriever_off.retrieve_multi(["attention encoder"], final_k=5, gate_query="attention encoder")
    assert "table" not in {r.chunk_id for r in results_off}

    retriever_on = _retriever(min_score=0.0, score_floor_ratio=0.0, rerank_enabled=False)
    results_on = retriever_on.retrieve_multi(
        ["attention encoder"], final_k=5, gate_query="讲讲 Table 6 的分数",
    )
    assert "table" in {r.chunk_id for r in results_on}


def test_gate_query_defaults_to_first_dense_text():
    """When no gate_query is supplied, behaviour is unchanged (gate reads the
    first retrieval text)."""

    retriever = _retriever(min_score=0.0, score_floor_ratio=0.0, rerank_enabled=False)
    results = retriever.retrieve_multi(["attention encoder table"], final_k=5)
    assert "table" in {r.chunk_id for r in results}


def test_diagnostics_always_expose_min_score_and_drop_count():
    retriever = _retriever(min_score=0.3, score_floor_ratio=0.0, rerank_enabled=False)
    retriever.retrieve_multi(["encoder attention"], final_k=5, gate_query="encoder attention")
    diags = retriever.last_diagnostics
    assert "dropped_by_min_score" in diags
    assert "min_score" in diags
    assert diags["min_score"] == 0.3


def test_candidate_k_defaults_to_the_shrunken_pool():
    """D30 (2026-09-20): the pool default went 40 -> 15.  Measured on the probe:
    recall@5 1.000 / MRR 0.8451 at 0.79s per question, versus pool 40's
    0.9412 / 0.8333 at 5.10s -- metrics up, retrieval time -84%."""

    from src.config.settings import RetrievalSettings as _RetrievalSettings

    assert _RetrievalSettings().candidate_k == 15
    assert _RetrievalSettings.from_env().candidate_k == 15
    # An omitted candidate_k must take the new default, not a stale literal.
    retriever = _retriever(min_score=0.0, score_floor_ratio=0.0, rerank_enabled=False)
    retriever.retrieve_multi(["encoder attention"], final_k=5, gate_query="encoder attention")
    assert retriever.last_diagnostics["candidate_k"] == 15
