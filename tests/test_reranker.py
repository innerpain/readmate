"""Unit tests for the local cross-encoder reranker (fake backend, no model download)."""

from __future__ import annotations

from src.retrieval.reranker import LocalCrossEncoderReranker


class _FakeBackend:
    def predict(self, pairs, **kwargs):
        # Prefer passages that contain the query token "alpha".
        scores = []
        for query, passage in pairs:
            scores.append(10.0 if "alpha" in passage.lower() else 0.1)
        return scores


def test_rerank_orders_by_backend_score_and_respects_top_k():
    reranker = LocalCrossEncoderReranker(backend=_FakeBackend())
    ranked = reranker.rerank(
        "find alpha evidence",
        [
            ("c1", "beta beta unrelated"),
            ("c2", "contains ALPHA signal"),
            ("c3", "gamma only"),
        ],
        top_k=2,
    )
    assert [item.chunk_id for item in ranked] == ["c2", "c1"]
    assert ranked[0].rerank_score > ranked[1].rerank_score


def test_rerank_skips_empty_candidates():
    reranker = LocalCrossEncoderReranker(backend=_FakeBackend())
    ranked = reranker.rerank(
        "alpha",
        [
            ("", "alpha"),
            ("c1", "   "),
            ("c2", "alpha hit"),
        ],
    )
    assert [item.chunk_id for item in ranked] == ["c2"]
