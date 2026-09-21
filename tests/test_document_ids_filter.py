"""The document_ids scope filter must never leak a document outside the set.

Same fake snapshot the multi-channel tests use: no Chroma, no real embedder.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.config.settings import ModelSettings, RetrievalSettings
from src.retrieval.embedding_generator import EmbeddingGenerator
from src.retrieval.persistent_retriever import PersistentRetriever

VOCAB = ["encoder", "attention", "layers", "masked", "bert"]


def _vector(text: str) -> np.ndarray:
    lowered = str(text).lower()
    values = np.array([lowered.count(term) for term in VOCAB], dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


class _FakeBackend:
    """Bag-of-words embedder: deterministic, comparable, dependency free."""

    def encode(self, texts, **kwargs):
        return np.vstack([_vector(text) for text in texts])


class _FakeStore:
    def __init__(self, chunks, version_id="v1"):
        self.version_id = version_id
        self._chunks = chunks

    def all_chunks(self):
        ids = list(self._chunks)
        return ids, [self._chunks[chunk_id]["text"] for chunk_id in ids]

    def query(self, embedding, k):
        query = np.asarray(embedding, dtype=np.float32)
        scored = []
        for chunk_id, payload in self._chunks.items():
            vector = _vector(payload["text"])
            similarity = float(np.dot(query, vector)) if vector.any() and query.any() else 0.0
            scored.append((chunk_id, similarity))
        scored.sort(key=lambda item: -item[1])
        top = scored[:k]
        return {
            "documents": [[self._chunks[chunk_id]["text"] for chunk_id, _ in top]],
            "metadatas": [[{**self._chunks[chunk_id]["metadata"], "chunk_id": chunk_id} for chunk_id, _ in top]],
            "distances": [[1.0 - score for _, score in top]],
        }

    def get_by_ids(self, chunk_ids):
        return {
            chunk_id: {
                "document": self._chunks[chunk_id]["text"],
                "metadata": {**self._chunks[chunk_id]["metadata"], "chunk_id": chunk_id},
            }
            for chunk_id in chunk_ids
            if chunk_id in self._chunks
        }


# d2's chunk is an exact match for the query terms; d1's is diluted with extra
# vocabulary, so an unfiltered search legitimately ranks the out-of-scope document
# first -- which is what makes the scope filter observable.
CHUNKS = {
    "c-masked": {
        "text": "attention encoder layers",
        "metadata": {"document_id": "d2", "filename": "bert.pdf", "page": 2, "section": "2", "content_kind": "text"},
    },
    "c-encoder": {
        "text": "encoder attention layers masked bert",
        "metadata": {"document_id": "d1", "filename": "attention.pdf", "page": 3, "section": "3.1", "content_kind": "text"},
    },
    "c-masked-far": {
        "text": "masked bert pretraining",
        "metadata": {"document_id": "d2", "filename": "bert.pdf", "page": 5, "section": "3", "content_kind": "text"},
    },
}


def _retriever(tmp_path=None, **overrides) -> PersistentRetriever:
    settings = RetrievalSettings(**overrides)
    index_dir = (Path(tmp_path) / "chroma") if tmp_path is not None else Path(tempfile.mkdtemp()) / "chroma"
    retriever = PersistentRetriever(
        embedder=EmbeddingGenerator(backend=_FakeBackend()),
        model_settings=ModelSettings(embedding_query_instruction=""),
        retrieval_settings=settings,
        index_dir=index_dir,
    )
    snapshot = _FakeStore(CHUNKS)
    retriever._load_snapshot = lambda: (  # type: ignore[assignment]
        snapshot,
        {
            "chunk_count": len(CHUNKS),
            "embedding_model": retriever.model_settings.embedding_model,
            "embedding_dimension": len(VOCAB),
            "normalize_embeddings": True,
        },
    )
    return retriever


def test_without_a_scope_the_other_document_wins(tmp_path):
    retriever = _retriever(tmp_path, candidate_k=3, final_k=3, rerank_enabled=False)
    hits = retriever.retrieve_multi(["attention encoder layers"], candidate_k=3, final_k=3)
    assert hits[0].document_id == "d2"
    assert retriever.last_diagnostics["document_ids_filter"] is None
    assert retriever.last_diagnostics["dropped_out_of_scope"] == 0


def test_scope_filter_keeps_only_documents_inside_the_collection(tmp_path):
    retriever = _retriever(tmp_path, candidate_k=3, final_k=3, rerank_enabled=False)
    hits = retriever.retrieve_multi(["attention encoder layers"], candidate_k=3, final_k=3, document_ids={"d1"})
    assert hits, "scoped retrieval returned nothing"
    assert {hit.document_id for hit in hits} == {"d1"}
    assert [hit.chunk_id for hit in hits] == ["c-encoder"]
    assert retriever.last_diagnostics["document_ids_filter"] == 1
    assert retriever.last_diagnostics["dropped_out_of_scope"] >= 1


def test_scope_filter_widens_the_raw_window(tmp_path):
    retriever = _retriever(tmp_path, candidate_k=2, final_k=2, rerank_enabled=False)
    retriever.retrieve_multi(["attention encoder layers"], candidate_k=2, final_k=2, document_ids={"d1"})
    # max(candidate_k, final_k) * FILTERED_WINDOW_MULTIPLIER, capped at MAX_FILTERED_WINDOW
    assert retriever.last_diagnostics["candidate_k"] == 8


def test_an_empty_scope_yields_nothing_instead_of_everything(tmp_path):
    retriever = _retriever(tmp_path, candidate_k=3, final_k=3, rerank_enabled=False)
    hits = retriever.retrieve_multi(["attention encoder layers"], candidate_k=3, final_k=3, document_ids=set())
    assert hits == []
    assert retriever.last_diagnostics["dropped_out_of_scope"] >= 1
