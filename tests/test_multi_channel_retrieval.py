"""Multi-channel retrieval end to end, with a fake snapshot and a fake embedder.

This exercises the parts that are easy to get wrong without an index: which
chunks are eligible for a slot, whether the fused order survives, whether a
sub-query's exclusive hit keeps a place, and whether single-query retrieval
still behaves exactly as it did before channels existed.
"""

import tempfile
from pathlib import Path

import numpy as np

from src.config.settings import ModelSettings, RetrievalSettings
from src.retrieval.embedding_generator import EmbeddingGenerator
from src.retrieval.persistent_retriever import PersistentRetriever

VOCAB = ["encoder", "attention", "rag", "curatedtrec", "p100", "training", "steps", "bleu", "table", "score"]


def _vector(text: str) -> np.ndarray:
    lowered = str(text).lower()
    values = np.array([lowered.count(term) for term in VOCAB], dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


class _FakeBackend:
    """Bag-of-words embedder: deterministic, comparable, and dependency free."""

    def encode(self, texts, **kwargs):
        return np.vstack([_vector(text) for text in texts])


class _FakeStore:
    """Snapshot stand-in exposing the same surface the retriever uses."""

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
            chunk_id: {"document": self._chunks[chunk_id]["text"],
                       "metadata": {**self._chunks[chunk_id]["metadata"], "chunk_id": chunk_id}}
            for chunk_id in chunk_ids
            if chunk_id in self._chunks
        }


CHUNKS = {
    "c-encoder": {"text": "encoder attention layer", "metadata": {"document_id": "d1", "filename": "a.pdf", "page": 3, "section": "3.1", "content_kind": "text"}},
    "c-rag": {"text": "rag generation curatedtrec", "metadata": {"document_id": "d2", "filename": "rag.pdf", "page": 6, "section": "4", "content_kind": "text"}},
    "c-p100": {"text": "p100 training steps hours", "metadata": {"document_id": "d1", "filename": "a.pdf", "page": 7, "section": "5.1", "content_kind": "text"}},
    "c-bleu": {"text": "bleu score translation", "metadata": {"document_id": "d1", "filename": "a.pdf", "page": 8, "section": "6", "content_kind": "text"}},
}


def _retriever(tmp_path=None, store=None, **settings_overrides):
    settings = RetrievalSettings(**settings_overrides)
    index_dir = (Path(tmp_path) / "chroma") if tmp_path is not None else Path(tempfile.mkdtemp()) / "chroma"
    retriever = PersistentRetriever(
        embedder=EmbeddingGenerator(backend=_FakeBackend()),
        model_settings=ModelSettings(embedding_query_instruction=""),
        retrieval_settings=settings,
        index_dir=index_dir,
    )
    snapshot = store or _FakeStore(CHUNKS)
    retriever._load_snapshot = lambda: (snapshot, {  # type: ignore[assignment]
        "chunk_count": len(CHUNKS),
        "embedding_model": retriever.model_settings.embedding_model,
        "embedding_dimension": len(VOCAB),
        "normalize_embeddings": True,
    })
    return retriever


def test_single_query_retrieval_keeps_the_dense_order(tmp_path):
    retriever = _retriever(tmp_path)
    results = retriever.retrieve("encoder attention")
    assert [result.chunk_id for result in results][0] == "c-encoder"
    assert results[0].score == max(result.score for result in results)
    assert results[0].content == CHUNKS["c-encoder"]["text"]


def test_fusion_orders_by_agreement_across_queries(tmp_path):
    retriever = _retriever(tmp_path)
    results = retriever.retrieve_multi(["encoder attention", "rag curatedtrec"], candidate_k=4, final_k=4)
    ids = [result.chunk_id for result in results]
    # Each query's best chunk makes the list; both are ranked above the rest.
    assert ids[0] in {"c-encoder", "c-rag"}
    assert {"c-encoder", "c-rag"}.issubset(set(ids))
    assert retriever.last_diagnostics["channels"] == ["dense:0", "dense:1"]


def test_quota_keeps_a_slot_for_a_sub_query_that_only_it_retrieved(tmp_path):
    # The question channel ranks the encoder chunk first; the sub-query is the
    # only channel that knows about the BLEU chunk.
    retriever = _retriever(tmp_path, candidate_k=2, final_k=2, quota_per_query=1)
    results = retriever.retrieve_multi(["encoder attention", "bleu score"], candidate_k=2, final_k=2)
    ids = [result.chunk_id for result in results]
    assert "c-encoder" in ids and "c-bleu" in ids


def test_a_chunk_no_channel_retrieved_never_gets_a_slot(tmp_path):
    """D19: the lexical channel is gone, so the only way into the window is a
    dense channel retrieving the chunk.  A candidate the embedder never returned
    must not appear -- and no keyword-only counter exists to report it any more.
    """

    store = _FakeStore({**CHUNKS, "c-only": {"text": "zebra", "metadata": {"document_id": "d3", "filename": "z.pdf", "page": 1, "section": "1"}}})
    # candidate_k=2 keeps 'c-only' out of every dense channel.
    retriever = _retriever(tmp_path, store=store, candidate_k=2, final_k=4)
    results = retriever.retrieve_multi(["encoder attention"], candidate_k=2, final_k=4)
    assert "c-only" not in [result.chunk_id for result in results]
    assert retriever.last_diagnostics["channels"] == ["dense:0"]
    assert "lexical_only_count" not in retriever.last_diagnostics
    assert "lexical_version" not in retriever.last_diagnostics


def test_prose_slots_are_per_chunk_not_per_page(tmp_path):
    """D22: prose used to collapse to one slot per (document, page), so the
    second passage of a page could never be retrieved even when it held the
    answer.  Two prose chunks on the same page must both keep a slot."""

    same_page = {
        "c-page-a": {"text": "encoder attention layers", "metadata": {"document_id": "d1", "filename": "a.pdf", "page": 3, "section": "3.1", "content_kind": "text"}},
        "c-page-b": {"text": "encoder attention mask", "metadata": {"document_id": "d1", "filename": "a.pdf", "page": 3, "section": "3.2", "content_kind": "text"}},
    }
    retriever = _retriever(tmp_path, store=_FakeStore(same_page), candidate_k=4, final_k=4)
    ids = [result.chunk_id for result in retriever.retrieve_multi(["encoder attention"], candidate_k=4, final_k=4)]
    assert {"c-page-a", "c-page-b"} <= set(ids)


def test_structured_chunk_metadata_survives_retrieval(tmp_path):
    store = _FakeStore({"c-table": {"text": "[table] Table 3: scores", "metadata": {
        "document_id": "d1", "filename": "rag.pdf", "page": 19, "section": "6.2",
        "content_kind": "table", "table_id": "t-9", "atomic": True, "degraded_structure": True,
        "structure_pages": [19, 20],
    }}})
    retriever = _retriever(tmp_path, store=store)
    result = retriever.retrieve("table scores")[0]
    assert result.content_kind == "table"
    assert result.table_id == "t-9"
    assert result.atomic is True
    assert result.degraded_structure is True
    assert result.structure_pages == [19, 20]
