"""Deterministic adapter stand-in: drives agent tests without an index."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src.adapter.contracts import (
    ADAPTER_API_VERSION,
    AdapterError,
    DocInfo,
    ReadResult,
    SearchHit,
)


class MockRagAdapter:
    """Scripted hits, documents and failures, with call counters."""

    api_version = ADAPTER_API_VERSION

    def __init__(
        self,
        *,
        hits: Sequence[SearchHit] | None = None,
        docs: Sequence[DocInfo] | None = None,
        reads: dict[str, ReadResult] | None = None,
        error: AdapterError | None = None,
        error_after: int | None = None,
        resolve_documents: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._hits = list(hits or [])
        self._docs = list(docs or [])
        self._reads = dict(reads or {})
        self._error = error
        self._error_after = error_after
        self._resolve = resolve_documents
        self.search_calls: list[dict[str, object]] = []
        self.read_calls: list[dict[str, object]] = []
        self.list_calls: list[str] = []

    def _maybe_fail(self) -> None:
        if self._error is None:
            return
        if self._error_after is None or len(self.search_calls) > self._error_after:
            raise self._error

    def search(
        self,
        collection_id: str,
        query: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
        gate_text: str | None = None,
        mode: str | None = None,
        previous_empty_searches: int = 0,
    ) -> list[SearchHit]:
        self.search_calls.append(
            {
                "collection_id": collection_id,
                "query": query,
                "top_k": top_k,
                "document_ids": list(document_ids or []),
                "gate_text": gate_text,
                "mode": mode,
                "previous_empty_searches": previous_empty_searches,
            }
        )
        self._maybe_fail()
        allowed = set(document_ids) if document_ids else None
        hits = [hit for hit in self._hits if allowed is None or hit.document_id in allowed]
        return hits[: int(top_k)] if top_k else hits

    def read(
        self,
        *,
        chunk_id: str | None = None,
        document_id: str | None = None,
        page: int | None = None,
        element_id: str | None = None,
    ) -> ReadResult:
        self.read_calls.append(
            {"chunk_id": chunk_id, "document_id": document_id, "page": page, "element_id": element_id}
        )
        if chunk_id and chunk_id in self._reads:
            return _normalize_read(self._reads[chunk_id])
        if self._reads:
            return _normalize_read(next(iter(self._reads.values())))
        raise AdapterError("chunk_not_found", "mock has no reads configured")

    def list_docs(self, collection_id: str) -> list[DocInfo]:
        self.list_calls.append(collection_id)
        return list(self._docs)


def _normalize_read(result: ReadResult) -> ReadResult:
    """Mirror ``RagAdapterImpl._read_chunk`` so a scripted chunk read is citable
    through the same ``chunk_ids`` whitelist the real adapter exposes."""

    if result.chunk_id and not result.chunk_ids:
        return result.model_copy(update={"chunk_ids": [result.chunk_id]})
    return result
