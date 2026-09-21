"""The RAG Adapter surface the Agent is allowed to call."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.adapter.contracts import DocInfo, ReadResult, SearchHit


class RagAdapter(Protocol):
    """search / read / list_docs -- and nothing else (no upload, no reindex)."""

    api_version: str

    def search(
        self,
        collection_id: str,
        query: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]: ...

    def read(
        self,
        *,
        chunk_id: str | None = None,
        document_id: str | None = None,
        page: int | None = None,
        element_id: str | None = None,
    ) -> ReadResult: ...

    def list_docs(self, collection_id: str) -> list[DocInfo]: ...
