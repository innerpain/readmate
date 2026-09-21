"""Frozen RAG Adapter contract -- ``readmate-rag-adapter-3``.

The Agent layer speaks only this vocabulary.  Changing a field name or type
means publishing a new ``ADAPTER_API_VERSION`` and a matching runtime change,
because ReadMate validates citations against exactly these fields.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

ADAPTER_API_VERSION = "readmate-rag-adapter-3"

# Stable error codes (readmate/RAG接入Agent方案.md 5.6).
COLLECTION_EMPTY = "collection_empty"
KNOWLEDGE_BASE_EMPTY = "knowledge_base_empty"
DOCUMENT_NOT_IN_COLLECTION = "document_not_in_collection"
CHUNK_NOT_FOUND = "chunk_not_found"
DOCUMENT_NOT_FOUND = "document_not_found"
EMBEDDING_INCOMPATIBLE = "embedding_incompatible"
BUILD_LOCKED = "build_locked"
INVALID_ARGUMENTS = "invalid_arguments"

ERROR_CODES = frozenset(
    {
        COLLECTION_EMPTY,
        KNOWLEDGE_BASE_EMPTY,
        DOCUMENT_NOT_IN_COLLECTION,
        CHUNK_NOT_FOUND,
        DOCUMENT_NOT_FOUND,
        EMBEDDING_INCOMPATIBLE,
        BUILD_LOCKED,
        INVALID_ARGUMENTS,
    }
)


class AdapterError(Exception):
    """Every adapter failure is a stable code plus a human message."""

    def __init__(self, code: str, message: str = "", *, retryable: bool = False) -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.retryable = retryable

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code}: {self.message}"


class SearchHit(BaseModel):
    """One retrieval hit.  ``score`` is always the dense cosine."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page: int = Field(ge=1)
    # A3: last page this hit covers; None when the passage does not straddle a
    # page break (or when an older snapshot has no span recorded).
    page_end: int | None = Field(default=None, ge=1)
    section: str = ""
    heading_path: list[str] = Field(default_factory=list)
    score: float
    chunk_type: str = ""
    content_kind: str = "text"
    table_id: str | None = None
    figure_id: str | None = None
    atomic: bool = False
    degraded_structure: bool = False
    presentation_content: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    expanded_neighbors: dict[str, list[dict[str, str]]] | None = None


# Asserted by tests so the contract cannot drift silently.
REQUIRED_SEARCH_FIELDS = (
    "chunk_id",
    "document_id",
    "filename",
    "page",
    # A3 (2026-09-19): the passage's last page.  Added to the guard on
    # 2026-09-20 -- it was in the model but not in this list, so the drift
    # assertion silently did not cover the newest field.
    "page_end",
    "score",
    "chunk_type",
    "content_kind",
    "atomic",
    "degraded_structure",
    "presentation_content",
    "excerpt",
)


class ReadResult(BaseModel):
    document_id: str = Field(min_length=1)
    chunk_id: str | None = None
    page: int | None = Field(default=None, ge=1)
    # A3: last page the read covers (a passage, or a page-read's rows, can span
    # more than one page); None when unknown or identical to ``page``.
    page_end: int | None = Field(default=None, ge=1)
    section: str = ""
    text: str = ""
    chunk_type: str | None = None
    truncated: bool = False
    # D34 (2026-09-20): the page branch cuts the text at the page ceiling, and
    # the tool layer used to compute "remaining" from the already-cut body, so
    # it always reported 0.  The adapter states the truth instead: how much of
    # the page is left, and where to resume.
    remaining_chars: int | None = Field(default=None, ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    structure_summary: dict[str, object] | None = None
    filename: str = ""
    # A page-level read spans several chunks; the model must be able to cite
    # any of them, so the adapter returns every chunk_id the page contains.
    chunk_ids: list[str] = Field(default_factory=list)


# Asserted by tests so the read surface cannot drift silently.  ``chunk_id`` is
# only present for the chunk branch; the page branch populates ``chunk_ids``.
REQUIRED_READ_FIELDS = (
    "document_id",
    "page",
    # A3 (2026-09-19); guard completed 2026-09-20 (see D36).
    "page_end",
    "text",
    "chunk_type",
    "truncated",
    "filename",
    "chunk_ids",
)


class DocInfo(BaseModel):
    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    status: str = ""
    page_count: int | None = Field(default=None, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    failure_code: str | None = None
