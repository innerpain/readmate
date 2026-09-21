"""Shared API and persistence schemas for the document knowledge base."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class IngestionStatus(str, Enum):
    """Business state of a document ingestion lifecycle."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionStage(str, Enum):
    """Fine-grained phase inside the document ingestion lifecycle."""

    QUEUED = "queued"
    PARSING = "parsing"
    PARSED = "parsed"
    OCR = "ocr"
    STRUCTURE = "structure"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    """Persistent metadata for one uploaded PDF."""

    document_id: str = Field(min_length=1)
    original_filename: str = Field(min_length=1)
    stored_filename: str = Field(min_length=1)
    revision: str = Field(default="legacy", min_length=1)
    file_size_bytes: int = Field(ge=0)
    status: IngestionStatus = IngestionStatus.QUEUED
    stage: IngestionStage = IngestionStage.QUEUED
    task_id: str | None = None
    page_count: int | None = Field(default=None, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    failure_code: str | None = None
    # FE-2: display name (empty = fall back to original_filename) and the retrieval
    # switch.  Both are product-level metadata: disabling never touches the index.
    alias: str = ""
    enabled: bool = True
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # D8/D11 (2026-09-20): the parse-quality summary.  It lives on the record --
    # and therefore on every route that returns one -- because ``response_model``
    # silently drops undeclared keys: the badge on the library cards renders from
    # ``GET /documents``, so declaring the fields only on the detail route left the
    # UI with nothing to show (found while verifying the parallel work).
    degraded: bool = False
    user_notice: str | None = None
    quality: dict | None = None


class DocumentUploadResponse(BaseModel):
    """Result returned after accepting a PDF upload."""

    document: DocumentRecord


class DocumentChangeResponse(BaseModel):
    """Result returned after replacing or otherwise changing a document."""

    document: DocumentRecord


class DocumentListResponse(BaseModel):
    """The local knowledge-base document registry."""

    documents: list[DocumentRecord]


class TaskStatusResponse(BaseModel):
    """Public Celery task status without exposing worker tracebacks."""

    task_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    result: dict[str, object] | None = None
    error_code: str | None = None
