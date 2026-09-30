"""Local registry for uploaded documents (D61: PDF + common office/text formats)."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator
from uuid import uuid4

from src.models.schemas import DocumentRecord, IngestionStage, IngestionStatus


MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
LOCK_STALE_SECONDS = 30

# D61: the accepted upload formats (docs/Docling多格式支持方案.md §8).  Kept as a
# frozen mapping of extension -> accepted MIME types; ``application/octet-stream``
# is accepted for every format because browsers and curl routinely send it, and
# the real format gate is the extension plus the magic-byte check below.
ACCEPTED_EXTENSIONS: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf"}),
    ".docx": frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    ),
    ".pptx": frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    ),
    ".xlsx": frozenset(
        {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    ),
    ".md": frozenset({"text/markdown", "text/plain"}),
    ".html": frozenset({"text/html", "text/plain"}),
    ".htm": frozenset({"text/html", "text/plain"}),
    # Windows Excel exports arrive as application/vnd.ms-excel far too often.
    ".csv": frozenset(
        {"text/csv", "text/plain", "application/vnd.ms-excel", "application/csv"}
    ),
}
_UNIVERSAL_MIME = {"application/octet-stream", ""}
# OOXML container formats share the ZIP local-file header; PDFs have %PDF-.
_ZIP_MAGIC = b"PK\x03\x04"
_MAGIC_BY_EXTENSION: dict[str, bytes] = {
    ".pdf": b"%PDF-",
    ".docx": _ZIP_MAGIC,
    ".pptx": _ZIP_MAGIC,
    ".xlsx": _ZIP_MAGIC,
}


class DocumentRegistryError(Exception):
    """Base error for document-registry operations."""


class InvalidUploadError(DocumentRegistryError):
    """Raised when an uploaded file does not meet PDF requirements."""


class DocumentNotFoundError(DocumentRegistryError):
    """Raised when a requested document ID is absent from the registry."""


class RegistryLockedError(DocumentRegistryError):
    """Raised when another process holds the short registry write lock."""


class RegistryDataError(DocumentRegistryError):
    """Raised when the persisted registry cannot be read safely."""


class DocumentBusyError(DocumentRegistryError):
    """Raised when a document already has an active ingestion task."""


class DocumentRegistry:
    """Stores PDFs by UUID and keeps only mutable document metadata."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    ):
        project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir is not None else project_root / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.registry_path = self.data_dir / "registry.json"
        self.lock_path = self.data_dir / "registry.lock"
        self.max_file_size_bytes = max_file_size_bytes
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> list[DocumentRecord]:
        """Return all registry records without exposing mutable internals."""

        return self._load_documents()

    def get_document(self, document_id: str) -> DocumentRecord:
        """Return a single document record by its stable UUID."""

        documents = self._load_documents()
        document = next(
            (item for item in documents if item.document_id == document_id),
            None,
        )

        if document is None:
            raise DocumentNotFoundError(document_id)

        return document

    def delete_document(self, document_id: str) -> DocumentRecord:
        """Remove one registry record and its UUID-named PDF atomically."""
        with self._registry_lock():
            documents = self._load_documents()
            target = next((item for item in documents if item.document_id == document_id), None)
            if target is None:
                raise DocumentNotFoundError(document_id)
            remaining = [item for item in documents if item.document_id != document_id]
            self._write_documents(remaining)
            (self.uploads_dir / target.stored_filename).unlink(missing_ok=True)
            return target

    def register_upload(
        self,
        filename: str | None,
        content_type: str | None,
        stream: BinaryIO,
    ) -> DocumentRecord:
        """Save one document; every upload creates a new document identity."""

        safe_filename, file_type = self._validate_upload_metadata(filename, content_type)
        temp_path, file_size = self._write_temp_file(stream, file_type)

        try:
            with self._registry_lock():
                documents = self._load_documents()
                document_id = str(uuid4())
                # The stored name keeps the *original* extension so the parse
                # stage (and the /file route) can tell the format from disk
                # alone; ``file_type`` on the record is the authoritative copy.
                stored_filename = f"{document_id}{file_type}"
                destination = self.uploads_dir / stored_filename
                document = DocumentRecord(
                    document_id=document_id,
                    original_filename=safe_filename,
                    stored_filename=stored_filename,
                    file_type=file_type,
                    revision=str(uuid4()),
                    file_size_bytes=file_size,
                    status=IngestionStatus.QUEUED,
                )

                os.replace(temp_path, destination)
                temp_path = None

                try:
                    self._write_documents([*documents, document])
                except Exception:
                    destination.unlink(missing_ok=True)
                    raise

                return document
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def replace_upload(
        self,
        document_id: str,
        filename: str | None,
        content_type: str | None,
        stream: BinaryIO,
    ) -> DocumentRecord:
        """Replace a document while preserving its document identity.

        D61: a replacement may change the format (pdf -> docx).  The stored file
        is therefore re-named to the new extension, and the *old* stored file is
        only deleted once the registry write has succeeded -- a crash between the
        two leaves an orphan file, never a document whose bytes are gone.
        """
        safe_filename, file_type = self._validate_upload_metadata(filename, content_type)
        temp_path, file_size = self._write_temp_file(stream, file_type)
        try:
            with self._registry_lock():
                documents = self._load_documents()
                index = next((i for i, item in enumerate(documents) if item.document_id == document_id), None)
                if index is None:
                    raise DocumentNotFoundError(document_id)
                current = documents[index]
                if current.status in {IngestionStatus.QUEUED, IngestionStatus.PROCESSING} and current.task_id:
                    raise DocumentBusyError(document_id)
                new_stored_filename = f"{document_id}{file_type}"
                destination = self.uploads_dir / new_stored_filename
                os.replace(temp_path, destination)
                temp_path = None
                updates = current.model_dump(mode="json")
                updates.update({
                    "original_filename": safe_filename,
                    "stored_filename": new_stored_filename,
                    "file_type": file_type,
                    "file_size_bytes": file_size,
                    "revision": str(uuid4()),
                    "status": IngestionStatus.QUEUED,
                    "stage": IngestionStage.QUEUED,
                    "task_id": None,
                    "page_count": None,
                    "chunk_count": 0,
                    "failure_code": None,
                    "updated_at": datetime.now(timezone.utc),
                })
                updated = DocumentRecord.model_validate(updates)
                documents[index] = updated
                self._write_documents(documents)
                if current.stored_filename != new_stored_filename:
                    (self.uploads_dir / current.stored_filename).unlink(missing_ok=True)
                return updated
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def update_document(
        self,
        document_id: str,
        *,
        status: IngestionStatus | None = None,
        stage: IngestionStage | None = None,
        task_id: str | None = None,
        page_count: int | None = None,
        page_kind: str | None = None,
        chunk_count: int | None = None,
        failure_code: str | None = None,
        clear_failure_code: bool = False,
    ) -> DocumentRecord:
        """Persist a state transition made by a future ingestion worker."""

        with self._registry_lock():
            documents = self._load_documents()
            for index, document in enumerate(documents):
                if document.document_id != document_id:
                    continue

                updates = document.model_dump(mode="json")
                updates["updated_at"] = datetime.now(timezone.utc)

                for field, value in {
                    "status": status,
                    "stage": stage,
                    "task_id": task_id,
                    "page_count": page_count,
                    "page_kind": page_kind,
                    "chunk_count": chunk_count,
                    "failure_code": failure_code,
                }.items():
                    if value is not None:
                        updates[field] = value

                if clear_failure_code:
                    updates["failure_code"] = None

                updated_document = DocumentRecord.model_validate(updates)
                documents[index] = updated_document
                self._write_documents(documents)
                return updated_document

        raise DocumentNotFoundError(document_id)

    def update_document_meta(
        self,
        document_id: str,
        *,
        alias: str | None = None,
        enabled: bool | None = None,
    ) -> DocumentRecord:
        """Rename (alias) and/or switch one document off (FE-2).

        A disabled document stays in the library and in its collections -- it is only
        excluded from retrieval (see ``CollectionService.document_ids``).  That is what
        "停用" has to mean for a RAG tool: the file is still yours, it just stops
        feeding answers, and no index rebuild is needed either way.
        """

        with self._registry_lock():
            documents = self._load_documents()
            for index, document in enumerate(documents):
                if document.document_id != document_id:
                    continue

                updates = document.model_dump(mode="json")
                if alias is not None:
                    updates["alias"] = alias.strip()[:120]
                if enabled is not None:
                    updates["enabled"] = bool(enabled)
                updates["updated_at"] = datetime.now(timezone.utc)

                updated_document = DocumentRecord.model_validate(updates)
                documents[index] = updated_document
                self._write_documents(documents)
                return updated_document

        raise DocumentNotFoundError(document_id)

    def _validate_upload_metadata(
        self,
        filename: str | None,
        content_type: str | None,
    ) -> tuple[str, str]:
        """Return ``(safe_filename, file_type)``, raising ``InvalidUploadError``.

        The extension is the format gate (D61); the MIME only has to be one of
        the plausible types for *that* extension -- or the universal
        ``application/octet-stream`` that browsers and curl send on a bad day.
        """

        if not filename:
            raise InvalidUploadError("missing_filename")

        safe_filename = Path(filename.replace("\\", "/")).name
        file_type = Path(safe_filename).suffix.lower()
        if file_type not in ACCEPTED_EXTENSIONS:
            raise InvalidUploadError("unsupported_file_type")

        normalized_content_type = (content_type or "").lower()
        if normalized_content_type not in _UNIVERSAL_MIME:
            if normalized_content_type not in ACCEPTED_EXTENSIONS[file_type]:
                raise InvalidUploadError("unsupported_content_type")

        return safe_filename, file_type

    def _write_temp_file(
        self, stream: BinaryIO, file_type: str
    ) -> tuple[Path, int]:
        """Spool one upload to a temp file with size/emptiness/magic checks."""

        temp_path = self.uploads_dir / f".{uuid4().hex}.uploading"
        first_bytes = b""
        file_size = 0

        try:
            with temp_path.open("xb") as destination:
                while chunk := stream.read(1024 * 1024):
                    if not first_bytes:
                        # 5 covers %PDF-, 4 covers the ZIP local header; keep a
                        # small margin so neither magic can straddle the cut.
                        first_bytes = chunk[:8]

                    file_size += len(chunk)
                    if file_size > self.max_file_size_bytes:
                        raise InvalidUploadError("file_too_large")

                    destination.write(chunk)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        if file_size == 0:
            temp_path.unlink(missing_ok=True)
            raise InvalidUploadError("empty_file")

        # Text formats have no magic number -- an unreadable one fails honestly
        # at parse time (D61: no silent guessing, no silent downgrade).
        magic = _MAGIC_BY_EXTENSION.get(file_type)
        if magic is not None and not first_bytes.startswith(magic):
            temp_path.unlink(missing_ok=True)
            raise InvalidUploadError("invalid_file_header")

        return temp_path, file_size

    def _load_documents(self) -> list[DocumentRecord]:
        if not self.registry_path.exists():
            return []

        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
            records = payload.get("documents", [])
            if not isinstance(records, list):
                raise TypeError("documents must be a list")

            return [DocumentRecord.model_validate(item) for item in records]
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            raise RegistryDataError("registry_unreadable") from error

    def _write_documents(self, documents: list[DocumentRecord]) -> None:
        payload = {
            "version": 1,
            "documents": [item.model_dump(mode="json") for item in documents],
        }
        temp_path = self.data_dir / f".registry-{uuid4().hex}.tmp"

        try:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_path, self.registry_path)
        finally:
            temp_path.unlink(missing_ok=True)

    @contextmanager
    def _registry_lock(self) -> Iterator[None]:
        """Use an atomic lock file for the short registry write critical section."""

        deadline = time.monotonic() + 5
        lock_fd: int | None = None

        while lock_fd is None:
            try:
                lock_fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                if self._lock_is_stale():
                    self.lock_path.unlink(missing_ok=True)
                    continue
                if time.monotonic() >= deadline:
                    raise RegistryLockedError("registry_busy")
                time.sleep(0.05)

        try:
            os.write(lock_fd, str(os.getpid()).encode("ascii"))
            yield
        finally:
            os.close(lock_fd)
            self.lock_path.unlink(missing_ok=True)

    def _lock_is_stale(self) -> bool:
        """Recover from a lock file left behind by a terminated process."""

        try:
            lock_age = time.time() - self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return False

        return lock_age > LOCK_STALE_SECONDS
