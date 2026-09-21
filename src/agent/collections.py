"""Collections are retrieval views over the single Chroma snapshot."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from src.adapter.contracts import (
    COLLECTION_EMPTY,
    AdapterError,
    DocInfo,
)


class CollectionService:
    """Maps a collection_id to document_ids and to registry status."""

    def __init__(self, *, db, registry) -> None:
        self.db = db
        self.registry = registry

    def disabled_document_ids(self) -> set[str]:
        """Documents switched off in the library (FE-2).

        Read from the registry, not from the index: disabling is a product decision
        and must not require a rebuild.  This is the single choke point that keeps a
        disabled document out of every retrieval path -- the adapter's
        ``resolve_documents=collections.document_ids`` and the multi-collection
        ``document_ids_multi`` both funnel through here.
        """

        try:
            records = self.registry.list_documents()
        except Exception:  # noqa: BLE001 - a registry hiccup must not block retrieval
            return set()
        return {str(record.document_id) for record in records if getattr(record, "enabled", True) is False}

    def document_ids(self, collection_id: str) -> list[str]:
        raw = self.db.collection_document_ids(collection_id)
        if not raw:
            raise AdapterError(COLLECTION_EMPTY, f"collection {collection_id} has no documents")
        disabled = self.disabled_document_ids()
        ids = [document_id for document_id in raw if document_id not in disabled]
        if not ids:
            raise AdapterError(COLLECTION_EMPTY, f"every document in collection {collection_id} is disabled")
        return ids

    def document_ids_multi(self, collection_ids: Sequence[str]) -> set[str]:
        """Union of document ids across several collections (R7 / D-5b).

        A missing/empty collection simply contributes nothing; only an entirely
        empty union is treated as ``collection_empty`` so the caller still gets a
        stable error when nothing is in scope.  Order is irrelevant (a set).
        """

        union: set[str] = set()
        for collection_id in collection_ids:
            if not collection_id:
                continue
            union.update(str(document_id) for document_id in self.db.collection_document_ids(collection_id))
        if not union:
            raise AdapterError(COLLECTION_EMPTY, "selected collections have no documents")
        union -= self.disabled_document_ids()  # FE-2: disabled docs are never searched
        if not union:
            raise AdapterError(COLLECTION_EMPTY, "every document in the selected collections is disabled")
        return union

    def list_docs_multi(self, collection_ids: Sequence[str]) -> list[DocInfo]:
        """``list_docs`` over the union of several collections, de-duplicated by
        document_id (R7): the multi-collection system prompt lists every in-scope
        PDF exactly once."""

        allowed: set[str] = set()
        for collection_id in collection_ids:
            if not collection_id:
                continue
            allowed.update(str(document_id) for document_id in self.db.collection_document_ids(collection_id))
        if not allowed:
            return []
        infos: list[DocInfo] = []
        for record in self.registry.list_documents():
            if record.document_id not in allowed:
                continue
            infos.append(
                DocInfo(
                    document_id=record.document_id,
                    filename=record.original_filename,
                    status=str(getattr(record.status, "value", record.status)),
                    page_count=record.page_count,
                    chunk_count=record.chunk_count,
                    failure_code=record.failure_code,
                )
            )
        return infos

    def list_docs(self, collection_id: str) -> list[DocInfo]:
        allowed = set(self.db.collection_document_ids(collection_id))
        if not allowed:
            return []
        infos: list[DocInfo] = []
        for record in self.registry.list_documents():
            if record.document_id not in allowed:
                continue
            infos.append(
                DocInfo(
                    document_id=record.document_id,
                    filename=record.original_filename,
                    status=str(getattr(record.status, "value", record.status)),
                    page_count=record.page_count,
                    chunk_count=record.chunk_count,
                    failure_code=record.failure_code,
                )
            )
        return infos

    # ------------------------------------------------------------- digest (第 7 条)
    def digest(
        self,
        collection_ids: Sequence[str],
        *,
        max_sections: int = 6,
        excerpt_chars: int = 140,
    ) -> str:
        """A short, stable description of what is in scope.

        问题.md 第二轮第 7 条: the material should be declared up front the way a
        tool schema is declared, instead of being discovered by searching.  Kept to
        filenames, sizes, section headings and the opening snippet of each PDF --
        small enough to freeze into the session prompt and stable enough that the
        prompt prefix does not drift between turns.
        """

        docs = self.list_docs_multi(collection_ids)
        disabled = self.disabled_document_ids()  # FE-2: declare only usable material
        docs = [doc for doc in docs if doc.document_id not in disabled]
        if not docs:
            return ""
        lines: list[str] = []
        for doc in docs:
            pages = doc.page_count if doc.page_count is not None else "?"
            lines.append(f"- {doc.filename} ({pages} pages, {doc.chunk_count} chunks)")
            sections, excerpt = self._chunk_digest(doc.document_id, excerpt_chars=excerpt_chars)
            if sections:
                lines.append(f"    sections: {' | '.join(sections[:max_sections])}")
            if excerpt:
                lines.append(f"    opens with: {excerpt}")
        return "\n".join(lines)

    def _chunk_digest(self, document_id: str, *, excerpt_chars: int = 140) -> tuple[list[str], str]:
        """Distinct section names plus the opening text of one document.

        Reads the chunk artifacts rather than the index so the digest needs no
        retrieval stack and cannot be affected by ranking settings.
        """

        data_dir = getattr(self.registry, "data_dir", None)
        if not data_dir:
            return [], ""
        path = Path(data_dir) / "parsed" / document_id / "chunks.jsonl"
        if not path.is_file():
            return [], ""
        sections: list[str] = []
        excerpt = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not excerpt:
                # D12: one text field in the artifact now; the legacy names stay in the
                # chain so chunks.jsonl written before the change still reads.
                text = str(
                    row.get("retrieval_text") or row.get("context_text") or row.get("text") or ""
                ).strip()
                excerpt = " ".join(text.split())[:excerpt_chars]
            section = str(row.get("section") or "").strip()
            if section and section != "Unknown" and section not in sections:
                sections.append(section)
        return sections, excerpt
