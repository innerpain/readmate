"""Stage-2 index build: ordered.json -> chunks -> embed -> Chroma publish."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.config.settings import get_ingestion_settings, get_model_settings
from src.ingestion.document_parser import DocumentParser
from src.ingestion.ordered_chunker import CHUNKER_VERSION, chunk_ordered_document, write_chunks_artifacts
from src.models.schemas import IngestionStage, IngestionStatus
from src.retrieval.chroma_store import ChromaVectorStore
from src.retrieval.embedding_generator import EmbeddingGenerator
from src.storage.document_registry import DocumentRegistry

logger = logging.getLogger(__name__)


class IndexBuildBusyError(RuntimeError):
    """Raised when another publish holds the rebuild lock."""


@dataclass
class IndexChunk:
    """Chroma publish adapter: page_content + metadata."""

    page_content: str
    metadata: dict[str, Any]


def _new_quality_summary() -> dict[str, Any]:
    """D16 (2026-09-20): the accumulator for the index-level quality summary."""

    return {
        "documents": 0,
        "elements": 0,
        "degraded_documents": 0,
        "table_unparsed": 0,
        "formula_unparsed": 0,
        "dropped_embed_candidates": 0,
        "formula_fixes": 0,
        "figure_path_fixes": 0,
        "roles": {},
        "parse_status": {},
        "unreadable_reports": [],
        "per_document": {},
    }


def _accumulate_quality(summary: dict[str, Any], document_id: str, out_dir: Path) -> None:
    """Fold one document's ``quality_report.json`` into the index-level summary.

    D16: the manifest used to be published with ``quality_config={}``, so an index
    could not say anything about how well its documents had parsed -- "is this index
    built on degraded parses?" was unanswerable from the index itself.  A missing or
    corrupt report is recorded as unreadable rather than failing the build: the index
    is still valid, it just cannot vouch for that document.
    """

    summary["documents"] += 1
    entry = {
        "elements": 0,
        "table_unparsed": 0,
        "formula_unparsed": 0,
        "dropped_embed_candidates": 0,
    }
    try:
        report = json.loads((out_dir / "quality_report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("quality report unreadable for %s: %s", document_id, error)
        summary["unreadable_reports"].append(document_id)
        summary["per_document"][document_id] = entry
        return

    stats = report.get("stats") if isinstance(report, dict) else None
    quality = report.get("quality") if isinstance(report, dict) else None
    stats = stats if isinstance(stats, dict) else {}
    quality = quality if isinstance(quality, dict) else {}

    entry["elements"] = int(stats.get("n_elements") or 0)
    summary["elements"] += entry["elements"]
    for key in ("table_unparsed", "formula_unparsed", "dropped_embed_candidates"):
        value = int(quality.get(key) or 0)
        entry[key] = value
        summary[key] += value
    summary["formula_fixes"] += len(quality.get("formula_fixes") or [])
    summary["figure_path_fixes"] += int(quality.get("figure_path_fixes") or 0)
    for role, count in (quality.get("role_counts") or {}).items():
        summary["roles"][str(role)] = summary["roles"].get(str(role), 0) + int(count or 0)
    for status, count in (quality.get("parse_status_counts") or {}).items():
        summary["parse_status"][str(status)] = summary["parse_status"].get(str(status), 0) + int(count or 0)
    if entry["table_unparsed"] or entry["formula_unparsed"]:
        summary["degraded_documents"] += 1
    summary["per_document"][document_id] = entry


class IndexBuilder:
    """Parse (Docling ordered) then chunk/embed/publish a versioned snapshot."""

    def __init__(
        self,
        registry: DocumentRegistry,
        *,
        embedder: EmbeddingGenerator | None = None,
        index_dir: Path | str | None = None,
        **_ignored: object,
    ) -> None:
        self.registry = registry
        self.parser = DocumentParser(registry)
        self.embedder = embedder or EmbeddingGenerator.from_settings(get_model_settings())
        project_root = Path(__file__).resolve().parents[2]
        self.index_dir = Path(index_dir) if index_dir is not None else project_root / "data" / "chroma"
        self._lock_path = self.index_dir / ".build.lock"

    def build(
        self,
        *,
        target_document_id: str | None = None,
        progress_callback: Callable[[IngestionStage], None] | None = None,
        skip_parse: bool = False,
    ) -> dict[str, object]:
        self._acquire_lock()
        try:
            if progress_callback is not None:
                progress_callback(IngestionStage.PARSING)

            if target_document_id:
                if not skip_parse:
                    self.parser.parse_document(target_document_id)
                documents = [self.registry.get_document(target_document_id)]
            else:
                if not skip_parse:
                    self.parser.parse_all()
                documents = list(self.registry.list_documents())

            # Full snapshot from every document that has ordered.json (keeps multi-doc index coherent).
            all_documents = list(self.registry.list_documents())
            if progress_callback is not None:
                progress_callback(IngestionStage.CHUNKING)

            index_chunks: list[IndexChunk] = []
            per_doc_counts: dict[str, int] = {}
            parser_config: dict[str, Any] = {}
            # D16: the manifest carries the index's own quality summary now.
            quality_summary = _new_quality_summary()
            for document in all_documents:
                ordered_path = self.registry.data_dir / "parsed" / document.document_id / "ordered.json"
                if not ordered_path.is_file():
                    logger.warning("skip document without ordered.json document_id=%s", document.document_id)
                    continue
                ordered = json.loads(ordered_path.read_text(encoding="utf-8"))
                if not parser_config:
                    parser_config = dict(ordered.get("parser") or {})
                chunks, report = chunk_ordered_document(
                    ordered,
                    filename=document.original_filename or document.stored_filename,
                )
                out_dir = ordered_path.parent
                write_chunks_artifacts(out_dir, chunks, report)
                _accumulate_quality(quality_summary, document.document_id, out_dir)
                per_doc_counts[document.document_id] = len(chunks)
                for chunk in chunks:
                    index_chunks.append(_to_index_chunk(chunk))

            if progress_callback is not None:
                progress_callback(IngestionStage.EMBEDDING)

            retrieval_texts = [str(c.metadata.get("retrieval_text") or c.page_content) for c in index_chunks]
            if retrieval_texts:
                embeddings = self.embedder.embed_texts(retrieval_texts)
            else:
                import numpy as np

                embeddings = np.empty((0, 0), dtype="float32")

            if progress_callback is not None:
                progress_callback(IngestionStage.INDEXING)

            model_settings = get_model_settings()
            documents_meta = [
                {
                    "document_id": doc.document_id,
                    "filename": doc.original_filename,
                    "revision": doc.revision,
                    "chunk_count": per_doc_counts.get(doc.document_id, 0),
                }
                for doc in all_documents
            ]
            manifest = ChromaVectorStore.publish(
                self.index_dir,
                index_chunks,
                embeddings,
                embedding_model=model_settings.embedding_model,
                chunking_config={
                    "chunker_version": CHUNKER_VERSION,
                    "prose_target_tokens": 320,
                    "prose_max_tokens": 512,
                    "prose_overlap_tokens": 64,
                },
                parser_config=parser_config,
                quality_config=quality_summary,
                documents=documents_meta,
                element_schema_version="ordered_document_v1",
                chunker_version=CHUNKER_VERSION,
                embedding_config=self.embedder.runtime_config(),
            )

            for document in all_documents:
                count = per_doc_counts.get(document.document_id)
                if count is None:
                    continue
                self.registry.update_document(
                    document.document_id,
                    status=IngestionStatus.COMPLETED,
                    stage=IngestionStage.COMPLETED,
                    chunk_count=count,
                    clear_failure_code=True,
                )

            if progress_callback is not None:
                progress_callback(IngestionStage.COMPLETED)

            result: dict[str, object] = {
                "chunk_count": int(manifest.get("chunk_count") or len(index_chunks)),
                "version_id": manifest.get("version_id"),
                "chunker_version": CHUNKER_VERSION,
                "parse_only": False,
                "document_counts": per_doc_counts,
            }
            if target_document_id:
                ordered_path = self.registry.data_dir / "parsed" / target_document_id / "ordered.json"
                result["document_id"] = target_document_id
                result["ordered_path"] = str(ordered_path)
                result["element_count"] = per_doc_counts.get(target_document_id, 0)
            else:
                result["document_count"] = len(all_documents)
            # silence unused local in single-target path documentation
            _ = documents
            return result
        finally:
            self._release_lock()

    def _acquire_lock(self) -> None:
        """Take the rebuild lock, or take over one that can no longer be live (D1).

        The lock file names its holder (pid, host, start time).  A lock that is
        expired, unreadable, or whose owner is gone is *stale*: a rebuild killed
        midway (OOM, container restart, Ctrl-C) used to leave the file behind and
        every later rebuild answered ``index_build_busy`` until a human deleted it
        by hand.
        """

        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": time.time(),
        }
        try:
            handle = open(self._lock_path, "x", encoding="utf-8")
        except FileExistsError:
            reason = self._stale_reason()
            if reason is None:
                raise IndexBuildBusyError("index_build_busy") from None
            logger.warning("taking over stale build lock path=%s reason=%s", self._lock_path, reason)
            self._lock_path.write_text(json.dumps(payload), encoding="utf-8")
            return
        with handle:
            handle.write(json.dumps(payload))

    def _stale_reason(self) -> str | None:
        """Why the existing lock may be taken over, or ``None`` while it is live."""

        try:
            raw = self._lock_path.read_text(encoding="utf-8")
        except OSError:
            return "unreadable"
        try:
            holder = json.loads(raw)
        except json.JSONDecodeError:
            # Pre-D1 locks held the literal text ``building``; a truncated write
            # lands here too, and neither can be validated.
            return "unparseable"
        if not isinstance(holder, dict):
            return "unparseable"
        started_at = float(holder.get("started_at") or 0.0)
        if started_at <= 0:
            return "no_start_time"
        ttl = self._lock_ttl_s()
        age = time.time() - started_at
        if age > ttl:
            return f"expired age={int(age)}s ttl={int(ttl)}s"
        same_host = str(holder.get("host") or "") == socket.gethostname()
        if same_host and not _pid_alive(int(holder.get("pid") or 0)):
            return f"holder_gone pid={holder.get('pid')}"
        return None

    def _lock_ttl_s(self) -> float:
        """Lock lifetime from settings; the default is used if the env is broken."""

        try:
            return float(get_ingestion_settings().index_build_lock_ttl_s)
        except Exception:  # noqa: BLE001 - a bad env must not block every rebuild
            logger.warning("index_build_lock_ttl_s unreadable, falling back to 3600s")
            return 3600.0

    def _release_lock(self) -> None:
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("failed to remove build lock path=%s", self._lock_path)


def _pid_alive(pid: int) -> bool:
    """Liveness probe for a lock holder on this host (D1).

    ``kill(pid, 0)`` sends no signal: it raises ``ProcessLookupError`` when the
    process is gone and ``PermissionError`` when it exists but is not ours (still
    a live holder, so the lock stays).
    """

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _to_index_chunk(chunk: dict[str, Any]) -> IndexChunk:
    metadata = {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk.get("document_id"),
        "filename": chunk.get("filename"),
        "page": int(chunk.get("page") or 1),
        # A3: the last page this chunk covers (== page when it does not straddle
        # a page break).  Without it a citation could only name the start page.
        "page_end": int(chunk.get("page_end") or chunk.get("page") or 1),
        "section": chunk.get("section") or "Unknown",
        "chunk_type": chunk.get("chunk_type"),
        "heading_path": chunk.get("heading_path") or [],
        "element_ids": chunk.get("element_ids") or [],
        "retrieval_text": chunk.get("retrieval_text") or "",
        "atomic": int(chunk.get("atomic") or 0),
        "degraded": int(chunk.get("degraded") or 0),
        "degraded_structure": bool(chunk.get("degraded")),
        "parse_status": chunk.get("parse_status") or "parsed",
        "table_id": chunk.get("table_id"),
        "figure_id": chunk.get("figure_id"),
        "content_kind": chunk.get("content_kind") or "text",
        "image_path": chunk.get("image_path") or "",
        "parent_id": chunk.get("parent_id") or "",
        "pack_ordinal": chunk.get("pack_ordinal", -1),
        "neighbor_prev_chunk_ids": chunk.get("neighbor_prev_chunk_ids") or [],
        "neighbor_next_chunk_ids": chunk.get("neighbor_next_chunk_ids") or [],
        "chunker_version": chunk.get("chunker_version") or CHUNKER_VERSION,
    }
    return IndexChunk(page_content=str(chunk.get("retrieval_text") or ""), metadata=metadata)
