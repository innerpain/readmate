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
from src.ingestion.ordered_chunker import (
    CHUNKER_VERSION,
    PROSE_MAX_TOKENS,
    PROSE_OVERLAP_TOKENS,
    PROSE_TARGET_TOKENS,
    chunk_ordered_document,
    write_chunks_artifacts,
)
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


def chunking_config() -> dict[str, Any]:
    """The chunker's own fingerprint, in one place (D18).

    The publish call used to inline these numbers, which meant the manifest could
    say one thing while the chunker did another.  The reuse decision compares this
    against the previous manifest, so it has to be the real thing.
    """

    return {
        "chunker_version": CHUNKER_VERSION,
        "prose_target_tokens": PROSE_TARGET_TOKENS,
        "prose_max_tokens": PROSE_MAX_TOKENS,
        "prose_overlap_tokens": PROSE_OVERLAP_TOKENS,
    }


# Anything here changing invalidates every document's chunks: the text a chunker
# produces is a function of the element stream, the chunker and the embedding model.
_FINGERPRINT_KEYS = ("chunker_version", "chunking_config", "parser_config", "element_schema_version", "embedding_model")
ELEMENT_SCHEMA_VERSION = "ordered_document_v1"


@dataclass
class ReusePlan:
    """D18 step 1: which documents may keep the ``chunks.jsonl`` they already have.

    Chunk ids are content-addressed, so an unchanged document's chunks are the same
    chunks -- re-running the chunker for it only burns CPU.  Reusing them is what
    makes a one-file upload cheap; the counters are logged so the decision can be
    audited against reality instead of trusted.
    """

    unchanged: set[str]
    reasons: dict[str, str]
    prev_version: str | None = None
    chunks_reused: int = 0

    @property
    def global_fingerprint_ok(self) -> bool:
        return self.prev_version is not None and not any(
            reason.startswith("fingerprint") for reason in self.reasons.values()
        )


def _load_chunks_jsonl(path: Path) -> list[dict[str, Any]] | None:
    """Read a stored ``chunks.jsonl``; ``None`` when it is missing or unreadable.

    A half-written or truncated artifact must never be reused -- the caller falls
    back to re-chunking, which is the only path that can be trusted.
    """

    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "chunk_id" not in row:
                return None
            rows.append(row)
    except (OSError, json.JSONDecodeError):
        return None
    return rows or None


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

    # ------------------------------------------------------------------ D18 reuse
    def _prev_manifest(self) -> tuple[dict[str, Any] | None, str | None]:
        """The manifest the ``current.json`` pointer names, plus its version id."""

        pointer = self.index_dir / "current.json"
        try:
            version_id = str(json.loads(pointer.read_text(encoding="utf-8")).get("version_id") or "")
        except (OSError, json.JSONDecodeError):
            return None, None
        if not version_id:
            return None, None
        try:
            manifest = json.loads(
                (self.index_dir / "versions" / version_id / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return None, version_id
        return manifest, version_id

    def _reuse_plan(self, gathered: list[tuple[Any, dict[str, Any], Path]], parser_config: dict[str, Any]) -> ReusePlan:
        """Which documents may keep the ``chunks.jsonl`` they already have (D18 step 1).

        Reuse needs **every** fingerprint to match -- the chunker and its token
        budget, the parser config, the element schema and the embedding model.  A
        mismatch means the stored text may no longer be what the current code would
        produce, so the whole library is re-chunked: a wasted minute is cheaper than
        an index nobody can explain.

        Per document the extra condition is ``revision``: the registry bumps it on
        every (re)parse, so an equal revision means "same file, parsed the same way".
        """

        plan = ReusePlan(unchanged=set(), reasons={})
        manifest, version_id = self._prev_manifest()
        plan.prev_version = version_id
        if manifest is None:
            plan.reasons["*"] = "no previous manifest"
            return plan

        current = {
            "chunker_version": CHUNKER_VERSION,
            "chunking_config": chunking_config(),
            "parser_config": parser_config,
            "element_schema_version": ELEMENT_SCHEMA_VERSION,
            "embedding_model": get_model_settings().embedding_model,
        }
        for key in _FINGERPRINT_KEYS:
            if manifest.get(key) != current[key]:
                plan.reasons["*"] = f"fingerprint changed: {key}"
                return plan

        prev_docs = {
            str(entry.get("document_id")): entry
            for entry in (manifest.get("documents") or [])
            if isinstance(entry, dict)
        }
        for document, _ordered, out_dir in gathered:
            previous = prev_docs.get(document.document_id)
            if previous is None:
                plan.reasons[document.document_id] = "new document"
                continue
            if str(previous.get("revision") or "") != str(document.revision or ""):
                plan.reasons[document.document_id] = "revision changed"
                continue
            rows = _load_chunks_jsonl(out_dir / "chunks.jsonl")
            if rows is None:
                plan.reasons[document.document_id] = "chunks.jsonl missing or unreadable"
                continue
            if int(previous.get("chunk_count") or -1) != len(rows):
                plan.reasons[document.document_id] = (
                    f"chunk count {len(rows)} != manifest {previous.get('chunk_count')}"
                )
                continue
            plan.unchanged.add(document.document_id)
            plan.reasons[document.document_id] = "reused"
        return plan

    def _assemble_embeddings(
        self,
        index_chunks: list[IndexChunk],
        retrieval_texts: list[str],
        plan: ReusePlan,
    ) -> tuple[Any, int]:
        """D18 step 2: embed what changed, copy the rest out of the previous version.

        A chunk whose id already exists in the published index *is* the same chunk
        (ids hash the document revision and the chunk's own text), so its vector is
        already correct and re-computing it is pure GPU time.  Every candidate is
        looked up and anything missing is embedded for real -- the failure mode is
        "slower", never "wrong".
        """

        import numpy as np

        if not index_chunks:
            return np.empty((0, 0), dtype="float32"), 0

        candidates = [
            str(chunk.metadata.get("chunk_id"))
            for chunk in index_chunks
            if str(chunk.metadata.get("document_id")) in plan.unchanged
        ]
        reused: dict[str, list[float]] = {}
        if candidates and plan.prev_version:
            previous = self._open_prev_store()
            if previous is not None:
                try:
                    reused = previous.vectors_for(candidates)
                finally:
                    # The GC deletes this version right after the publish; release the
                    # handles now so the removal is not refused (D14b).
                    previous.close()
        if not reused:
            return self.embedder.embed_texts(retrieval_texts), 0

        dimension = len(next(iter(reused.values())))
        matrix = np.zeros((len(index_chunks), dimension), dtype="float32")
        pending_index: list[int] = []
        pending_texts: list[str] = []
        for position, chunk in enumerate(index_chunks):
            vector = reused.get(str(chunk.metadata.get("chunk_id")))
            if vector is None or len(vector) != dimension:
                pending_index.append(position)
                pending_texts.append(retrieval_texts[position])
                continue
            matrix[position] = vector

        if pending_texts:
            fresh = self.embedder.embed_texts(pending_texts)
            # Defensive: a dimension change the fingerprint could not see (an env
            # override of the model) would otherwise mix incompatible rows.  Re-embed
            # everything rather than publish a matrix nobody can explain.
            if getattr(fresh, "shape", (0, 0))[0] != len(pending_texts) or (
                len(pending_texts) and fresh.shape[1] != dimension
            ):
                logger.warning("vector reuse rejected: dimension mismatch, re-embedding all")
                return self.embedder.embed_texts(retrieval_texts), 0
            for row, position in enumerate(pending_index):
                matrix[position] = fresh[row]
        return matrix, len(index_chunks) - len(pending_index)

    def _open_prev_store(self):  # -> ChromaVectorStore | None
        """The published store, or ``None`` when there is nothing to reuse from."""

        try:
            store, _manifest = ChromaVectorStore.load(self.index_dir)
        except Exception as error:  # noqa: BLE001 - reuse is optional, never required
            logger.warning("previous index not readable, embedding everything: %s", error)
            return None
        return store

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

            # D18 step 1: read every document's element stream first, so the reuse
            # decision sees the whole picture (it needs the parser fingerprint) before
            # any chunking happens.
            gathered: list[tuple[Any, dict[str, Any], Path]] = []
            parser_config: dict[str, Any] = {}
            for document in all_documents:
                ordered_path = self.registry.data_dir / "parsed" / document.document_id / "ordered.json"
                if not ordered_path.is_file():
                    logger.warning("skip document without ordered.json document_id=%s", document.document_id)
                    continue
                ordered = json.loads(ordered_path.read_text(encoding="utf-8"))
                if not parser_config:
                    parser_config = dict(ordered.get("parser") or {})
                gathered.append((document, ordered, ordered_path.parent))

            plan = self._reuse_plan(gathered, parser_config)
            started = time.monotonic()

            index_chunks: list[IndexChunk] = []
            per_doc_counts: dict[str, int] = {}
            chunked_documents = 0
            # D16: the manifest carries the index's own quality summary now.
            quality_summary = _new_quality_summary()
            for document, ordered, out_dir in gathered:
                chunks: list[dict[str, Any]] | None = None
                if document.document_id in plan.unchanged:
                    chunks = _load_chunks_jsonl(out_dir / "chunks.jsonl")
                    if chunks is None:
                        # The artifact the plan trusted is not actually usable; fall
                        # back to re-chunking rather than publishing a guess.
                        plan.unchanged.discard(document.document_id)
                        plan.reasons[document.document_id] = "unreadable chunks.jsonl"
                if chunks is None:
                    chunks, report = chunk_ordered_document(
                        ordered,
                        filename=document.original_filename or document.stored_filename,
                    )
                    write_chunks_artifacts(out_dir, chunks, report)
                    chunked_documents += 1
                _accumulate_quality(quality_summary, document.document_id, out_dir)
                per_doc_counts[document.document_id] = len(chunks)
                for chunk in chunks:
                    index_chunks.append(_to_index_chunk(chunk))

            plan.chunks_reused = sum(per_doc_counts.get(doc_id, 0) for doc_id in plan.unchanged)

            if progress_callback is not None:
                progress_callback(IngestionStage.EMBEDDING)

            retrieval_texts = [str(c.metadata.get("retrieval_text") or c.page_content) for c in index_chunks]
            embeddings, reused_vectors = self._assemble_embeddings(index_chunks, retrieval_texts, plan)
            logger.info(
                "index reuse: unchanged_docs=%d/%d reused_chunks=%d/%d chunked_docs=%d "
                "reused_vectors=%d embedded=%d prev_version=%s elapsed=%.1fs",
                len(plan.unchanged), len(gathered), plan.chunks_reused, len(index_chunks),
                chunked_documents, reused_vectors, len(index_chunks) - reused_vectors,
                plan.prev_version or "-", time.monotonic() - started,
            )
            if not plan.unchanged:
                # Worth a line of its own: "nothing was reused" with no reason is
                # exactly the ambiguity this feature was meant to remove.
                logger.info(
                    "index reuse: nothing reused (%s)",
                    plan.reasons.get("*") or "every document failed its own check",
                )
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
                chunking_config=chunking_config(),
                parser_config=parser_config,
                quality_config=quality_summary,
                documents=documents_meta,
                element_schema_version=ELEMENT_SCHEMA_VERSION,
                chunker_version=CHUNKER_VERSION,
                embedding_config=self.embedder.runtime_config(),
                reuse_stats={
                    "reused_documents": len(plan.unchanged),
                    "reused_chunks": plan.chunks_reused,
                    "reused_vectors": reused_vectors,
                    "chunked_documents": chunked_documents,
                    "previous_version": plan.prev_version,
                },
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
                # D18: what the reuse decision actually did, so a caller (and the log
                # above) can tell an incremental build from a full one.
                "reused_documents": sorted(plan.unchanged),
                "reused_chunks": plan.chunks_reused,
                "chunked_documents": chunked_documents,
                "reused_vectors": reused_vectors,
                "embedded_chunks": len(index_chunks) - reused_vectors,
                "reuse_prev_version": plan.prev_version,
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
        # D61: what that number means (page / slide / sheet / section).
        "page_kind": str(chunk.get("page_kind") or "page"),
        # D63: a heading-less document (CSV, or a DOCX/MD/HTML with no headings)
        # carries an empty section -- never the word "Unknown".
        "section": chunk.get("section") or "",
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
