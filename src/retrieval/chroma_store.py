"""Versioned local ChromaDB storage for document chunks."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


COLLECTION_NAME = "document_chunks"


def _release_chroma_client(client) -> None:
    """Best effort: drop Chroma's file handles so the staged dir can be renamed.

    D14b (2026-09-21): on Docker Desktop's Windows bind mount an open
    ``chroma.sqlite3`` makes ``os.replace()`` of the directory fail with EACCES.
    Clearing the shared system releases it.  Never fatal -- the copy fallback in
    :meth:`ChromaVectorStore.publish` covers the case where this does not work.
    """

    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception as error:  # noqa: BLE001 - a private-ish API; absence is fine
        logger.debug("could not release the chroma client: %s", error)


def _gc_versions(root: Path, *, keep: str, stale_staging_after_s: float = 3600.0) -> int:
    """D14: one published version is the product's contract, so older ones go.

    Never touches ``keep`` (the version the pointer now names), and never raises: a
    version that cannot be removed (a reader still holding a handle on Windows) is
    left for the next rebuild.

    A ``.building-*`` directory belongs to a build in flight and is normally spared --
    but a build that *crashed* leaves one behind forever, which is the same orphan
    problem D14 set out to fix.  Only staging dirs older than
    ``stale_staging_after_s`` (default: the index-build lock TTL) are reclaimed, so a
    concurrent build's directory is still safe.
    """

    removed = 0
    versions_dir = root / "versions"
    now = datetime.now(timezone.utc).timestamp()
    try:
        for entry in versions_dir.iterdir():
            if not entry.is_dir() or entry.name == keep:
                continue
            if entry.name.startswith("."):
                try:
                    if now - entry.stat().st_mtime <= stale_staging_after_s:
                        continue
                except OSError:
                    continue
            try:
                shutil.rmtree(entry)
                removed += 1
            except OSError:
                continue
    except OSError:
        return removed
    return removed


class ChromaVectorStore:
    """Publish immutable Chroma collections and query the current version."""

    def __init__(self, root: Path | str):
        import chromadb
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        current = self.root / "current.json"
        if not current.is_file():
            raise FileNotFoundError("chroma_current_missing")
        payload = json.loads(current.read_text(encoding="utf-8"))
        version_id = str(payload["version_id"])
        version_dir = self.root / "versions" / version_id
        if not version_dir.is_dir():
            raise FileNotFoundError("chroma_version_missing")
        self.version_id = version_id
        self.client = chromadb.PersistentClient(path=str(version_dir))
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))

    @classmethod
    def publish(cls, index_root, chunks, embeddings, embedding_model, chunking_config, parser_config, quality_config, documents, element_schema_version="legacy", chunker_version="legacy", embedding_config=None):
        root = Path(index_root)
        version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        # D14 (2026-09-20): build in a temporary directory, publish by rename.
        # A crash mid-build used to leave a half-written version behind (and a
        # rebuild left the previous one forever -- measured: 11 orphans, 34 MB).
        # The directory name starts with "." so it is never mistaken for a
        # version, and os.replace() makes the publish atomic on one filesystem.
        staging_dir = root / "versions" / f".building-{uuid4().hex[:8]}"
        version_dir = root / "versions" / version_id
        staging_dir.mkdir(parents=True, exist_ok=False)
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(staging_dir))
            collection = client.get_or_create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            ids = [str(c.metadata["chunk_id"]) for c in chunks]
            if ids:
                collection.add(
                    ids=ids,
                    documents=[c.page_content for c in chunks],
                    embeddings=embeddings.tolist(),
                    metadatas=[_metadata(c.metadata) for c in chunks],
                )
            manifest = {
                "version_id": version_id,
                "embedding_model": embedding_model,
                # Batch sizing decides peak GPU memory; record it so an audit can
                # tell which runtime settings actually produced this index.
                "embedding_config": embedding_config or {},
                "embedding_dimension": int(embeddings.shape[1]) if len(embeddings) else None,
                "normalize_embeddings": True,
                "chunk_count": len(chunks),
                "chunking_config": chunking_config,
                "parser_config": parser_config,
                "quality_config": quality_config,
                "element_schema_version": element_schema_version,
                "chunker_version": chunker_version,
                "documents": documents,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (staging_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            # Atomic publish: the staging dir becomes the version, then the pointer
            # flips.  A failure before this point leaves the old index untouched.
            #
            # D14b (2026-09-21): the rename fails on Docker Desktop's Windows bind
            # mount while the Chroma client still holds chroma.sqlite3 open --
            # measured as ``PermissionError: [Errno 13]`` during a real rebuild, i.e.
            # this path had never actually run on this machine.  Release the client
            # first, and fall back to a copy when the filesystem refuses the rename.
            # Either way the version only becomes visible through the pointer flip
            # below, so a partially published version can never be loaded.
            _release_chroma_client(client)
            try:
                os.replace(staging_dir, version_dir)
            except OSError as error:
                logger.warning("atomic rename unavailable (%s); copying the staged index", error)
                shutil.copytree(staging_dir, version_dir)
                shutil.rmtree(staging_dir, ignore_errors=True)
            pointer = root / "current.json"
            temporary = root / f".current-{uuid4().hex}.tmp"
            temporary.write_text(json.dumps({"version_id": version_id}, indent=2), encoding="utf-8")
            os.replace(temporary, pointer)
            _gc_versions(root, keep=version_id)
            return manifest
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    @classmethod
    def load(cls, root):
        store = cls(root)
        return store, store.manifest

    def query(self, query_embedding, k: int):
        result = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=k, include=["documents", "metadatas", "distances"])
        return result

    def all_chunks(self) -> tuple[list[str], list[str]]:
        """Every published chunk id and text, for building a derived index."""

        result = self.collection.get(include=["documents"])
        ids = [str(item) for item in (result.get("ids") or [])]
        documents = [str(item) for item in (result.get("documents") or [])]
        if len(documents) < len(ids):
            documents.extend([""] * (len(ids) - len(documents)))
        return ids, documents

    def siblings_by_table(self, table_id: str) -> list[dict]:
        """Every chunk of one table, in pack order (D4).

        A long table is published as one ``table_summary`` plus N ``table_pack``
        shards sharing a ``table_id``.  Diversity deliberately keeps a single
        retrieval slot per ``table_id``, so without this the other shards are
        unreachable and the model only ever sees one slice of the table.
        Returns [] whenever the filter is unsupported or the table is unknown --
        a lookup failure must never fail a search.
        """

        if not table_id:
            return []
        try:
            result = self.collection.get(where={"table_id": str(table_id)}, include=["documents", "metadatas"])
        except Exception:  # noqa: BLE001 - unsupported filter / backend quirk
            return []
        ids = [str(item) for item in (result.get("ids") or [])]
        documents = [str(item) for item in (result.get("documents") or [])]
        metadatas = list(result.get("metadatas") or [])
        rows: list[dict] = []
        for index, chunk_id in enumerate(ids):
            metadata = dict(metadatas[index]) if index < len(metadatas) and metadatas[index] else {}
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": metadata,
                }
            )
        try:
            rows.sort(key=lambda row: int(row["metadata"].get("pack_ordinal", -1)))
        except (TypeError, ValueError):
            pass
        return rows

    def get_by_ids(self, chunk_ids):
        """Fetch specific chunks (text + metadata) without a similarity search."""

        wanted = [str(chunk_id) for chunk_id in chunk_ids]
        if not wanted:
            return {}
        result = self.collection.get(ids=wanted, include=["documents", "metadatas"])
        ids = [str(item) for item in (result.get("ids") or [])]
        documents = list(result.get("documents") or [])
        metadatas = list(result.get("metadatas") or [])
        fetched = {}
        for index, chunk_id in enumerate(ids):
            document = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            fetched[chunk_id] = {"document": document, "metadata": dict(metadata)}
        return fetched


def _metadata(metadata: dict) -> dict:
    output = {}
    keys = (
        "chunk_id",
        "document_id",
        "filename",
        "page",
        # A3: the last page of a passage that straddles a page break.  Omitting it
        # from this whitelist silently dropped the span at publish time, so every
        # citation fell back to the start page.
        "page_end",
        "section",
        "text",
        "chunk_type",
        "heading_path",
        "element_ids",
        "provenance",
        "retrieval_text",
        "quality_score",
        "chunk_ordinal",
        "text_source",
        "source_parser",
        "ocr_status",
        "atomic",
        "degraded",
        "parse_status",
        "table_id",
        "figure_id",
        "figure_type",
        "structure_source",
        "content_kind",
        "image_path",
        "page_kind",
        "caption",
        "parent_id",
        "pack_ordinal",
        "neighbor_prev_chunk_ids",
        "neighbor_next_chunk_ids",
        "chunker_version",
    )
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            value = 1 if value else 0
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        output[key] = value
    if metadata.get("degraded_structure"):
        # Only stored when true: older snapshots then keep their exact shape.
        output["degraded_structure"] = 1
    structure_pages = metadata.get("structure_pages")
    if structure_pages:
        output["structure_pages"] = json.dumps(list(structure_pages), ensure_ascii=False)
    for key in ("issue_codes", "region_ids", "ocr_region_ids"):
        value = metadata.get(key)
        if value is not None:
            output[key] = json.dumps(value, ensure_ascii=False)
    return output
