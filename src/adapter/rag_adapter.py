"""The real RAG Adapter: wraps the retrieval stack, nothing else does."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from src.config.settings import get_agent_settings
from src.adapter.contracts import (
    ADAPTER_API_VERSION,
    BUILD_LOCKED,
    CHUNK_NOT_FOUND,
    DOCUMENT_NOT_FOUND,
    EMBEDDING_INCOMPATIBLE,
    INVALID_ARGUMENTS,
    KNOWLEDGE_BASE_EMPTY,
    AdapterError,
    DocInfo,
    ReadResult,
    SearchHit,
)
from src.adapter.route_planner import RetrievalRouter, compute_signals
from src.retrieval.chroma_store import ChromaVectorStore
from src.retrieval.persistent_retriever import (
    EmbeddingCompatibilityError,
    KnowledgeBaseUnavailableError,
    PersistentRetriever,
)

# D35 (2026-09-20): the read ceilings are settings now
# (``AgentSettings.adapter_page_chars`` / ``adapter_element_chars``).  They stay
# here as the fallback for callers that build the adapter without settings.
DEFAULT_PAGE_CHARS = 6000
MAX_ELEMENT_TEXT_CHARS = 4000


class RagAdapterImpl:
    api_version = ADAPTER_API_VERSION

    def __init__(
        self,
        *,
        retriever: PersistentRetriever,
        resolve_documents: Callable[[str], list[str]],
        registry,
        router: RetrievalRouter | None = None,
        index_dir: Path | str | None = None,
        parsed_dir: Path | str | None = None,
        excerpt_chars: int = 300,
        page_chars: int | None = None,
        element_chars: int | None = None,
    ) -> None:
        self.retriever = retriever
        self.resolve_documents = resolve_documents
        self.registry = registry
        self.router = router or RetrievalRouter(llm=None)   # rules only when no model
        self.last_route = None                              # last decide() result, for traces
        self.excerpt_chars = excerpt_chars
        agent = get_agent_settings()
        # D35: explicit argument wins, then settings, then the module fallback.
        self.page_chars = int(page_chars or agent.adapter_page_chars or DEFAULT_PAGE_CHARS)
        self.element_chars = int(element_chars or agent.adapter_element_chars or MAX_ELEMENT_TEXT_CHARS)
        # D33 (2026-09-20): element_id -> the ordered.json that owns it.  Building
        # it meant loading every ordered.json (0.5 MB each) on every element read;
        # the signature (path, mtime) invalidates it when a re-parse rewrites one.
        self._element_index_cache: tuple[tuple, dict[str, Path]] = ((), {})
        project_root = Path(__file__).resolve().parents[2]
        self.index_dir = Path(index_dir) if index_dir is not None else project_root / "data" / "chroma"
        self.parsed_dir = Path(parsed_dir) if parsed_dir is not None else project_root / "data" / "parsed"

    # ----------------------------------------------------------------- search
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
        scope = set(document_ids) if document_ids else set(self.resolve_documents(collection_id))
        if not scope:
            raise AdapterError("collection_empty", f"collection {collection_id} has no documents")
        # C5: the router must actually see ``mode`` and ``previous_empty_searches``
        # so "empty last search -> planned" can fire.  Defaults (mode=None -> "deep",
        # previous_empty_searches=0) keep the pre-R6 behaviour byte-for-byte.
        decision = self.router.decide(
            query,
            compute_signals(
                query,
                mode=mode or "deep",
                previous_empty_searches=int(previous_empty_searches or 0),
                collection_size=len(scope),
            ),
        )
        self.last_route = decision
        # A8: ``last_diagnostics`` already knows whether the weak-evidence floor
        # had to be used; surface it the same way ``last_route`` is surfaced so
        # the tool layer can tell the model the evidence is thin instead of
        # letting the retriever silently return nothing.
        self.last_low_confidence = False
        self.last_signals: dict = {}
        texts = decision.dense_texts(query)
        try:
            chunks = self.retriever.retrieve_multi(
                texts,
                final_k=top_k,
                document_ids=scope,
                # A4/D5: the user's raw question, not the model's rewrite, decides
                # whether tables pass the B/C gate.
                gate_query=gate_text or query,
            )
        except KnowledgeBaseUnavailableError as error:
            raise AdapterError(KNOWLEDGE_BASE_EMPTY, str(error)) from error
        except EmbeddingCompatibilityError as error:
            raise AdapterError(EMBEDDING_INCOMPATIBLE, str(error)) from error
        diagnostics = getattr(self.retriever, "last_diagnostics", None) or {}
        self.last_low_confidence = bool(diagnostics.get("low_confidence_applied"))
        # D24/D25/D31/D37 (2026-09-20): the retriever knows a lot more than the one
        # boolean it used to surface, and everything else -- a silently skipped
        # reranker, a wiped window, a routing downgrade -- was invisible to both
        # the model and the UI.  One dict, three consumers: the tool layer writes
        # it into the observation, the runtime turns it into answer warnings, and
        # the trace keeps it for diagnostics.
        self.last_signals = {
            "low_confidence_applied": bool(diagnostics.get("low_confidence_applied")),
            "low_confidence_count": int(diagnostics.get("low_confidence_count") or 0),
            "dropped_by_min_score": int(diagnostics.get("dropped_by_min_score") or 0),
            "rerank_enabled": bool(diagnostics.get("rerank_enabled")),
            "rerank_applied": bool(diagnostics.get("rerank_applied")),
            "judge_mismatch": bool(diagnostics.get("judge_mismatch")),
            "route_fallback": bool(getattr(decision, "fallback", False)),
        }
        return [_to_hit(chunk, self.excerpt_chars) for chunk in chunks if chunk.document_id in scope]

    # ------------------------------------------------------------------- read
    def read(
        self,
        *,
        chunk_id: str | None = None,
        document_id: str | None = None,
        page: int | None = None,
        element_id: str | None = None,
    ) -> ReadResult:
        if chunk_id:
            return self._read_chunk(chunk_id)
        if document_id and page:
            return self._read_page(document_id, int(page))
        if element_id:
            return self._read_element(element_id)
        raise AdapterError(INVALID_ARGUMENTS, "read needs chunk_id, document_id+page, or element_id")

    # -------------------------------------------------------------- list_docs
    def list_docs(self, collection_id: str) -> list[DocInfo]:
        scope = set(self.resolve_documents(collection_id))
        infos: list[DocInfo] = []
        for record in self.registry.list_documents():
            if record.document_id not in scope:
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

    # ------------------------------------------------------------- internals
    def _store(self):
        try:
            return ChromaVectorStore.load(self.index_dir)[0]
        except FileNotFoundError as error:
            raise AdapterError(KNOWLEDGE_BASE_EMPTY, "no published snapshot") from error
        except (KeyError, ValueError, OSError) as error:
            raise AdapterError(KNOWLEDGE_BASE_EMPTY, f"snapshot unreadable: {error}") from error

    def _read_chunk(self, chunk_id: str) -> ReadResult:
        fetched = self._store().get_by_ids([chunk_id])
        payload = fetched.get(chunk_id)
        if not payload:
            raise AdapterError(CHUNK_NOT_FOUND, chunk_id)
        metadata = payload.get("metadata") or {}
        document_id = str(metadata.get("document_id") or "")
        read_page = int(metadata.get("page") or 1) or 1
        read_page_end = int(metadata.get("page_end") or 0) or 0
        return ReadResult(
            document_id=document_id,
            chunk_id=chunk_id,
            page=read_page,
            page_end=read_page_end if read_page_end > read_page else None,
            section=str(metadata.get("section") or ""),
            text=str(payload.get("document") or ""),
            chunk_type=str(metadata.get("chunk_type") or ""),
            filename=self._resolve_filename(document_id, str(metadata.get("filename") or "")),
            chunk_ids=[chunk_id],
        )

    def _available_document_hint(self) -> str:
        """D58 (2026-09-21): a model that passes a collection id (``col_...``) or a
        filename where a document UUID belongs used to get "no chunks for <value>" --
        a dead end it could not recover from (measured: it burned one of six steps that
        way).  Naming the ids that *do* work turns the dead end into a self-correction."""

        try:
            records = list(self.registry.list_documents())
        except Exception:  # a hint must never break the read path
            return ""
        ids = [str(getattr(record, "document_id", "") or "") for record in records]
        ids = [value for value in ids if value]
        if not ids:
            return ""
        shown = ", ".join(ids[:6])
        more = f" (+{len(ids) - 6} more)" if len(ids) > 6 else ""
        return (
            f"Available document_id values: {shown}{more}. "
            "Pass one of these, or a chunk_id from a search hit -- not a collection id (col_...), not a filename."
        )

    def _read_page(self, document_id: str, page: int) -> ReadResult:
        document_id = self._resolve_document_id(str(document_id))
        path = self.parsed_dir / document_id / "chunks.jsonl"
        if not path.exists():
            hint = self._available_document_hint()
            raise AdapterError(
                DOCUMENT_NOT_FOUND,
                f"no document {document_id!r} in this collection." + (f" {hint}" if hint else ""),
            )
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                pages = [int(value) for value in (item.get("pages") or [])]
                if page in pages or int(item.get("page") or 0) == page:
                    rows.append(item)
        if not rows:
            raise AdapterError(CHUNK_NOT_FOUND, f"page {page} of {document_id} has no chunks")
        rows.sort(key=lambda item: (int(item.get("source_ordinal") or 10**9), int(item.get("pack_ordinal") or -1)))
        text = "\n\n".join(
            # D12: the chunk stores one text field now (``retrieval_text``); the old
            # names stay in the chain so an index or artifact published before the
            # change keeps working.
            str(item.get("retrieval_text") or item.get("context_text") or item.get("text") or "")
            for item in rows
        )
        truncated = len(text) > self.page_chars
        full_chars = len(text)
        remaining_chars = max(0, full_chars - self.page_chars) if truncated else 0
        chunk_ids = [str(item.get("chunk_id")) for item in rows if item.get("chunk_id")]
        page_filename = next((str(item.get("filename") or "") for item in rows if item.get("filename")), "")
        # A3: the rows of one page can themselves reach into the next page, so the
        # read reports how far it really goes.
        read_page_end = page
        for item in rows:
            try:
                candidate = int(item.get("page_end") or item.get("page") or page)
            except (TypeError, ValueError):
                continue
            read_page_end = max(read_page_end, candidate)
        return ReadResult(
            document_id=document_id,
            # No single chunk owns the whole page, so ``chunk_id`` stays None; the
            # citable ids live in ``chunk_ids`` so a page-read still reaches the
            # citation whitelist via :meth:`ToolRunner._read`.
            chunk_id=None,
            page=page,
            page_end=read_page_end if read_page_end > page else None,
            section=str(rows[0].get("section") or ""),
            text=text[: self.page_chars],
            chunk_type=str(rows[0].get("chunk_type") or ""),
            truncated=truncated,
            remaining_chars=remaining_chars or None,
            next_offset=self.page_chars if remaining_chars else None,
            filename=self._resolve_filename(document_id, page_filename),
            chunk_ids=chunk_ids,
        )

    def _resolve_filename(self, document_id: str, preferred: str) -> str:
        """Metadata / row filename first; fall back to the registry so a citation
        never ships with an empty ``filename`` when the id is known."""

        if preferred:
            return preferred
        if not document_id:
            return ""
        try:
            record = self.registry.get_document(document_id)
        except Exception:  # registry shape varies; absence is not fatal to the read
            return ""
        return str(getattr(record, "original_filename", "") or "")

    def _resolve_document_id(self, value: str) -> str:
        """A1 (D-9c): a page read may be addressed by *filename* when the model
        passes one instead of the UUID.  Resolve it through the registry; a
        genuine id (an existing parsed dir) is returned untouched."""

        if (self.parsed_dir / value).exists():
            return value
        try:
            records = list(self.registry.list_documents())
        except Exception:  # registry shape varies; a name that is not a real id
            return value      # still fails closed as DOCUMENT_NOT_FOUND downstream
        wanted = _name_variants(value)
        matched: dict[str, object] = {}
        for record in records:
            if wanted & _name_variants(str(getattr(record, "original_filename", "") or "")):
                matched[str(record.document_id)] = record
        ids = list(matched)
        if len(ids) == 1:
            return ids[0]
        if len(ids) > 1:
            raise AdapterError(
                INVALID_ARGUMENTS,
                f"document {value!r} is ambiguous across {len(ids)} ids; pass the UUID from search/list_docs",
            )
        return value

    def _element_index(self) -> dict[str, Path]:
        """element_id -> ordered.json path, cached and invalidated by mtime (D33).

        Before this, every element read walked *all* parsed documents and parsed
        each ``ordered.json`` in full (measured: 5 files, ~0.5 MB each) to find one
        element.  A re-parse rewrites the file's mtime, so the signature catches it.
        """

        try:
            signature = tuple(
                sorted(
                    (str(path), path.stat().st_mtime_ns)
                    for path in self.parsed_dir.glob("*/ordered.json")
                )
            )
        except OSError:  # a half-written parsed dir must not break reads
            signature = ()
        cached_signature, cached_index = self._element_index_cache
        if signature and cached_signature == signature:
            return cached_index
        index: dict[str, Path] = {}
        for path in sorted(self.parsed_dir.glob("*/ordered.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for element in payload.get("sequence") or payload.get("elements") or []:
                element_id = str(element.get("element_id") or "")
                if element_id:
                    index.setdefault(element_id, path)
        self._element_index_cache = (signature, index)
        return index

    def _read_element(self, element_id: str) -> ReadResult:
        index = self._element_index()
        hit_path = index.get(str(element_id))
        paths = [hit_path] if hit_path is not None else sorted(self.parsed_dir.glob("*/ordered.json"))
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # ``sequence`` is the key the ordered-document writer emits; ``elements``
            # never existed, so this branch answered CHUNK_NOT_FOUND for every
            # element id it was ever asked for (found while fixing A3).
            for element in payload.get("sequence") or payload.get("elements") or []:
                if str(element.get("element_id") or "") != element_id:
                    continue
                element_document_id = str(payload.get("document_id") or path.parent.name)
                element_page = int(element.get("page") or 1) or 1
                element_page_end = int(element.get("page_end") or 0) or 0
                return ReadResult(
                    document_id=element_document_id,
                    chunk_id=None,
                    page=element_page,
                    page_end=element_page_end if element_page_end > element_page else None,
                    section=str(element.get("section") or ""),
                    text=self._element_text(element),
                    chunk_type=str(element.get("type") or ""),
                    filename=self._resolve_filename(element_document_id, ""),
                )
        raise AdapterError(CHUNK_NOT_FOUND, element_id)


    def _element_text(self, element: dict) -> str:
        """D35: the element ceiling comes from settings, not a module constant."""

        text = str(element.get("text") or "")
        return text[: self.element_chars]


def _name_variants(value: str) -> set[str]:
    """Basename, lower-cased, plus the extension-stripped form -- the keys a
    filename may be addressed by (``attention.pdf`` / ``attention`` / the real
    stored name)."""

    base = str(value).replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    variants = {base} if base else set()
    if "." in base:
        variants.add(base.rsplit(".", 1)[0])
    return variants


def _to_hit(chunk, excerpt_chars: int) -> SearchHit:
    """``score`` stays the dense cosine (the evidence gate's unit)."""

    presentation = chunk.presentation_content or chunk.content
    page = int(chunk.page) or 1
    page_end = int(getattr(chunk, "page_end", 0) or 0) or None
    return SearchHit(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename or chunk.document_id,
        page=page,
        page_end=page_end if page_end and page_end > page else None,
        section=chunk.section,
        heading_path=list(chunk.heading_path),
        score=float(chunk.score),
        chunk_type=chunk.chunk_type,
        content_kind=chunk.content_kind,
        table_id=chunk.table_id,
        figure_id=chunk.figure_id,
        atomic=bool(chunk.atomic),
        degraded_structure=bool(chunk.degraded_structure),
        presentation_content=presentation,
        excerpt=presentation[:excerpt_chars],
        expanded_neighbors=dict(chunk.expanded_neighbors) or None,
    )
