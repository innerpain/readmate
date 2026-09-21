"""ReadMate agent routes: the product main path (``/agent/chat``)."""

from __future__ import annotations

import json
import logging
import queue
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

from src.adapter.contracts import AdapterError
from src.adapter.rag_adapter import RagAdapterImpl
from src.agent.collections import CollectionService
from src.adapter.route_planner import RetrievalRouter
from src.api.document_quality import document_quality
from src.llm import LLMClient, ModelRegistry
from src.agent.memory.service import MemoryService
from src.agent.runtime import AgentRuntime
from src.agent.tools import ToolRunner
from src.config.settings import get_agent_settings, get_retrieval_settings
from src.models.agent_schemas import (
    AgentChatRequest,
    AgentChatResponse,
    CollectionCreateRequest,
    CollectionDocumentsRequest,
    CollectionRenameRequest,
    MemoryConfirmRequest,
    MemoryEntryRequest,
    MemoryRejectRequest,
    PreferenceRequest,
    ProfileRequest,
    ProgressRequest,
    ReadRequest,
    SessionCreateRequest,
    SessionTitleRequest,
)
from src.rag.query_planner import QueryPlanner
from src.retrieval.model_cache import retrieval_model_kwargs
from src.retrieval.persistent_retriever import PersistentRetriever
from src.storage.agent_db import AgentDB, session_scope
from src.storage.document_registry import DocumentNotFoundError, DocumentRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@lru_cache(maxsize=1)
def _cached_agent_settings():
    """D42: process-wide settings (frozen dataclass, read-only) -- safe to share."""

    return get_agent_settings()


def get_agent_db() -> AgentDB:
    """D42: one ``AgentDB`` per path, reused across requests.

    ``AgentDB`` holds no per-request state at all -- every method opens (and
    closes) its own short-lived SQLite connection via ``_connect``, and the only
    instance attribute is the path.  Migrations therefore run once per process
    instead of once per request.
    """

    return _agent_db(_cached_agent_settings().db_path or "")


@lru_cache(maxsize=4)
def _agent_db(db_path: str) -> AgentDB:
    """The cached instance behind :func:`get_agent_db` (keyed by resolved path)."""

    return AgentDB(db_path or None)


@lru_cache(maxsize=1)
def _agent_models():
    """D42: the parsed model registry -- a pure function of the config file."""

    return ModelRegistry.load(_cached_agent_settings().models_config_path or None)


def build_runtime() -> tuple[AgentRuntime, AgentDB, MemoryService, RagAdapterImpl]:
    settings = _cached_agent_settings()
    db = get_agent_db()
    registry = DocumentRegistry()
    collections = CollectionService(db=db, registry=registry)
    models = _agent_models()
    llm = LLMClient(models.resolve("agent"))          # ReAct 主模型（必须 supports_tools）
    router_llm = LLMClient(models.resolve("router"))  # 选轨/规划，可用便宜模型
    # C9: the retriever used to be rebuilt per request, which reloaded the
    # embedding model and the reranker (measured: 58-80s on the first search of
    # every turn).  Only those two objects are shared; the database, the LLM
    # clients and the tool runner stay per-request so no request-scoped state is
    # ever shared.
    #
    # D42 (user decision = b): only the *stateless* pieces became process-wide --
    # the settings object, the AgentDB handle and the resolved model registry
    # above.  ``ToolRunner``, ``RagAdapterImpl`` and ``PersistentRetriever`` must
    # keep being constructed per request: they carry per-request diagnostic state
    # (``adapter.last_low_confidence``, ``retriever.last_diagnostics``,
    # ``adapter.last_route``).  Sharing them would make two concurrent turns
    # overwrite each other's diagnostics, so the UI would attribute one request's
    # rerank/low-confidence verdict to another request's answer.
    adapter = RagAdapterImpl(
        retriever=PersistentRetriever(**retrieval_model_kwargs()),
        resolve_documents=collections.document_ids,
        registry=registry,
        router=RetrievalRouter(
            llm=router_llm,
            planner=QueryPlanner(llm=router_llm),
            mode=settings.route_mode,
        ),
        excerpt_chars=get_retrieval_settings().excerpt_chars,
    )
    memory = MemoryService(db=db, collections=collections, settings=settings)
    runner = ToolRunner(adapter=adapter, collections=collections, memory=memory, settings=settings)
    return AgentRuntime(llm=llm, tools=runner, db=db, settings=settings), db, memory, adapter


def _effective_collection_ids(db: AgentDB, request: AgentChatRequest) -> list[str]:
    """Resolve the turn's scope from ``collection_ids`` (preferred) or the legacy
    ``collection_id`` (R7 / D-5b), rejecting any id that is not an existing
    collection.

    C1 (D-7a): the chat route used to call ``upsert_collection(request.collection_id)``
    which treated the id as a *name* and minted a fresh 0-document "ghost"
    collection on every message.  We now only *read* collections; an unknown id
    is a stable 400 instead of a silent create.  Existing 0-document collections
    are left untouched (deleting data is a separate, user-approved batch).
    """

    raw = request.collection_ids or ([request.collection_id] if request.collection_id else [])
    ids = list(dict.fromkeys(cid for cid in raw if cid))
    if not ids:
        return []
    known = {collection["id"] for collection in db.list_collections()}
    missing = [cid for cid in ids if cid not in known]
    if missing:
        raise HTTPException(status_code=400, detail={"code": "collection_not_found", "message": ",".join(missing)})
    return ids


def _requested_collection_ids(request) -> list[str]:
    """What the client asked for, unvalidated.

    Once a session is locked (第 8 条) the request's ids are irrelevant to retrieval,
    so they are deliberately *not* validated -- a stale id must not 400 a turn.  They
    are still forwarded so the runtime can record the mismatch it ignored.
    """

    raw = request.collection_ids or ([request.collection_id] if request.collection_id else [])
    return list(dict.fromkeys(cid for cid in raw if cid))


def _session_scope_ids(db: AgentDB, session_id: str | None) -> list[str] | None:
    """The scope a session is already locked to, or ``None`` while undecided.

    问题.md 第二轮第 8 条: once a conversation has started, its material scope is
    read from the session rather than from the request, so a stray ``collection_ids``
    cannot silently re-point an ongoing dialogue.  The runtime enforces the same
    rule; resolving it here as well keeps the unknown-collection 400 from firing on
    ids that were never going to be used.
    """

    if not session_id:
        return None
    try:
        session = db.get_session(session_id)
    except KeyError:
        return None
    return session_scope(session)


def _maybe_enqueue_summary(runtime: AgentRuntime, session_id: str) -> None:
    """批 D 阶段 2 (D32): compress the window in the background once older turns fall
    out of it.

    Deliberately fire-and-forget: the turn has already been answered and stored, and
    the summary only helps the *next* turn -- paying for it inline would add latency
    for nothing.  Every failure path is swallowed: a dead broker or a model outage
    must not turn an answered question into an error.
    """

    if not get_agent_settings().session_summary_enabled:
        return
    try:
        if not runtime.summary_due(session_id):
            return
        from src.tasks.summary import summarize_session_task

        summarize_session_task.delay(session_id)
    except Exception as error:  # noqa: BLE001 - optimisation only
        logger.warning("summary enqueue failed for %s: %s", session_id, error)


@router.post("/chat", response_model=AgentChatResponse)
def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    runtime, db, _, _ = build_runtime()
    session_id = request.session_id
    locked_scope = _session_scope_ids(db, session_id)
    if locked_scope is not None:
        # Locked (第 8 条): the request's ids are forwarded unvalidated so the runtime
        # can record the mismatch it ignores; the session's scope is what retrieves.
        ids = _requested_collection_ids(request)
    else:
        ids = _effective_collection_ids(db, request)
    if not session_id:
        # Bind a *single* home collection to the session (FK-safe); a
        # multi-collection turn binds none but still locks the scope on the first
        # turn (第 8 条; the lock itself lives on the session row).
        bind = ids[0] if len(ids) == 1 else None
        session_id = db.create_session(collection_id=bind, mode=request.mode or "deep")
    try:
        result = runtime.run(session_id, request.message, request.mode, collection_ids=ids or None)
    except KeyError as error:
        # C6: the runtime loads the session by id and raises KeyError for one that
        # does not exist; unhandled it became a 500.  Same code the SSE route and
        # the memory routes already answer with.
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "message": session_id}) from error
    except AdapterError as error:
        raise HTTPException(status_code=409, detail={"code": error.code, "message": error.message}) from error
    except HTTPException:
        # 400/404 raised above must reach the client unchanged.
        raise
    except Exception as error:
        # C10: the upstream model fails for reasons that are nobody's bug here --
        # quota exhausted, network drop, provider 5xx.  Unhandled, those became a
        # bare 500 with no code for the UI to key on (measured: an exhausted quota
        # answered `HTTP Error 500` to a plain POST).  The SSE twin already
        # degrades to an `error` event; the sync route now answers 503 with a
        # structured code and a message a user can act on.
        raise HTTPException(
            status_code=503,
            detail={
                "code": "llm_unavailable",
                "message": "模型服务暂时不可用（额度耗尽或网络中断），请稍后重试",
                "reason": str(error)[:200],
            },
        ) from error
    _maybe_enqueue_summary(runtime, session_id)
    return AgentChatResponse(session_id=session_id, **result.__dict__)


def _sse(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@router.post("/chat/stream")
def agent_chat_stream(request: AgentChatRequest):
    """SSE twin of /agent/chat (frontend plan §5); the sync route stays the
    eval harness's path.  The turn runs in a worker thread; events hop over a
    queue so backpressure never blocks the ReAct loop."""

    runtime, db, _, _ = build_runtime()
    locked_scope = _session_scope_ids(db, request.session_id)
    if locked_scope is not None:
        # Locked (第 8 条): see ``agent_chat`` -- the request's ids no longer decide.
        ids = _requested_collection_ids(request)
    else:
        ids = _effective_collection_ids(db, request)
    session_id = request.session_id or db.create_session(
        collection_id=(ids[0] if len(ids) == 1 else None), mode=request.mode or "deep"
    )
    try:
        effective_mode = request.mode or db.get_session(session_id).get("mode") or "deep"
    except KeyError:
        effective_mode = request.mode or "deep"
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            result = runtime.run(
                session_id,
                request.message,
                request.mode,
                on_event=lambda name, payload: events.put((name, payload)),
                stream=True,
                collection_ids=ids or None,
            )
            response = AgentChatResponse(session_id=session_id, **result.__dict__)
            events.put(("turn_end", response.model_dump()))
            _maybe_enqueue_summary(runtime, session_id)
        except AdapterError as error:
            events.put(("error", {"code": error.code, "message": error.message}))
        except Exception as error:  # pragma: no cover - surfaced as an SSE error event
            events.put(("error", {"code": "runtime_error", "message": str(error)}))
        finally:
            events.put(None)

    def event_stream():
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        yield _sse("turn_start", {"session_id": session_id, "mode": effective_mode})
        while True:
            try:
                item = events.get(timeout=15)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if item is None:
                return
            name, payload = item
            yield _sse(name, payload)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sessions/{session_id}/compact")
def compact_session(session_id: str) -> dict:
    """批 D 阶段 2 (D32): compress this session's window now, synchronously.

    The manual twin of the background job.  A human who asks for it gets the result
    -- including the honest reason when there was nothing to fold in.  ``force``
    ignores the failure breaker: a human has context the counter does not.
    """

    db = get_agent_db()
    try:
        db.get_session(session_id)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail={"code": "session_not_found", "message": session_id}
        ) from error
    from src.tasks.summary import run_summary

    return run_summary(session_id, force=True)


@router.post("/collections")
def create_collection(request: CollectionCreateRequest) -> dict:
    runtime, db, _, _ = build_runtime()
    collection_id = db.upsert_collection(request.name)
    if request.document_ids:
        db.add_documents(collection_id, request.document_ids)
    return {"collection_id": collection_id, "document_count": len(db.collection_document_ids(collection_id))}


@router.get("/collections")
def list_collections() -> dict:
    """Read-only collection listing for the UI (B0-1; the DB method predates it)."""
    _, db, _, _ = build_runtime()
    return {"collections": db.list_collections()}


@router.get("/sessions")
def list_sessions(collection_id: str | None = None, limit: int = 20, offset: int = 0) -> dict:
    """Most-recent sessions for the UI sidebar (B0-4).

    Each row also reports the scope the session locked on its first turn
    (``scope_ids`` + ``scope_locked``), so reopening an old conversation shows the
    material it was actually started with instead of whatever is selected now.
    """
    _, db, _, _ = build_runtime()
    rows = db.list_sessions(collection_id, limit=limit, offset=offset)
    for row in rows:
        scope = session_scope(row)
        row["scope_ids"] = scope
        row["scope_locked"] = scope is not None
    return {"sessions": rows}


@router.post("/collections/{collection_id}/documents")
def add_documents(collection_id: str, request: CollectionDocumentsRequest) -> dict:
    _, db, _, _ = build_runtime()
    db.add_documents(collection_id, request.document_ids)
    return {"collection_id": collection_id, "document_count": len(db.collection_document_ids(collection_id))}


@router.get("/collections/{collection_id}/documents")
def list_collection_documents(collection_id: str) -> dict:
    """Documents bound to one collection (资料管理页 per-collection view).

    D8 + D11: each entry also carries ``degraded`` / ``user_notice`` / ``quality``
    read from that document's parse artifacts, so the library page can badge a
    degraded parse without a second request per row.  Artifacts are read from the
    same registry the rest of this module uses (one ``data/parsed/<id>/`` lookup
    per document) and a missing/corrupt file yields ``quality: null`` -- the
    listing never fails because of a half-written parse directory.
    """

    runtime, db, _, adapter = build_runtime()
    docs = adapter.list_docs(collection_id)
    parsed_root = DocumentRegistry().data_dir / "parsed"
    entries: list[dict] = []
    for doc in docs:
        entry = doc.model_dump()
        entry.update(document_quality(doc.document_id, parsed_root))
        entries.append(entry)
    return {"collection_id": collection_id, "documents": entries}


def _require_collection(db: AgentDB, collection_id: str) -> None:
    """404 for an unknown collection (same code the chat route answers with)."""

    if collection_id not in {collection["id"] for collection in db.list_collections()}:
        raise HTTPException(status_code=404, detail={"code": "collection_not_found", "message": collection_id})


@router.delete("/collections/{collection_id}/documents")
def remove_collection_documents(collection_id: str, request: CollectionDocumentsRequest) -> dict:
    """C2: unbind documents from a collection.

    ``AgentDB.remove_document`` (and the FK that made it necessary) predates this
    route by weeks -- only the HTTP surface was missing.  Unbinding never touches
    the documents themselves or the index.
    """

    _, db, _, _ = build_runtime()
    _require_collection(db, collection_id)
    for document_id in request.document_ids:
        db.remove_document(collection_id, document_id)
    return {"collection_id": collection_id, "document_count": len(db.collection_document_ids(collection_id))}


@router.delete("/collections/{collection_id}")
def delete_collection(collection_id: str) -> dict:
    """C3: delete a collection (the ghost-collection cleanup path).

    Guarded deliberately: it removes a *grouping*, never the documents, and
    sessions bound to it are detached rather than deleted, so deleting a
    collection cannot destroy conversation history.
    """

    _, db, _, _ = build_runtime()
    report = db.delete_collection(collection_id)
    if not report["deleted"]:
        raise HTTPException(status_code=404, detail={"code": "collection_not_found", "message": collection_id})
    return report


@router.post("/sessions")
def create_session(request: SessionCreateRequest) -> dict:
    _, db, _, _ = build_runtime()
    session_id = db.create_session(collection_id=request.collection_id, mode=request.mode or "deep")
    return {"session_id": session_id}


def _require_session(db: AgentDB, session_id: str) -> None:
    """404 for an unknown session (same code the chat route answers with)."""

    try:
        db.get_session(session_id)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail={"code": "session_not_found", "message": session_id}
        ) from error


@router.patch("/sessions/{session_id}")
def rename_session(session_id: str, request: SessionTitleRequest) -> dict:
    """C4: rename a session.

    ``create_session`` has accepted a title since B0-4 and ``list_sessions`` has
    returned it ever since; the only missing piece was the write route, so the
    sidebar showed "精读会话 3f2a1c" forever.
    """

    _, db, _, _ = build_runtime()
    _require_session(db, session_id)
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail={"code": "title_empty", "message": session_id})
    db.set_session_title(session_id, title)
    return {"session_id": session_id, "title": title}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    """C4: delete a session and its transcript.

    Deliberately narrow: it removes the conversation (messages + pending memory
    candidates cascade in the DB) and never touches documents, collections or the
    index -- deleting a chat cannot destroy evidence.
    """

    _, db, _, _ = build_runtime()
    report = db.delete_session(session_id)
    if not report["deleted"]:
        raise HTTPException(
            status_code=404, detail={"code": "session_not_found", "message": session_id}
        )
    return report


@router.get("/sessions/{session_id}/messages")
def session_messages(
    session_id: str,
    limit: int = Query(100, ge=1, le=1000),
    before_id: int | None = Query(None, ge=1),
) -> dict:
    """Session transcript for the UI sidebar.

    R7: parse the persisted ``payload_json`` into a structured ``payload`` field
    so an already-recorded assistant row exposes ``rounds`` / ``tool_trace_digest``
    / ``observed_chunk_ids`` etc. the same way the live HTTP response does.  A
    row with no / malformed payload still returns normally -- the caller can fall
    back to ``content`` (this is a thin wrapper; the DB schema is unchanged).

    D41: the response used to grow without bound (a long conversation returned
    every row, payloads included, on every session switch).  It now answers the
    **newest** ``limit`` rows in ascending id order -- exactly what the chat view
    renders -- plus ``has_more`` so a client can tell there is older history and
    page back with ``before_id`` (the id of the oldest row it already holds).
    ``has_more`` is a single-row existence probe (``LIMIT 1``), not a count, so
    the cost does not scale with the transcript.  ``limit=100`` is the default
    the frontend relies on; omitting both parameters keeps the pre-D41 behaviour
    of "everything, oldest first" only if the transcript is shorter than 100 rows.
    """

    _, db, _, _ = build_runtime()
    rows = db.list_messages(session_id, limit=limit, before_id=before_id)
    for row in rows:
        raw = row.get("payload_json")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            row["payload"] = json.loads(raw)
        except json.JSONDecodeError:
            continue
    has_more = False
    if rows:
        has_more = bool(db.list_messages(session_id, limit=1, before_id=int(rows[0]["id"])))
    return {"session_id": session_id, "messages": rows, "has_more": has_more}


@router.get("/memory")
def get_memory(collection_id: str | None = None) -> dict:
    _, db, memory, _ = build_runtime()
    return {
        "digest": memory.digest(collection_id),
        "profile": db.get_profile(),
        "preferences": db.get_preferences(),
        "progress": db.get_progress(collection_id) if collection_id else None,
    }


@router.get("/memory/candidates")
def list_memory_candidates(session_id: str) -> dict:
    """Pending long-term-memory candidates for one session (B0-2)."""
    _, db, _, _ = build_runtime()
    return {"candidates": db.list_candidates(session_id, status="pending")}


@router.post("/memory/confirm")
def confirm_memory(request: MemoryConfirmRequest) -> dict:
    """Promote pending candidates.  Resolves the session's collection so a
    ``kind=progress`` candidate actually writes into ``learning_progress``
    instead of being silently marked confirmed without landing (B1)."""

    _, db, memory, _ = build_runtime()
    try:
        session = db.get_session(request.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "message": request.session_id})
    collection_id = session.get("collection_id") if session else None
    details = memory.confirm_with_details(
        session_id=request.session_id,
        candidate_ids=request.candidate_ids,
        collection_id=collection_id,
    )
    return {"confirmed": int(details["marked"]), "landed": int(details["landed"])}


@router.post("/memory/reject")
def reject_memory(request: MemoryRejectRequest) -> dict:
    """Reject pending candidates (B0-3; closes the §9 gap: reject() had no route).

    C8: an unknown session id used to answer ``200 {"rejected": 0}`` while
    ``confirm`` answered 404 for the same input, so a typo'd id read as "nothing
    to reject".  Both routes now report the missing session.
    """

    _, db, memory, _ = build_runtime()
    try:
        db.get_session(request.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "message": request.session_id})
    return {"rejected": memory.reject(session_id=request.session_id, candidate_ids=request.candidate_ids)}


@router.post("/read")
def read_passage(request: ReadRequest) -> dict:
    _, _, _, adapter = build_runtime()
    try:
        result = adapter.read(
            chunk_id=request.chunk_id,
            document_id=request.document_id,
            page=request.page,
            element_id=request.element_id,
        )
    except AdapterError as error:
        raise HTTPException(status_code=404, detail={"code": error.code, "message": error.message}) from error
    return result.model_dump()


# ---------------------------------------------------------------------------
# R7-2 · read-only original / figure routes (frontend plan: 证据查看).
# Safety (plan §P10): id must be a known document; figure name is reduced to its
# basename and resolve()-checked to stay inside that document's figures dir; the
# extension is whitelisted; a directory is never listed and a bare name is 404.
# Failures return a stable JSON error code so the frontend can degrade (P5/P6) --
# the body shape is FastAPI's ``{"detail": {"code", "message"}}``.
# ---------------------------------------------------------------------------

_INLINE_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _known_document(registry: DocumentRegistry, document_id: str):
    try:
        return registry.get_document(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "document_not_found", "message": str(error)}) from error


@router.get("/documents/{document_id}/file")
def document_file(document_id: str):
    """Stream the stored PDF inline. ``document_id`` is the UUID from search /
    ``/agent/collections/{id}/documents``; unknown ids 404 (no listing, no
    filesystem probing)."""

    registry = DocumentRegistry()
    record = _known_document(registry, document_id)
    path = (registry.uploads_dir / record.stored_filename).resolve()
    if path.parent != registry.uploads_dir.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail={"code": "file_not_found", "message": document_id})
    return FileResponse(path, media_type="application/pdf", headers={"Content-Disposition": "inline"})


@router.get("/documents/{document_id}/figures")
def document_figures(document_id: str) -> dict:
    """List a document's extracted figure images with **ready-to-use URLs**
    (P3): ``base`` is the figures path prefix and each item carries a ``url`` the
    frontend can drop into ``<img src>`` without knowing the on-disk layout."""

    registry = DocumentRegistry()
    _known_document(registry, document_id)
    figures_dir = (registry.data_dir / "parsed" / document_id / "figures").resolve()
    base = f"/agent/documents/{document_id}/figures/"
    figures: list[dict[str, str]] = []
    if figures_dir.is_dir():
        for entry in sorted(figures_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in _INLINE_IMAGE_TYPES:
                figures.append({"name": entry.name, "url": base + entry.name})
    return {"document_id": document_id, "base": base, "figures": figures}


@router.get("/documents/{document_id}/figures/{name}")
def document_figure(document_id: str, name: str):
    """Serve one figure. The name is basename-only and must resolve inside the
    document's own figures dir (no ``..`` traversal, no symlink escape), carry a
    whitelisted image extension, and be a real file -- otherwise 404.  Inline so
    the browser renders it (P10)."""

    registry = DocumentRegistry()
    _known_document(registry, document_id)
    figures_dir = (registry.data_dir / "parsed" / document_id / "figures").resolve()
    safe_name = Path(name).name  # strip any directory components / traversal
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _INLINE_IMAGE_TYPES:
        raise HTTPException(status_code=404, detail={"code": "unsupported_type", "message": safe_name})
    target = (figures_dir / safe_name).resolve()
    try:
        target.relative_to(figures_dir)
    except ValueError as error:  # resolved outside the figures dir -> reject
        raise HTTPException(status_code=404, detail={"code": "invalid_path", "message": safe_name}) from error
    if not target.is_file():
        raise HTTPException(status_code=404, detail={"code": "figure_not_found", "message": safe_name})
    return FileResponse(target, media_type=_INLINE_IMAGE_TYPES[suffix], headers={"Content-Disposition": "inline"})


# ------------------------------------------------------------------ FE-2 additions
@router.patch("/collections/{collection_id}")
def rename_collection(collection_id: str, request: CollectionRenameRequest) -> dict:
    """FE-2: rename one collection (分区改名).

    ``name`` is UNIQUE in the schema, so a clash is reported as 409 rather than a
    500: the UI can say "该名称已存在" instead of "服务器错误".
    """

    _, db, _, _ = build_runtime()
    try:
        return db.rename_collection(collection_id, request.name.strip())
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail={"code": "collection_name_taken", "message": request.name},
        ) from error


@router.delete("/collections/{collection_id}/documents/{document_id}")
def remove_collection_document(collection_id: str, document_id: str) -> dict:
    """FE-2: move one document out of one collection (资料移动位置).

    The pre-existing ``DELETE /collections/{id}/documents`` takes a body and is
    batch-shaped; this one names a single membership so a chip can carry an ×.
    Removing a document from a collection never deletes the file or the index --
    only the membership.
    """

    _, db, _, _ = build_runtime()
    if collection_id not in {str(row["id"]) for row in db.list_collections()}:
        raise HTTPException(status_code=404, detail={"code": "collection_not_found", "message": collection_id})
    before = set(db.collection_document_ids(collection_id))
    db.remove_document(collection_id, document_id)
    after = set(db.collection_document_ids(collection_id))
    return {
        "removed": document_id in before and document_id not in after,
        "collection_id": collection_id,
        "document_id": document_id,
        "document_count": len(after),
    }


# ------------------------------------------------------------ FE-3 additions
@router.get("/memory/entries")
def list_memory_entries(status: str = "confirmed") -> dict:
    """FE-3: the editable view of long-term memory (default: confirmed rows)."""

    _, db, _, _ = build_runtime()
    return {"entries": db.list_memory_entries(status)}


@router.patch("/memory/entries/{entry_id}")
def update_memory_entry(entry_id: int, request: MemoryEntryRequest) -> dict:
    """FE-3: edit one memory entry in place (key and/or value)."""

    _, db, _, _ = build_runtime()
    try:
        return {"entry": db.update_memory_entry(entry_id, key=request.key, value=request.value)}
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail={"code": "memory_entry_not_found", "message": str(entry_id)},
        ) from error


@router.delete("/memory/entries/{entry_id}")
def delete_memory_entry(entry_id: int) -> dict:
    """FE-3: forget one memory entry."""

    _, db, _, _ = build_runtime()
    return {"deleted": db.delete_memory_entry(entry_id), "entry_id": entry_id}


@router.patch("/profile")
def update_profile(request: ProfileRequest) -> dict:
    """FE-3: edit the learner profile (identity / major / goal)."""

    _, db, _, _ = build_runtime()
    db.set_profile(identity=request.identity, major=request.major, goal=request.goal)
    return {"profile": db.get_profile()}


@router.put("/preferences/{key}")
def set_preference(key: str, request: PreferenceRequest) -> dict:
    """FE-3: set one preference value (upsert)."""

    _, db, _, _ = build_runtime()
    db.set_preference(key, request.value)
    return {"preferences": db.get_preferences()}


@router.delete("/preferences/{key}")
def delete_preference(key: str) -> dict:
    """FE-3: drop one preference."""

    _, db, _, _ = build_runtime()
    return {"deleted": db.delete_preference(key), "key": key}


@router.patch("/progress/{collection_id}")
def update_progress(collection_id: str, request: ProgressRequest) -> dict:
    """FE-3: edit the learning-progress record for one collection.

    ``learning_progress`` is per collection (that is how the memory service writes it),
    so the route is keyed by collection rather than by session.
    """

    _, db, _, _ = build_runtime()
    db.set_progress(
        collection_id,
        last_focus=request.last_focus,
        open_questions=request.open_questions,
        next_step=request.next_step,
    )
    return {"progress": db.get_progress(collection_id)}
