import logging
import os
import time
from collections import deque
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.retrieval.chroma_store import ChromaVectorStore
from src.models.schemas import (
    DocumentListResponse,
    DocumentUploadResponse, DocumentChangeResponse,
    TaskStatusResponse,
)
from src.models.agent_schemas import DocumentMetaRequest
from src.storage.document_registry import (
    DocumentRegistry,
    DocumentRegistryError,
    DocumentNotFoundError,
    InvalidUploadError,
    RegistryDataError,
    RegistryLockedError, DocumentBusyError,
)
from src.tasks.ingestion import parse_document_task, rebuild_index_task
from src.tasks.celery_app import celery_app
from celery.result import AsyncResult
from src.api.agent_routes import router as agent_router
from src.api.document_quality import document_quality

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Document AI Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# ReadMate agent routes: the product main path (/agent/chat).
app.include_router(agent_router)


def get_document_registry() -> DocumentRegistry:
    """Create a lightweight registry service for the current request."""

    return DocumentRegistry()


# C7: Celery reports an id that was never queued as PENDING too -- its result
# backend simply has no key for it -- so ``/tasks/{id}`` used to answer
# "pending" for tasks that do not exist.  Remember what *this* process queued so
# an unknown id can be an honest 404.  The ledger is deliberately in-process: a
# restart falls back to the old, permissive answer instead of 404-ing a task
# that really is queued (the failure direction is "too helpful", not "wrong").
_QUEUED_TASKS: deque[str] = deque(maxlen=512)


def _remember_task(task_id: str) -> str:
    """Record a freshly queued task id and pass it through."""

    _QUEUED_TASKS.append(task_id)
    return task_id


def _is_known_task(task_id: str) -> bool:
    """True when this process queued the id or a registry record claims it."""

    if task_id in _QUEUED_TASKS:
        return True
    try:
        documents = get_document_registry().list_documents()
    except DocumentRegistryError:
        # Registry unreadable: do not invent a 404 for a task we cannot rule out.
        return True
    return any(getattr(document, "task_id", None) == task_id for document in documents)


@app.post(
    "/documents",
    response_model=DocumentUploadResponse,
)
def register_document(
    file: UploadFile = File(...),
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """Validate and persist a PDF registration without starting ingestion."""

    try:
        document = registry.register_upload(
            filename=file.filename,
            content_type=file.content_type,
            stream=file.file,
        )

        # D39: this used to be ``should_enqueue = True`` immediately followed by
        # ``if should_enqueue:`` -- a hard-coded switch left over from an early
        # "parse asynchronously?" toggle.  It could never be False, so it is gone
        # and the task is queued directly (behaviour unchanged).
        try:
            async_result = parse_document_task.delay(document.document_id)
            document = registry.update_document(
                document.document_id,
                task_id=_remember_task(async_result.id),
            )
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="Document was registered but the ingestion task could not be queued.",
            ) from error

        return {"document": document}
    except InvalidUploadError as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid PDF upload: {error}",
        ) from error
    except RegistryLockedError as error:
        raise HTTPException(
            status_code=503,
            detail="Document registry is busy. Please retry shortly.",
        ) from error
    except RegistryDataError as error:
        raise HTTPException(
            status_code=500,
            detail="Document registry is unavailable.",
        ) from error
    except DocumentRegistryError as error:
        raise HTTPException(
            status_code=500,
            detail="Document registration failed.",
        ) from error
    finally:
        file.file.close()


@app.put("/documents/{document_id}", response_model=DocumentChangeResponse)
def replace_registered_document(
    document_id: str,
    file: UploadFile = File(...),
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """Replace a PDF while keeping its document identity."""
    try:
        document = registry.replace_upload(document_id, file.filename, file.content_type, file.file)
        task = parse_document_task.delay(document.document_id)
        document = registry.update_document(document.document_id, task_id=task.id)
        return {"document": document}
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail="Document not found.") from error
    except DocumentBusyError as error:
        raise HTTPException(status_code=409, detail="Document ingestion is already running.") from error
    except InvalidUploadError as error:
        raise HTTPException(status_code=400, detail=f"Invalid PDF upload: {error}") from error
    except RegistryLockedError as error:
        raise HTTPException(status_code=503, detail="Document registry is busy. Please retry shortly.") from error
    except RegistryDataError as error:
        raise HTTPException(status_code=500, detail="Document registry is unavailable.") from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Document update could not be queued.") from error
    finally:
        file.file.close()


@app.patch("/documents/{document_id}", response_model=DocumentChangeResponse)
def patch_registered_document(
    document_id: str,
    request: DocumentMetaRequest,
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """FE-2: rename (alias) and/or switch a document off.

    Deliberately separate from ``PUT /documents/{id}``, which *replaces the PDF* and
    re-runs ingestion: renaming a document must never cost a re-index.
    """

    try:
        document = registry.update_document_meta(
            document_id,
            alias=request.alias,
            enabled=request.enabled,
        )
        return {"document": document}
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail="Document not found.") from error
    except RegistryLockedError as error:
        raise HTTPException(status_code=503, detail="Document registry is busy. Please retry shortly.") from error
    except RegistryDataError as error:
        raise HTTPException(status_code=500, detail="Document registry is unavailable.") from error


@app.get("/documents/{document_id}")
def get_registered_document(
    document_id: str,
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """D8 + D11: one document's registry record **plus** its parse-quality summary.

    ``GET /documents`` deliberately stays a plain ``list[DocumentRecord]`` -- the
    library page fetches that one on every poll and enriching every row would pay
    the artifact read on every tick.  The quality fields are served here (and, for
    the collection view, on the ``/agent/collections/{id}/documents`` entries).
    A missing or corrupted parse artifact answers ``quality: null``; it never 500s.
    """

    try:
        document = registry.get_document(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail="Document not found.") from error
    except RegistryDataError as error:
        raise HTTPException(status_code=500, detail="Document registry is unavailable.") from error
    payload = document.model_dump()
    payload.update(document_quality(document.document_id, registry.data_dir / "parsed"))
    return {"document": payload}


@app.get(
    "/documents",
    response_model=DocumentListResponse,
)
def list_registered_documents(
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """List locally registered PDFs and their current ingestion states."""

    try:
        rows = []
        parsed_dir = registry.data_dir / "parsed"
        for document in registry.list_documents():
            row = dict(document)
            # D8/D11: the UI badge lives on these cards, so the fields have to be
            # here (cached -- see _quality_probe).
            row.update(_quality_probe(str(row.get("document_id") or ""), parsed_dir))
            rows.append(row)
        return {"documents": rows}
    except RegistryDataError as error:
        raise HTTPException(
            status_code=500,
            detail="Document registry is unavailable.",
        ) from error


@app.delete("/documents/{document_id}")
def delete_registered_document(
    document_id: str,
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """Delete a registered PDF and enqueue a complete index rebuild."""
    try:
        deleted = registry.delete_document(document_id)
        try:
            task = rebuild_index_task.delay()
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="Document deleted, but index rebuild could not be queued.",
            ) from error
        return {"document_id": deleted.document_id, "rebuild_task_id": task.id}
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail="Document not found.") from error
    except RegistryDataError as error:
        raise HTTPException(status_code=500, detail="Document registry is unavailable.") from error


@app.post("/indexes/rebuild")
def rebuild_index():
    """Queue a full index rebuild from the current local registry."""
    try:
        task = rebuild_index_task.delay()
    except Exception as error:
        raise HTTPException(status_code=503, detail="Index rebuild could not be queued.") from error
    return {"task_id": _remember_task(task.id), "status": "queued"}


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """Return a safe, serializable view of one Celery task."""
    result = AsyncResult(task_id, app=celery_app)
    state = str(result.state).lower()
    if state == "pending" and not _is_known_task(task_id):
        raise HTTPException(status_code=404, detail={"code": "task_not_found", "message": task_id})
    payload = result.result if state == "success" and isinstance(result.result, dict) else None
    error_code = None
    if state == "failure":
        error_code = _task_error_code(result.result)
    return TaskStatusResponse(task_id=task_id, status=state, result=payload, error_code=error_code)


@app.post("/documents/{document_id}/retry", response_model=DocumentUploadResponse)
def retry_document(
    document_id: str,
    registry: DocumentRegistry = Depends(get_document_registry),
):
    """Requeue only a failed document while preserving its identity."""
    try:
        document = registry.get_document(document_id)
        if document.status.value != "failed":
            raise HTTPException(status_code=409, detail="Only failed documents can be retried.")
        document = registry.update_document(
            document_id,
            status="queued",
            stage="queued",
            task_id=None,
            clear_failure_code=True,
        )
        task = parse_document_task.delay(document_id)
        return {"document": registry.update_document(document_id, task_id=task.id)}
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail="Document not found.") from error
    except RegistryLockedError as error:
        raise HTTPException(status_code=503, detail="Document registry is busy. Please retry shortly.") from error


def _task_error_code(value: object) -> str:
    """Map Celery's failure payload to a stable public error code."""
    if isinstance(value, dict) and isinstance(value.get("failure_code"), str):
        return value["failure_code"]
    text = str(value).lower()
    if "busy" in text:
        return "index_build_busy"
    if "embedding" in text or "vector" in text:
        return "embedding_failed"
    return "task_failed"


# ---------------------------------------------------------------------------
# D43: /health index probe cache.
#
# The frontend polls /health every 15s (App.tsx) and every call used to run
# ``ChromaVectorStore.load()`` -- constructing a Chroma client and opening the
# published snapshot, for a page that only needs "is the index up?".
#
# The probe result is therefore memoised for a short TTL.  Deliberately *not* an
# event-driven invalidation: publishing happens in the Celery worker, a different
# process from the API, so an in-process signal would never arrive and the API
# would keep reporting a stale index forever.  A 30s TTL self-heals on its own
# and costs at most two snapshot opens per minute instead of four.
# ---------------------------------------------------------------------------
_HEALTH_CACHE_TTL_S = 30.0
# (monotonic_expiry, index_loaded, chunk_count, index_error) -- None until first probe.
_health_probe_cache: tuple[float, bool, int, str | None] | None = None


_quality_cache: dict[str, tuple[tuple, dict]] = {}


def _quality_probe(document_id: str, parsed_dir) -> dict:
    """D8/D11: parse-quality summary for one document, cached by artifact mtime.

    The detail route reads the artifacts once per request; the *list* route is
    polled every 3 seconds while an ingestion runs, so re-reading a 0.5 MB
    ``ordered.json`` per row per tick was not acceptable -- that is why the list
    originally had no quality fields, which left the UI badge with nothing to
    render (found while verifying the W3 work).  Cache key = the mtimes of the
    artifacts the summary is built from, so a re-parse invalidates it.
    """

    root = Path(parsed_dir) / document_id
    try:
        signature = tuple(
            (name, (root / name).stat().st_mtime_ns)
            for name in ("quality_report.json", "ordered.json", "chunk_report.json")
            if (root / name).is_file()
        )
    except OSError:
        signature = ()
    cached = _quality_cache.get(document_id)
    if cached is not None and cached[0] == signature:
        return cached[1]
    payload = document_quality(document_id, parsed_dir)
    _quality_cache[document_id] = (signature, payload)
    return payload


def _index_probe(current_dir) -> tuple[bool, int, str | None]:
    """Cached "is the published snapshot loadable, and how big is it" probe."""

    global _health_probe_cache
    now = time.monotonic()
    if _health_probe_cache is not None and now < _health_probe_cache[0]:
        return _health_probe_cache[1], _health_probe_cache[2], _health_probe_cache[3]

    index_loaded = False
    chunk_count = 0
    index_error = None
    # The directory check stays *outside* the cached branch on purpose: it is a
    # single stat, and answering "index_error: snapshot_invalid" for an index that
    # was never built would be a lie the frontend would badge.
    if current_dir.is_dir():
        try:
            _, manifest = ChromaVectorStore.load(current_dir)
            index_loaded = True
            chunk_count = int(manifest.get("chunk_count", 0))
        except Exception as error:
            # Previously this only set ``index_error = "snapshot_invalid"`` and
            # threw the real cause away, so a broken index was undiagnosable from
            # the logs (D43).  The public code stays the same; the traceback now
            # goes to the log at warning level.
            index_error = "snapshot_invalid"
            logger.warning("health index probe failed for %s", current_dir, exc_info=True)

    _health_probe_cache = (now + _HEALTH_CACHE_TTL_S, index_loaded, chunk_count, index_error)
    return index_loaded, chunk_count, index_error


@app.get("/health")
def health(registry: DocumentRegistry = Depends(get_document_registry)) -> dict[str, object]:
    """Report local registry and published-index availability without model calls.

    The returned shape is frozen -- the frontend ``HealthDot`` reads every key.
    """

    documents = registry.list_documents()
    current_dir = registry.data_dir / "chroma"
    index_loaded, chunk_count, index_error = _index_probe(current_dir)
    return {
        "status": "ok" if index_loaded or not documents else "degraded",
        "index_loaded": index_loaded,
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "index_error": index_error,
        "llm_configured": bool(
            (os.getenv("API_KEY") or os.getenv("api_key"))
            and (os.getenv("BASE_URL") or os.getenv("base_url"))
            and (os.getenv("MODEL") or os.getenv("model"))
        ),
    }


# ---------------------------------------------------------------------------
# SPA hosting (frontend plan B0-5): serve the built ReadMate UI from
# frontend/dist when it exists.  Registered last so every API route above
# (and /docs, /openapi.json) keeps priority; unknown non-API paths fall back
# to index.html for client-side routing.
# ---------------------------------------------------------------------------
_DIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend",
    "dist",
)

if os.path.isdir(_DIST_DIR):
    _assets_dir = os.path.join(_DIST_DIR, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="spa-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        candidate = os.path.normpath(os.path.join(_DIST_DIR, full_path))
        if full_path and candidate.startswith(os.path.normpath(_DIST_DIR)) and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_DIST_DIR, "index.html"))


