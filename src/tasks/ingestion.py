"""Celery tasks for document ingestion (parse -> chunk -> embed -> publish)."""

from src.models.schemas import IngestionStage, IngestionStatus
from src.retrieval.index_builder import IndexBuildBusyError, IndexBuilder
from src.storage.document_registry import DocumentRegistry

from .celery_app import celery_app


@celery_app.task(
    bind=True,
    name="document_qa.parse_document",
)
def parse_document_task(self, document_id: str) -> dict[str, object]:
    """Parse one PDF, then rebuild the versioned vector index from all ordered docs."""

    registry = DocumentRegistry()
    document = registry.get_document(document_id)
    task_id = self.request.id

    registry.update_document(
        document_id,
        status=IngestionStatus.PROCESSING,
        stage=IngestionStage.PARSING,
        task_id=task_id,
        clear_failure_code=True,
    )

    pdf_path = registry.uploads_dir / document.stored_filename

    try:
        if not pdf_path.is_file():
            raise FileNotFoundError("stored PDF is missing")

        def report_stage(stage: IngestionStage) -> None:
            # COMPLETED must not force status back to processing; IndexBuilder
            # already marks docs completed before this final callback.
            status = (
                IngestionStatus.COMPLETED
                if stage == IngestionStage.COMPLETED
                else IngestionStatus.PROCESSING
            )
            registry.update_document(
                document_id,
                status=status,
                stage=stage,
                task_id=task_id,
                clear_failure_code=True,
            )

        manifest = IndexBuilder(registry).build(
            target_document_id=document_id,
            progress_callback=report_stage,
        )
        completed = registry.get_document(document_id)
        return {
            "document_id": completed.document_id,
            "task_id": task_id,
            "stage": completed.stage.value,
            "page_count": completed.page_count,
            "chunk_count": completed.chunk_count,
            "ordered_path": manifest.get("ordered_path"),
            "version_id": manifest.get("version_id"),
            "chunker_version": manifest.get("chunker_version"),
            "parse_only": False,
        }
    except Exception as error:
        registry.update_document(
            document_id,
            status=IngestionStatus.FAILED,
            stage=IngestionStage.FAILED,
            task_id=task_id,
            failure_code=_failure_code(error),
        )
        raise


@celery_app.task(bind=True, name="document_qa.rebuild_index")
def rebuild_index_task(self) -> dict[str, object]:
    """Re-parse every registered document and republish the vector index."""

    registry = DocumentRegistry()
    manifest = IndexBuilder(registry).build()
    return {
        "task_id": self.request.id,
        "stage": IngestionStage.COMPLETED.value,
        "chunk_count": manifest.get("chunk_count"),
        "document_count": manifest.get("document_count"),
        "version_id": manifest.get("version_id"),
        "chunker_version": manifest.get("chunker_version"),
        "parse_only": False,
    }


def _failure_code(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return "stored_file_missing"
    if isinstance(error, IndexBuildBusyError):
        return "index_build_busy"
    if isinstance(error, ValueError):
        return "pdf_parse_failed"
    message = str(error).lower()
    if "embedding" in message:
        return "embedding_failed"
    if "chroma" in message:
        return "index_publish_failed"
    if "docling" in message:
        return "pdf_parse_failed"
    return "pdf_parse_failed"
