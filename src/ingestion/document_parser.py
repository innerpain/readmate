"""Parse-only document pipeline: PDF -> ordered_document_v1 artifact."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.ingestion.docling_converter import PARSER_OPTIONS, convert_pdf
from src.ingestion.ordered_refiner import refine_rag_document
from src.ingestion.rag_export import build_rag_document
from src.models.schemas import IngestionStage
from src.storage.document_registry import DocumentRegistry

logger = logging.getLogger(__name__)


def _parse_debug_enabled() -> bool:
    """D10: whether the parse stage should also dump ``rag_debug.json``.

    Opt-in on purpose: the dump is ~250 KB per document and nothing in the codebase
    reads it, so it is written only when someone is actually debugging a parse.
    """

    return os.getenv("PARSE_DEBUG", "").strip().lower() in {"1", "true", "yes"}


class DocumentParser:
    """Replace the legacy parse stack with Docling + ordered refinement."""

    def __init__(self, registry: DocumentRegistry) -> None:
        self.registry = registry
        self.parsed_dir = registry.data_dir / "parsed"

    def parse_document(self, document_id: str) -> dict[str, Any]:
        document = self.registry.get_document(document_id)
        pdf_path = self.registry.uploads_dir / document.stored_filename
        if not pdf_path.is_file():
            raise FileNotFoundError(document.stored_filename)

        out_dir = self.parsed_dir / document_id
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("docling convert start document_id=%s", document_id)
        docling_doc = convert_pdf(pdf_path, artifacts_dir=out_dir)
        rag = build_rag_document(
            docling_doc,
            document_id=document_id,
            source_pdf=pdf_path,
            source_docling_json=out_dir / "docling.json",
            parser_options=PARSER_OPTIONS,
        )
        # D10 (2026-09-20): ``rag_debug.json`` is a dead artifact -- measured at
        # 249,635 B per document with no reader anywhere in the codebase (only the
        # writer existed).  It is now opt-in: set PARSE_DEBUG=1 when actually
        # debugging a parse, otherwise a normal parse no longer pays ~250 KB of
        # serialisation and disk per document.
        if _parse_debug_enabled():
            (out_dir / "rag_debug.json").write_text(
                json.dumps(rag, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        ordered = refine_rag_document(rag, output_dir=out_dir, copy_figures=True)
        ordered["revision"] = document.revision
        ordered_path = out_dir / "ordered.json"
        ordered_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
        quality_path = out_dir / "quality_report.json"
        quality_path.write_text(
            json.dumps(
                {
                    "document_id": document_id,
                    "ordered_path": str(ordered_path),
                    "stats": ordered.get("stats"),
                    "quality": ordered.get("quality"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        seq_pages = {
            e.get("page")
            for e in ordered.get("sequence") or []
            if isinstance(e.get("page"), int)
        }
        page_count = max(seq_pages) if seq_pages else None

        self.registry.update_document(
            document_id,
            status=None,
            stage=IngestionStage.PARSED,
            page_count=page_count,
            chunk_count=0,
            clear_failure_code=True,
        )
        return {
            "document_id": document_id,
            "ordered_path": str(ordered_path),
            "quality_path": str(quality_path),
            "page_count": page_count,
            "element_count": len(ordered.get("sequence") or []),
            "stats": ordered.get("stats"),
            "stage": IngestionStage.PARSED.value,
        }

    def parse_all(self) -> dict[str, Any]:
        results = []
        for document in self.registry.list_documents():
            results.append(self.parse_document(document.document_id))
        return {
            "document_count": len(results),
            "results": results,
            "stage": IngestionStage.COMPLETED.value,
            "chunk_count": 0,
        }
