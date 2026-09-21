"""Ingestion package: Docling ordered parse + stage-2 chunking."""

from __future__ import annotations

__all__ = ["DocumentParser", "refine_rag_document", "chunk_ordered_document"]


def __getattr__(name: str):
    if name == "DocumentParser":
        from src.ingestion.document_parser import DocumentParser

        return DocumentParser
    if name == "refine_rag_document":
        from src.ingestion.ordered_refiner import refine_rag_document

        return refine_rag_document
    if name == "chunk_ordered_document":
        from src.ingestion.ordered_chunker import chunk_ordered_document

        return chunk_ordered_document
    raise AttributeError(name)
