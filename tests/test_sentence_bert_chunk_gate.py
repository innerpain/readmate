"""Chunk the audited sentence-bert ordered.json fixture (no embed/publish)."""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.ordered_chunker import chunk_ordered_document, estimate_tokens, write_chunks_artifacts

FIXTURE = Path("data/parsed/sentence-bert-docker/ordered.json")


def test_sentence_bert_ordered_chunks_gate():
    if not FIXTURE.is_file():
        return  # fixture absent in some checkouts
    ordered = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chunks, report = chunk_ordered_document(ordered, filename="sentence-bert.pdf")
    assert report["n_chunks"] > 0
    by_type = report["n_chunks_by_type"]
    assert by_type.get("table_summary", 0) == 7
    assert by_type.get("table_pack", 0) >= 7
    assert by_type.get("formula", 0) == 2
    assert by_type.get("figure", 0) == 2
    assert report["max_retrieval_tokens"] <= 520
    for chunk in chunks:
        assert estimate_tokens(chunk["retrieval_text"]) <= 520
        if chunk["chunk_type"] in {"table_summary", "table_pack"}:
            assert chunk["neighbor_prev_chunk_ids"] or chunk["neighbor_next_chunk_ids"]
            assert "x" * 1000 not in chunk["retrieval_text"]


def test_write_chunks_artifacts(tmp_path: Path):
    ordered = {
        "schema": "ordered_document_v1",
        "document_id": "demo",
        "revision": "r",
        "sequence": [
            {
                "element_id": "e1",
                "ordinal": 0,
                "type": "paragraph",
                "role": "body",
                "index_policy": "embed",
                "page": 1,
                "heading_path_norm": ["A"],
                "text": "Hello world from the abstract body.",
                "search_text": "Hello world from the abstract body.",
                "structure": {},
                "atomic": False,
                "parse_status": "parsed",
            }
        ],
    }
    chunks, report = chunk_ordered_document(ordered, filename="demo.pdf")
    paths = write_chunks_artifacts(tmp_path, chunks, report)
    assert Path(paths["chunks_path"]).is_file()
    assert Path(paths["report_path"]).is_file()
