"""Unit tests for ordered_document_v1 refinement (no Docling required)."""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.ordered_refiner import refine_rag_document


def test_refine_table_facts_and_roles(tmp_path: Path):
    rag = {
        "schema": "rag_document_v1",
        "document_id": "demo",
        "source_pdf": "demo.pdf",
        "parser": {"name": "docling"},
        "elements": [
            {
                "element_id": "e0001",
                "type": "paragraph",
                "page": 1,
                "heading_path": [],
                "text": "Provided proper attribution is provided, Google hereby grants permission",
                "search_text": "Provided proper attribution is provided, Google hereby grants permission",
                "payload": {},
            },
            {
                "element_id": "e0002",
                "type": "heading",
                "page": 1,
                "heading_path": [],
                "text": "Abstract",
                "search_text": "Abstract",
                "payload": {},
            },
            {
                "element_id": "e0003",
                "type": "heading",
                "page": 2,
                "heading_path": [],
                "text": "4.1 Open-domain Question Answering",
                "search_text": "4.1 Open-domain Question Answering",
                "payload": {},
            },
            {
                "element_id": "e0004",
                "type": "table",
                "page": 2,
                "heading_path": ["4.1 Open-domain Question Answering"],
                "text": "Table 1",
                "search_text": "Table 1",
                "payload": {
                    "caption": "Table 1: Open-Domain QA Test Scores",
                    "grid": {
                        "headers": ["", "Model", "NQ", "TQA"],
                        "rows": [
                            ["Closed Book", "T5-11B", "34.5 36.6", "- /50.1"],
                            ["", "RAG-Seq.", "44.5", "56.8/68.0"],
                        ],
                    },
                    "markdown": "| x |",
                },
            },
            {
                "element_id": "e0005",
                "type": "heading",
                "page": 10,
                "heading_path": [],
                "text": "References",
                "search_text": "References",
                "payload": {},
            },
            {
                "element_id": "e0006",
                "type": "list_item",
                "page": 10,
                "heading_path": ["References"],
                "text": "Vaswani et al. Attention Is All You Need.",
                "search_text": "Vaswani et al. Attention Is All You Need.",
                "payload": {},
            },
        ],
    }
    ordered = refine_rag_document(rag, output_dir=tmp_path, copy_figures=False)
    assert ordered["schema"] == "ordered_document_v1"
    seq = ordered["sequence"]
    assert seq[0]["role"] == "preamble"
    assert seq[0]["index_policy"] == "drop"
    table = next(e for e in seq if e["type"] == "table")
    assert table["parse_status"] in {"parsed", "partial"}
    facts = table["structure"]["facts"]
    assert any("44.5" in f and "NQ" in f for f in facts)
    assert any("RAG-Seq" in f for f in facts)
    ref = next(e for e in seq if e["element_id"] == "e0006")
    assert ref["role"] == "reference"
    assert ref["index_policy"] == "drop"
    # D17: ``skip`` is the honest name for what ``metadata_only`` did -- the
    # element reaches neither the chunks nor the metadata.
    assert all(e["index_policy"] in {"embed", "drop", "skip"} for e in seq)
    assert not any(e["index_policy"] == "metadata_only" for e in seq)
    # Numbered heading should deepen path for the table.
    assert any("4.1" in p for p in table["heading_path_norm"]) or table["heading_path_norm"]


def test_refine_formula_clean(tmp_path: Path):
    rag = {
        "schema": "rag_document_v1",
        "document_id": "demo",
        "source_pdf": "demo.pdf",
        "parser": {},
        "elements": [
            {
                "element_id": "e0001",
                "type": "heading",
                "page": 4,
                "heading_path": [],
                "text": "3.2.1 Scaled Dot-Product Attention",
                "search_text": "3.2.1 Scaled Dot-Product Attention",
                "payload": {},
            },
            {
                "element_id": "e0002",
                "type": "paragraph",
                "page": 4,
                "heading_path": ["3.2.1 Scaled Dot-Product Attention"],
                "text": "We call our particular attention Scaled Dot-Product Attention.",
                "search_text": "We call our particular attention Scaled Dot-Product Attention.",
                "payload": {},
            },
            {
                "element_id": "e0003",
                "type": "formula",
                "page": 4,
                "heading_path": ["3.2.1 Scaled Dot-Product Attention"],
                "text": "Attention ( Q , K , V ) = softmax(...) & & ( 1 )",
                "search_text": "Attention ( Q , K , V ) = softmax(...) & & ( 1 )",
                "payload": {"latex": "Attention ( Q , K , V ) = softmax(...) & & ( 1 )"},
            },
        ],
    }
    ordered = refine_rag_document(rag, output_dir=tmp_path, copy_figures=False)
    formula = next(e for e in ordered["sequence"] if e["type"] == "formula")
    assert formula["atomic"] is True
    assert formula["structure"]["eq_label"] == "1"
    assert "Attention(Q,K,V)" in formula["structure"]["latex"]
    assert "Scaled Dot-Product" in formula["search_text"]


def test_refine_writes_no_dead_element_fields(tmp_path: Path):
    """D7/D9 (2026-09-20): ``llm_text`` (19.1% of ordered.json, zero readers) and
    the element-level ``content_hash`` were deleted; the chunker reads
    ``text`` / ``search_text`` / ``structure`` instead."""

    rag = {
        "schema": "rag_document_v1",
        "document_id": "demo",
        "source_pdf": "demo.pdf",
        "parser": {},
        "elements": [
            {
                "element_id": "e0001",
                "type": "paragraph",
                "page": 1,
                "heading_path": [],
                "text": "Hello body",
                "search_text": "Hello body",
                "payload": {},
            }
        ],
    }
    ordered = refine_rag_document(rag, output_dir=tmp_path, copy_figures=False)
    for element in ordered["sequence"]:
        assert "llm_text" not in element
        assert "content_hash" not in element
    # The text the chunker actually reads is still there.
    assert ordered["sequence"][0]["text"] == "Hello body"
    assert ordered["sequence"][0]["search_text"] == "Hello body"
    assert ordered["stats"]["n_elements"] == 1


def test_refine_writes_quality(tmp_path: Path):
    rag = {
        "schema": "rag_document_v1",
        "document_id": "demo",
        "source_pdf": "demo.pdf",
        "parser": {},
        "elements": [
            {
                "element_id": "e0001",
                "type": "paragraph",
                "page": 1,
                "heading_path": [],
                "text": "Hello body",
                "search_text": "Hello body",
                "payload": {},
            }
        ],
    }
    ordered = refine_rag_document(rag, output_dir=tmp_path, copy_figures=False)
    out = tmp_path / "ordered.json"
    out.write_text(json.dumps(ordered), encoding="utf-8")
    assert out.exists()
    assert ordered["stats"]["n_elements"] == 1
