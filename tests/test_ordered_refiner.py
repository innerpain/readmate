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


def test_header_only_table_keeps_its_headers(tmp_path: Path):
    """D61 follow-up (measured): a one-row worksheet arrives from Docling as a
    header-only table.  Its column names *are* the content, so the refine pass
    keeps them for the summary chunk instead of reporting an unparseable table
    and losing the sheet."""

    rag = {
        "schema": "rag_document_v1",
        "document_id": "demo",
        "source_pdf": "book.xlsx",
        "parser": {},
        "elements": [
            {
                "element_id": "e0001",
                "type": "table",
                "page": 2,
                "heading_path": ["Notes-EU"],
                "text": "",
                "search_text": "Notes-EU",
                "payload": {
                    "caption": "",
                    "grid": {
                        "headers": ["memo", "carried"],
                        "n_rows": 0,
                        "n_cols": 2,
                        "rows": [],
                    },
                    "markdown": "| memo   | carried   |\n|--------|-----------|",
                },
            }
        ],
    }
    ordered = refine_rag_document(rag, output_dir=tmp_path, copy_figures=False)
    element = ordered["sequence"][0]
    assert element["parse_status"] == "partial"
    assert element["structure"]["headers"] == ["memo", "carried"]
    assert element["structure"]["rows"] == []
    assert ordered["quality"]["table_unparsed"] == 0


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


def _rag(elements: list[dict], *, file_type: str | None = ".pdf") -> dict:
    """Minimal rag_document_v1 for the heading-prefix tests."""

    rag: dict = {
        "schema": "rag_document_v1",
        "document_id": "demo",
        "source_pdf": "demo",
        "parser": {},
        "elements": elements,
    }
    if file_type is not None:
        rag["file_type"] = file_type
    return rag


def _heading(
    eid: str,
    text: str,
    *,
    level: int,
    source_label: str = "section_header",
    page: int = 1,
) -> dict:
    return {
        "element_id": eid,
        "type": "heading",
        "page": page,
        "level": level,
        "source_label": source_label,
        "heading_path": [],
        "text": text,
        "search_text": text,
        "payload": {},
    }


def _paragraph(eid: str, text: str, *, page: int = 1, level: int = 3) -> dict:
    return {
        "element_id": eid,
        "type": "paragraph",
        "page": page,
        "level": level,
        "source_label": "text",
        "heading_path": [],
        "text": text,
        "search_text": text,
        "payload": {},
    }


def _norms(ordered: dict) -> dict[str, list[str]]:
    return {e["element_id"]: e["heading_path_norm"] for e in ordered["sequence"]}


def test_unnumbered_top_headings_start_a_new_chain_outside_pdf(tmp_path: Path):
    """D62 (measured, ``tmp/fmt-probe/probe_heading_nesting.py``): two DOCX
    chapters used to come out as ``Chapter One > Chapter Two`` -- the second
    chapter is a sibling, and the ``§N`` section rule already said so."""

    rag = _rag(
        [
            _heading("e0001", "Chapter One", level=2),
            _paragraph("e0002", "Body of the first chapter, with a fact: 42 widgets."),
            _heading("e0003", "Chapter Two", level=2, page=2),
            _paragraph("e0004", "Second chapter body text.", page=2),
        ],
        file_type=".docx",
    )
    ordered = refine_rag_document(rag, output_dir=tmp_path, copy_figures=False)
    norms = _norms(ordered)
    assert norms["e0001"] == []
    assert norms["e0002"] == ["Chapter One"]
    assert norms["e0003"] == []                # was ["Chapter One"] before D62
    assert norms["e0004"] == ["Chapter Two"]   # was ["Chapter One", "Chapter Two"]


def test_pdf_keeps_the_conservative_unnumbered_nesting(tmp_path: Path):
    """The PDF exclusion is a measurement, not an oversight: Docling gives every
    PDF heading ``level=1`` and no ``title``, so the predicate would call them all
    top-level and 52 of the 275 shipped chunk rows would drift
    (``tmp/fmt-probe/probe_d62_drift.py``)."""

    for file_type in (".pdf", None):  # None = legacy artifact with no file_type
        rag = _rag(
            [
                _heading("e0001", "Attention Is All You Need", level=1),
                _heading("e0002", "Abstract", level=1),
                _paragraph("e0003", "The dominant sequence transduction model.", level=1),
            ],
            file_type=file_type,
        )
        ordered = refine_rag_document(rag, output_dir=tmp_path, copy_figures=False)
        norms = _norms(ordered)
        assert norms["e0002"] == ["Attention Is All You Need"], file_type
        assert norms["e0003"] == ["Attention Is All You Need", "Abstract"], file_type


def test_unnumbered_sub_heading_still_nests_outside_pdf(tmp_path: Path):
    """Only *top-level* unnumbered headings reset: a deeper one stays a child
    (measured on ``levels.docx``: Top A at L2, Sub A1 at L3)."""

    rag = _rag(
        [
            _heading("e0001", "Top A", level=2),
            _paragraph("e0002", "Body of Top A."),
            _heading("e0003", "Sub A1", level=3),
            _paragraph("e0004", "Body of Sub A1."),
        ],
        file_type=".docx",
    )
    ordered = refine_rag_document(rag, output_dir=tmp_path, copy_figures=False)
    norms = _norms(ordered)
    assert norms["e0003"] == ["Top A"]
    assert norms["e0004"] == ["Top A", "Sub A1"]


def test_spreadsheet_sheet_anchors_are_siblings(tmp_path: Path):
    """XLSX: the injected sheet heading (label ``sheet``, level 1) is the anchor
    every row of that worksheet cites through, so two sheets are two sections --
    not one nested inside the other."""

    rag = _rag(
        [
            _heading("e0001", "Sales-Q3", level=1, source_label="sheet"),
            _paragraph("e0002", "region | total"),
            _heading("e0003", "Notes-EU", level=1, source_label="sheet", page=2),
            _paragraph("e0004", "memo | carried", page=2),
        ],
        file_type=".xlsx",
    )
    ordered = refine_rag_document(rag, output_dir=tmp_path, copy_figures=False)
    norms = _norms(ordered)
    assert norms["e0003"] == []            # was ["Sales-Q3"] before D62
    assert norms["e0004"] == ["Notes-EU"]  # was ["Sales-Q3", "Notes-EU"]
