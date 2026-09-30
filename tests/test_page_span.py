"""A3: a passage that straddles a page break must report the pages it covers.

Root cause (2026-09-19): ``rag_export`` kept only ``prov[0].page_no`` of a Docling
text item, the chunker therefore saw a single page, and the citation for a
sentence printed on p4 was labelled p3.  These tests pin every hop of the span:
parse -> chunk -> index metadata -> retrieval contract -> citation -> eval hit.

No Docling / no embedding: every layer is exercised through its own seam.
"""

from __future__ import annotations

from types import SimpleNamespace

from eval.agent_eval import _covers_page
from src.adapter.rag_adapter import _to_hit
from src.agent.answer_protocol import AgentAnswer, CitationOut, validate_citations
from src.ingestion.ordered_chunker import _pages_of, chunk_ordered_document
from src.retrieval.index_builder import _to_index_chunk
from src.retrieval.persistent_retriever import _to_retrieved_chunk


def _el(element_id: str, *, text: str, ordinal: int, page: int, page_end: int | None = None) -> dict:
    return {
        "element_id": element_id,
        "ordinal": ordinal,
        "type": "paragraph",
        "role": "body",
        "index_policy": "embed",
        "page": page,
        "page_end": page_end,
        "heading_path_norm": ["2 Methods"],
        "text": text,
        "search_text": text,
        "structure": {},
        "atomic": False,
        "parse_status": "parsed",
        "payload": {},
    }


# --------------------------------------------------------------- chunker seam
def test_pages_of_collects_both_ends() -> None:
    assert _pages_of({"page": 3, "page_end": 4}) == [3, 4]
    assert _pages_of({"page": 3, "page_end": None}) == [3]
    assert _pages_of({"page": 3, "page_end": "4"}) == [3, 4]
    assert _pages_of({}) == [1]


def test_prose_chunk_keeps_the_page_span() -> None:
    """The Q16 shape: one 677-char paragraph starting on p3 and ending on p4."""

    ordered = {
        "schema": "ordered_document_v1",
        "document_id": "doc-rag",
        "revision": "r1",
        "source_pdf": "rag.pdf",
        "sequence": [
            _el("e0031", text="We jointly train the retriever and generator components without supervision.", ordinal=0, page=3, page_end=4),
        ],
    }
    chunks, _report = chunk_ordered_document(ordered, filename="rag.pdf")
    prose = [chunk for chunk in chunks if chunk["chunk_type"] == "prose"]
    assert prose, "expected a prose chunk"
    chunk = prose[0]
    assert chunk["page"] == 3
    assert chunk["page_end"] == 4, "the chunk must not claim to end where it starts"
    assert chunk["pages"] == [3, 4]


def test_single_page_chunk_reports_one_page() -> None:
    ordered = {
        "schema": "ordered_document_v1",
        "document_id": "doc-rag",
        "revision": "r1",
        "source_pdf": "rag.pdf",
        "sequence": [_el("e0001", text="A paragraph that lives on one page only.", ordinal=0, page=7)],
    }
    chunks, _report = chunk_ordered_document(ordered, filename="rag.pdf")
    prose = [chunk for chunk in chunks if chunk["chunk_type"] == "prose"][0]
    assert prose["page"] == 7
    assert prose["page_end"] == 7


# ---------------------------------------------------------- index / retrieval
def test_index_metadata_carries_the_span() -> None:
    straddling = _to_index_chunk({"chunk_id": "c1", "page": 3, "page_end": 4, "text": "x"})
    assert straddling.metadata["page"] == 3
    assert straddling.metadata["page_end"] == 4

    legacy = _to_index_chunk({"chunk_id": "c2", "page": 5, "text": "x"})
    assert legacy.metadata["page_end"] == 5, "a chunk without a span ends where it starts"


def test_retrieval_contract_exposes_the_span() -> None:
    chunk = _to_retrieved_chunk(
        chunk_id="c1",
        content="text",
        metadata={"document_id": "d1", "filename": "rag.pdf", "page": 3, "page_end": 4, "section": "2.4"},
        score=0.9,
    )
    assert (chunk.page, chunk.page_end) == (3, 4)

    old_snapshot = _to_retrieved_chunk(
        chunk_id="c2",
        content="text",
        metadata={"document_id": "d1", "filename": "rag.pdf", "page": 3, "section": "2.4"},
        score=0.9,
    )
    assert (old_snapshot.page, old_snapshot.page_end) == (3, 0), "pre-upgrade snapshots read as 0/unknown"


def test_adapter_hit_carries_the_span_only_when_it_spans() -> None:
    spanning = _to_hit(
        SimpleNamespace(
            chunk_id="c1", document_id="d1", filename="rag.pdf", page=3, page_end=4,
            section="2.4", heading_path=[], score=0.9, chunk_type="prose", content_kind="text",
            table_id=None, figure_id=None, atomic=False, degraded_structure=False,
            presentation_content="body", content="body", expanded_neighbors={},
        ),
        excerpt_chars=200,
    )
    assert (spanning.page, spanning.page_end) == (3, 4)

    single = _to_hit(
        SimpleNamespace(
            chunk_id="c2", document_id="d1", filename="rag.pdf", page=3, page_end=3,
            section="2.4", heading_path=[], score=0.9, chunk_type="prose", content_kind="text",
            table_id=None, figure_id=None, atomic=False, degraded_structure=False,
            presentation_content="body", content="body", expanded_neighbors={},
        ),
        excerpt_chars=200,
    )
    assert single.page_end is None, "an equal end page is noise, not a span"


# --------------------------------------------------------------------- citation
SPAN_META = {"document_id": "d1", "filename": "rag.pdf", "page": 3, "page_end": 4}


def _cite(model_page: int | None, meta: dict | None = None) -> CitationOut:
    answer = AgentAnswer(text="answer", citations=[CitationOut(chunk_id="c1", page=model_page)])
    validated = validate_citations(answer, {"c1": meta or SPAN_META})
    assert validated.citations, "the citation was observed, so it must survive"
    return validated.citations[0]


def test_model_page_inside_the_span_is_kept() -> None:
    """The model read "p3-4" and named p4 -- that is the page its quote sits on."""

    citation = _cite(4)
    assert citation.page == 4
    assert citation.page_end == 4


def test_model_page_outside_the_span_falls_back_to_the_observed_page() -> None:
    assert _cite(9).page == 3
    assert _cite(1).page == 3


def test_missing_model_page_uses_the_observed_start() -> None:
    assert _cite(None).page == 3


def test_legacy_observation_without_span_behaves_as_before() -> None:
    legacy = {"document_id": "d1", "filename": "rag.pdf", "page": 5}
    assert _cite(None, legacy).page == 5
    assert _cite(5, legacy).page == 5
    assert _cite(2, legacy).page == 5
    assert _cite(5, legacy).page_end == 5


# --------------------------------------------------------------------- eval hit
def test_eval_counts_a_span_that_covers_the_expected_page() -> None:
    assert _covers_page({"page": 3, "page_end": 4}, {4}) is True
    assert _covers_page({"page": 3, "page_end": 4}, {3}) is True
    assert _covers_page({"page": 3, "page_end": 4}, {5}) is False
    assert _covers_page({"page": 3, "page_end": None}, {4}) is False
    assert _covers_page({"page": None, "page_end": None}, {4}) is False


# --------------------------------------------------------------- HTTP response
def test_response_schema_keeps_the_span() -> None:
    """The HTTP schema is the last hop before the UI; an undeclared field is
    dropped by pydantic and the citation quietly falls back to the start page."""

    from src.models.agent_schemas import AgentChatResponse

    response = AgentChatResponse(
        session_id="ses_x",
        mode="deep",
        answer="answer",
        citations=[{"chunk_id": "c1", "page": 3, "page_end": 4, "quote": "q"}],
    )
    assert response.citations[0].page_end == 4
    assert response.model_dump()["citations"][0]["page_end"] == 4


# ------------------------------------------------- D63: no-heading documents
def test_a_document_without_headings_has_no_section_label() -> None:
    """D63 (2026-09-30): a heading-less document (CSV, or a DOCX/MD/HTML without
    headings) used to reach the user as the literal ``"Unknown"`` -- the preview
    modal renders ``section``, so that word was printed above the passage.  The
    honest value is empty, and every hop has to agree."""

    ordered = {
        "schema": "ordered_document_v1",
        "document_id": "d1",
        "source_pdf": "data.csv",
        "file_type": ".csv",
        "page_kind": "section",
        "sequence": [
            {
                "ordinal": 0,
                "element_id": "e0001",
                "type": "table",
                "role": "body",
                "index_policy": "embed",
                "page": 1,
                "page_end": 1,
                "heading_path": [],
                "heading_path_norm": [],  # no headings anywhere in this document
                "text": "名称 | 数量",
                "search_text": "名称 | 数量",
                "structure": {
                    "caption": "",
                    "headers": ["名称", "数量"],
                    "rows": [["苹果", "3"]],
                },
                "atomic": False,
                "parse_status": "parsed",
                "payload": {},
            }
        ],
    }
    chunks, _ = chunk_ordered_document(ordered, filename="data.csv")
    assert chunks, "the table still has to be chunked"
    assert all(chunk["section"] == "" for chunk in chunks), [c["section"] for c in chunks]

    indexed = _to_index_chunk({**chunks[0], "document_id": "d1"})
    assert indexed.metadata["section"] == ""

    retrieved = _to_retrieved_chunk(
        chunk_id=chunks[0]["chunk_id"],
        content=chunks[0]["retrieval_text"],
        metadata=indexed.metadata,
        score=0.5,
    )
    assert retrieved.section == ""


def test_a_legacy_unknown_sentinel_is_not_invented_at_read_time() -> None:
    """A snapshot published before D63 may still carry ``"Unknown"``: the read
    path passes whatever is stored through instead of minting a new label."""

    retrieved = _to_retrieved_chunk(
        chunk_id="c1",
        content="text",
        metadata={"document_id": "d1", "filename": "legacy.csv", "page": 1, "section": "Unknown"},
        score=0.5,
    )
    assert retrieved.section == "Unknown"
