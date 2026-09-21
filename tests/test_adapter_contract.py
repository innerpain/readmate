"""Contract tests for the frozen adapter surface (AD2 / AD3)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.adapter.contracts import (
    ADAPTER_API_VERSION,
    COLLECTION_EMPTY,
    ERROR_CODES,
    KNOWLEDGE_BASE_EMPTY,
    REQUIRED_READ_FIELDS,
    REQUIRED_SEARCH_FIELDS,
    AdapterError,
    ReadResult,
    SearchHit,
)
from src.adapter.mock_adapter import MockRagAdapter
from src.adapter.rag_adapter import RagAdapterImpl, _to_hit
from src.agent.tools import ToolRunner
from src.llm.types import ToolCall


def _hit(chunk_id: str = "c1", document_id: str = "d1", page: int = 3, score: float = 0.8) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="attention.pdf",
        page=page,
        section="3 Model Architecture",
        score=score,
        chunk_type="prose",
        presentation_content="encoders are a stack of six layers",
        excerpt="six layers",
    )


def test_api_version_is_frozen():
    assert ADAPTER_API_VERSION == "readmate-rag-adapter-3"


def test_search_hit_exposes_every_required_field():
    assert set(REQUIRED_SEARCH_FIELDS) <= set(SearchHit.model_fields)


def test_read_result_exposes_every_required_field():
    assert set(REQUIRED_READ_FIELDS) <= set(ReadResult.model_fields)
    # New fields default cleanly so every existing ReadResult(...) call keeps working.
    minimal = ReadResult(document_id="d1")
    assert minimal.filename == ""
    assert minimal.chunk_ids == []
    assert minimal.remaining_chars is None and minimal.next_offset is None


def test_the_drift_guards_cover_page_end():
    """D36 (2026-09-20): ``page_end`` was added to both models by A3 but never
    to these two tuples, so the "the contract cannot drift silently" assertion
    did not cover the newest field at all."""

    assert "page_end" in REQUIRED_SEARCH_FIELDS
    assert "page_end" in REQUIRED_READ_FIELDS
    assert set(REQUIRED_SEARCH_FIELDS) <= set(SearchHit.model_fields)
    assert set(REQUIRED_READ_FIELDS) <= set(ReadResult.model_fields)


def test_error_code_table_covers_the_documented_codes():
    for code in (
        "collection_empty",
        "knowledge_base_empty",
        "document_not_in_collection",
        "chunk_not_found",
        "document_not_found",
        "embedding_incompatible",
        "build_locked",
    ):
        assert code in ERROR_CODES


def test_adapter_error_carries_a_stable_code():
    error = AdapterError(COLLECTION_EMPTY, "collection has no documents")
    assert error.code == "collection_empty"
    assert str(error) == "collection_empty: collection has no documents"


def test_mock_adapter_scopes_hits_to_the_requested_documents():
    adapter = MockRagAdapter(hits=[_hit("c1", "d1"), _hit("c2", "d2")])
    scoped = adapter.search("col", "layers", document_ids=["d1"])
    assert [hit.document_id for hit in scoped] == ["d1"]
    assert len(adapter.search_calls) == 1
    assert adapter.search_calls[0]["document_ids"] == ["d1"]


def test_mock_adapter_raises_the_scripted_error():
    adapter = MockRagAdapter(error=AdapterError(KNOWLEDGE_BASE_EMPTY, "no snapshot"))
    with pytest.raises(AdapterError) as raised:
        adapter.search("col", "layers")
    assert raised.value.code == "knowledge_base_empty"


def test_to_hit_prefers_the_presentation_window_and_keeps_the_dense_score():
    chunk = SimpleNamespace(
        chunk_id="c9",
        document_id="d1",
        filename="attention.pdf",
        page=5,
        section="6.2",
        heading_path=["6 Results", "6.2 Model Variations"],
        score=0.61,
        chunk_type="table_pack",
        content_kind="table",
        table_id="t-3",
        figure_id=None,
        atomic=True,
        degraded_structure=False,
        content="raw rows",
        presentation_content="[context:prev #1] header rows [primary] Table 3",
        expanded_neighbors={"prev": [{"content": "intro"}], "next": []},
    )
    hit = _to_hit(chunk, 40)
    assert hit.presentation_content.startswith("[context:prev #1]")
    assert hit.excerpt.startswith("[context:prev #1]")
    assert hit.score == 0.61
    assert hit.chunk_type == "table_pack"
    assert hit.atomic is True
    assert hit.table_id == "t-3"
    assert hit.heading_path == ["6 Results", "6.2 Model Variations"]


def test_to_hit_falls_back_to_content_when_no_window_exists():
    chunk = SimpleNamespace(
        chunk_id="c10",
        document_id="d1",
        filename="attention.pdf",
        page=2,
        section="2",
        heading_path=[],
        score=0.5,
        chunk_type="prose",
        content_kind="text",
        table_id=None,
        figure_id=None,
        atomic=False,
        degraded_structure=False,
        content="plain body text",
        presentation_content=None,
        expanded_neighbors={},
    )
    hit = _to_hit(chunk, 40)
    assert hit.presentation_content == "plain body text"
    assert hit.excerpt == "plain body text"
    assert hit.expanded_neighbors is None


def _write_chunks(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _adapter(tmp_path, registry=None):
    return RagAdapterImpl(
        retriever=None,  # not touched by the read surface
        resolve_documents=lambda _cid: [],
        registry=registry or SimpleNamespace(get_document=lambda _d: None),
        parsed_dir=tmp_path,
    )


def test_read_page_reports_how_much_text_is_left(tmp_path):
    """D34 (2026-09-20): the page branch cut the text at the ceiling and reported
    a bare ``truncated: bool``, so the model could not tell how much it had not
    seen nor how to resume.  The adapter states the truth now."""

    _write_chunks(
        tmp_path / "doc-long" / "chunks.jsonl",
        [{"chunk_id": "chk_long", "document_id": "doc-long", "filename": "long.pdf", "page": 1,
          "source_ordinal": 1, "pack_ordinal": -1, "text": "x" * 900, "chunk_type": "prose"}],
    )
    adapter = _adapter(tmp_path)
    adapter.page_chars = 200  # the ceiling is settings-driven (D35)
    result = adapter.read(document_id="doc-long", page=1)
    assert result.truncated is True
    assert len(result.text) == 200
    assert result.remaining_chars == 700
    assert result.next_offset == 200

    # A page that fits reports nothing left, so the model is not told to resume.
    _write_chunks(
        tmp_path / "doc-short" / "chunks.jsonl",
        [{"chunk_id": "chk_short", "document_id": "doc-short", "filename": "short.pdf", "page": 1,
          "source_ordinal": 1, "pack_ordinal": -1, "text": "small body", "chunk_type": "prose"}],
    )
    short = _adapter(tmp_path).read(document_id="doc-short", page=1)
    assert short.truncated is False
    assert short.remaining_chars is None and short.next_offset is None


def test_read_page_returns_filename_and_citable_chunk_ids(tmp_path):
    """A7/D6: a page read now yields a filename and one citable chunk_id per row."""

    _write_chunks(
        tmp_path / "doc-a" / "chunks.jsonl",
        [
            {"chunk_id": "chk_a", "document_id": "doc-a", "filename": "attention.pdf", "page": 8,
             "source_ordinal": 1, "pack_ordinal": -1, "text": "first block", "chunk_type": "prose"},
            {"chunk_id": "chk_b", "document_id": "doc-a", "filename": "attention.pdf", "page": 8,
             "source_ordinal": 2, "pack_ordinal": -1, "text": "second block", "chunk_type": "prose"},
        ],
    )
    result = _adapter(tmp_path).read(document_id="doc-a", page=8)
    assert result.filename == "attention.pdf"
    assert result.chunk_id is None
    assert result.chunk_ids == ["chk_a", "chk_b"]


def test_read_page_falls_back_to_registry_filename(tmp_path):
    """Metadata missing the filename must still resolve to the registry's name (D6 root cause)."""

    registry = SimpleNamespace(get_document=lambda doc_id: SimpleNamespace(original_filename="reg.pdf"))
    _write_chunks(
        tmp_path / "doc-b" / "chunks.jsonl",
        [{"chunk_id": "chk_c", "document_id": "doc-b", "page": 2,
          "source_ordinal": 1, "pack_ordinal": -1, "text": "body", "chunk_type": "prose"}],
    )
    result = _adapter(tmp_path, registry=registry).read(document_id="doc-b", page=2)
    assert result.filename == "reg.pdf"
    assert result.chunk_ids == ["chk_c"]


def test_page_read_hits_reach_the_tool_observation(tmp_path):
    """Every page chunk_id must appear in the tool observation so ``validate_citations``
    keeps citations for page-level reads (the reason page evidence used to vanish)."""

    _write_chunks(
        tmp_path / "doc-c" / "chunks.jsonl",
        [
            {"chunk_id": "chk_x", "document_id": "doc-c", "filename": "attention.pdf", "page": 4,
             "source_ordinal": 1, "pack_ordinal": -1, "text": "alpha", "chunk_type": "prose"},
            {"chunk_id": "chk_y", "document_id": "doc-c", "filename": "attention.pdf", "page": 4,
             "source_ordinal": 2, "pack_ordinal": -1, "text": "beta", "chunk_type": "table_pack"},
        ],
    )
    runner = ToolRunner(
        adapter=_adapter(tmp_path),
        collections=None,
        memory=None,
        settings=SimpleNamespace(tool_text_chars=2000, tool_max_hits=8),
    )
    observation = runner._read(ToolCall(id="1", name="read", arguments={"document_id": "doc-c", "page": 4}))
    assert observation.ok
    assert [hit["chunk_id"] for hit in observation.hits] == ["chk_x", "chk_y"]
    assert all(hit["filename"] == "attention.pdf" for hit in observation.hits)


def test_chunk_read_emits_a_single_hit_with_filename():
    """A chunk-level read stays a single citable hit and now carries the filename."""

    adapter = MockRagAdapter(reads={
        "c1": ReadResult(document_id="d1", chunk_id="c1", page=3, text="six layers", filename="attention.pdf")
    })
    runner = ToolRunner(
        adapter=adapter,
        collections=None,
        memory=None,
        settings=SimpleNamespace(tool_text_chars=2000, tool_max_hits=8),
    )
    observation = runner._read(ToolCall(id="1", name="read", arguments={"chunk_id": "c1"}))
    assert [hit["chunk_id"] for hit in observation.hits] == ["c1"]
    assert observation.hits[0]["filename"] == "attention.pdf"



def _name_registry(pairs):
    """Registry stand-in whose list_documents yields records with a UUID + name."""
    from types import SimpleNamespace as _NS
    records = [_NS(document_id=d, original_filename=n) for d, n in pairs]
    return _NS(
        list_documents=lambda: list(records),
        get_document=lambda doc_id: next((r for r in records if r.document_id == doc_id), None),
    )


def test_read_page_resolves_a_filename_to_the_real_document_id(tmp_path):
    """A1 (D-9c): passing a *filename* instead of a UUID must resolve, so the
    read succeeds instead of burning a step on document_not_found."""

    _write_chunks(
        tmp_path / "uuid-77" / "chunks.jsonl",
        [{"chunk_id": "chk_f", "document_id": "uuid-77", "filename": "attention.pdf", "page": 4,
          "source_ordinal": 1, "pack_ordinal": -1, "text": "body", "chunk_type": "prose"}],
    )
    registry = _name_registry([("uuid-77", "attention.pdf")])
    result = _adapter(tmp_path, registry=registry).read(document_id="attention.pdf", page=4)
    assert result.document_id == "uuid-77"
    assert result.chunk_ids == ["chk_f"]


def test_read_page_ambiguous_filename_raises_invalid_arguments(tmp_path):
    """Two stored documents share a name: the fallback must refuse rather than
    guess a target."""

    registry = _name_registry([("uuid-a", "paper.pdf"), ("uuid-b", "paper.pdf")])
    with pytest.raises(AdapterError) as raised:
        _adapter(tmp_path, registry=registry).read(document_id="paper.pdf", page=1)
    assert raised.value.code == "invalid_arguments"


def test_read_page_unknown_name_still_document_not_found(tmp_path):
    registry = _name_registry([("uuid-a", "attention.pdf")])
    with pytest.raises(AdapterError) as raised:
        _adapter(tmp_path, registry=registry).read(document_id="does-not-exist.pdf", page=1)
    assert raised.value.code == "document_not_found"
