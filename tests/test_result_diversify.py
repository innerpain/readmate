"""Unit tests for final-window diversity and the table gate.

D21 (2026-09-20): the B/C noise heuristics (``is_bc_noise`` + the two
demo-corpus regexes) were deleted.  ``drops_unwanted_table`` is the gate's
enforcement only, so the corpus-hardcoded cases that used to live here are gone
along with the code they tested.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.retrieval.result_diversify import (
    collapse_key,
    diversify_hits,
    drops_unwanted_table,
    ensure_hit_present,
    prefers_over,
    query_wants_tables,
)


def _hit(chunk_id: str):
    return SimpleNamespace(chunk_id=chunk_id)


def test_diversify_keeps_every_prose_chunk_and_prefers_table_pack():
    """D22: prose no longer collapses per (document, page), so ``a`` and ``b`` --
    two passages of the same page -- both keep a slot.  Tables and figures keep
    their one-slot-per-table/figure constraint.

    NOTE: ``chunk_id`` is part of the metadata the retriever stores (chroma's
    whitelist keeps it), so the fixture carries it too -- without it the key
    falls back to ``document:page`` and the old per-page behaviour returns.
    """
    meta = {
        "a": {"chunk_id": "a", "document_id": "d1", "page": 3, "chunk_type": "prose"},
        "b": {"chunk_id": "b", "document_id": "d1", "page": 3, "chunk_type": "prose"},
        "c": {"chunk_id": "c", "document_id": "d1", "page": 4, "chunk_type": "prose"},
        "s": {"chunk_id": "s", "document_id": "d1", "page": 5, "chunk_type": "table_summary", "table_id": "t1"},
        "p": {"chunk_id": "p", "document_id": "d1", "page": 5, "chunk_type": "table_pack", "table_id": "t1"},
        "f": {"chunk_id": "f", "document_id": "d1", "page": 3, "chunk_type": "formula"},
    }
    hits = [_hit("a"), _hit("b"), _hit("s"), _hit("c"), _hit("p"), _hit("f")]
    # Metric-seeking query keeps tables; pack replaces summary.
    selected = diversify_hits(hits, meta, final_k=5, query="STS Spearman score in Table 1")
    ids = [h.chunk_id for h in selected]
    assert "a" in ids
    assert "b" in ids, "the second passage of page 3 must keep its own slot"
    assert "c" in ids
    assert "p" in ids
    assert "s" not in ids
    assert "f" in ids


def test_collapse_key_gives_each_prose_chunk_its_own_slot():
    """D22: the key was ``("prose", document_id, page)``, so the second paragraph
    of a page could never be retrieved even when it held the answer.  It is the
    chunk id now; table / figure / formula keep their slot constraint."""

    same_page_a = {"document_id": "d1", "page": 3, "chunk_type": "prose", "chunk_id": "c-1"}
    same_page_b = {"document_id": "d1", "page": 3, "chunk_type": "prose", "chunk_id": "c-2"}
    assert collapse_key(same_page_a) != collapse_key(same_page_b)
    assert collapse_key(same_page_a) == ("prose", "c-1")
    # ``chunk_id`` is what the key reads, so it must be the metadata's own id --
    # the retriever keeps it in ``metadata_by_id`` (chroma's whitelist includes it).
    assert collapse_key({"document_id": "d1", "page": 3, "chunk_type": "prose", "chunk_id": "c-9"}) == (
        "prose", "c-9",
    )
    # One slot per table / figure / formula is still the rule.
    assert collapse_key({"chunk_type": "formula", "document_id": "d1", "page": 3, "chunk_id": "x"}) == (
        "formula", "d1", 3,
    )
    assert collapse_key({"chunk_type": "table_summary", "table_id": "t1"}) == ("table", "t1")


def test_diversify_never_admits_a_table_the_question_did_not_ask_for():
    meta = {
        "gold": {"document_id": "d1", "page": 3, "chunk_type": "prose"},
        "results": {"document_id": "d1", "page": 9, "chunk_type": "table_summary", "table_id": "t3"},
    }
    hits = [_hit("results"), _hit("gold")]
    selected = diversify_hits(hits, meta, final_k=3, query="Transformer 的编码器由什么结构组成？")
    ids = [h.chunk_id for h in selected]
    assert ids[0] == "gold"
    assert "results" not in ids


def test_score_floor_drops_weak_tails():
    meta = {
        "a": {"document_id": "d1", "page": 1, "chunk_type": "prose"},
        "b": {"document_id": "d1", "page": 2, "chunk_type": "prose"},
        "c": {"document_id": "d1", "page": 3, "chunk_type": "prose"},
    }
    scores = {"a": 0.80, "b": 0.78, "c": 0.40}
    selected = diversify_hits(
        [_hit("a"), _hit("b"), _hit("c")],
        meta,
        final_k=5,
        query="model architecture layers",
        score_fn=lambda h: scores[h.chunk_id],
        score_floor_ratio=0.82,
    )
    ids = [h.chunk_id for h in selected]
    assert "a" in ids and "b" in ids
    assert "c" not in ids


def test_ensure_hit_present_inserts_dense_top():
    """D19-a: only the dense-top insurance survives -- the literal-anchor path
    that needed the keyword channel was deleted."""
    selected = [_hit("a"), _hit("b"), _hit("c")]
    out = ensure_hit_present(selected, _hit("dense1"), final_k=3)
    assert [h.chunk_id for h in out] == ["dense1", "a", "b"]


def test_helpers():
    assert query_wants_tables("NQ Exact Match 分数")
    assert not query_wants_tables("编码器有多少层")
    assert collapse_key({"table_id": "t", "chunk_type": "table_summary"})[0] == "table"
    assert prefers_over({"chunk_type": "table_pack"}, {"chunk_type": "table_summary"})


def test_the_gate_is_the_only_thing_left_and_reads_no_corpus_names():
    """D21: the dataset-name / paper-title regexes are gone, so a table is kept
    or dropped purely on ``allow_tables`` -- never on which paper it came from."""

    import inspect

    from src.retrieval import result_diversify

    assert not hasattr(result_diversify, "is_bc_noise")
    assert not hasattr(result_diversify, "_AUTHORISH_PATH")
    assert "nq" not in result_diversify._TABLE_METRIC_CUES.pattern
    assert "trivia" not in result_diversify._TABLE_METRIC_CUES.pattern
    assert "webquestions" not in result_diversify._TABLE_METRIC_CUES.pattern
    assert "curatedtrec" not in result_diversify._TABLE_METRIC_CUES.pattern
    assert "allow_tables" in inspect.signature(drops_unwanted_table).parameters


def test_shallow_path_summary_is_kept_when_table_gate_is_open():
    """R4 / diag B-3: a legitimate result table with a single-section path must
    survive when the query actually wants tables."""

    assert not drops_unwanted_table(
        {"chunk_type": "table_summary", "heading_path": ["6 Results"]},
        allow_tables=True,
    )


def test_front_matter_author_summary_is_now_kept_too():
    """D21: this table used to be filtered by a hardcoded paper-title regex --
    which only ever worked for the three demo papers.  The gate keeps it now;
    the per-channel quota pick is the remaining safety net."""

    assert not drops_unwanted_table(
        {"chunk_type": "table_summary", "heading_path": ["Attention Is All You Need"]},
        allow_tables=True,
    )
    assert drops_unwanted_table(
        {"chunk_type": "table_summary", "heading_path": ["Attention Is All You Need"]},
        allow_tables=False,
    )
