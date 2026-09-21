"""Tests for the primary-first presentation window (R3 / A5)."""

from __future__ import annotations

from src.retrieval.neighbor_expand import build_presentation


def _part(chunk_id: str, content: str) -> dict[str, str]:
    return {"chunk_id": chunk_id, "content": content, "page": "1", "section": ""}


def test_no_neighbors_primary_gets_full_budget():
    """Without neighbours, ``primary`` uses the whole budget (marker appended
    only if it overflows)."""

    primary = "P" * 500
    text = build_presentation(primary, [], [], budget=600)
    assert text.startswith("[primary]\n")
    assert "P" * 500 in text
    assert "…[本体省略" not in text


def test_no_neighbors_primary_overflow_marks_body_truncated():
    primary = "P" * 2000
    text = build_presentation(primary, [], [], budget=300)
    assert text.startswith("[primary]\n")
    assert "…[本体省略 1710 字符]" in text   # 300 - len("[primary]\n") = 290 body chars
    assert len(text) <= 300 + 30   # marker may spill; still tight


def test_primary_first_ordering_survives_neighbours():
    """A5 fix: the [primary] block must start at offset 0 regardless of how
    many neighbours are attached, so a downstream head-slice still sees the
    body.  ``compose_evidence_window`` used to bury primary under neighbours
    (diag B-2: 16/21 table_packs had primary_starts_at >= 1200)."""

    primary = "P" * 400
    prev = [_part("p1", "A" * 300), _part("p2", "B" * 300)]
    nxt = [_part("n1", "C" * 300), _part("n2", "D" * 300)]
    text = build_presentation(primary, prev, nxt, budget=1200)
    assert text.index("[primary]\n") == 0


def test_neighbor_order_prev1_next1_prev2_next2():
    """Neighbours fill leftover in ``prev1 -> next1 -> prev2 -> next2``."""

    primary = "P" * 100
    prev = [_part("p1", "PREV1"), _part("p2", "PREV2")]
    nxt = [_part("n1", "NEXT1"), _part("n2", "NEXT2")]
    text = build_presentation(primary, prev, nxt, budget=1200)
    order = [text.index(tag) for tag in ("PREV1", "NEXT1", "PREV2", "NEXT2")]
    assert order == sorted(order)


def test_primary_short_neighbours_take_leftover():
    """Short primary (below the floor) yields leftover to neighbours; nothing
    is wasted."""

    primary = "P" * 50
    neighbour_content = "N" * 300
    prev = [_part("p1", neighbour_content), _part("p2", neighbour_content)]
    text = build_presentation(primary, prev, [], budget=1000)
    assert text.count(neighbour_content) == 2   # both neighbours included whole


def test_primary_long_preserves_floor_and_drops_tight_neighbours():
    """Long primary still keeps at least ``primary_min_ratio*budget`` chars of
    body visible; when a neighbour slot cannot host ``neighbor_min_chars`` it
    is replaced by an omission marker."""

    primary = "P" * 900
    neighbour_content = "N" * 500
    prev = [_part("p1", neighbour_content), _part("p2", neighbour_content)]
    nxt = [_part("n1", neighbour_content), _part("n2", neighbour_content)]
    text = build_presentation(
        primary, prev, nxt, budget=1000,
        primary_min_ratio=0.6, neighbor_min_chars=120,
    )
    assert "P" * 900 in text          # primary fully visible (<= ceiling)
    assert "…[邻居省略" in text         # at least one neighbour omitted


def test_neighbor_omission_marker_reports_char_count():
    """The omission marker must state how many characters were dropped, so the
    reader knows what they're missing."""

    primary = "P" * 100
    prev = [_part("p1", "A" * 200), _part("p2", "B" * 200)]
    nxt = [_part("n1", "C" * 200), _part("n2", "D" * 200)]
    text = build_presentation(
        primary, prev, nxt, budget=150,
        primary_min_ratio=0.6, neighbor_min_chars=120,
    )
    # budget 150, header 10, floor 90 -> primary 100 > floor, so primary_budget=ceil(140)
    # Actually primary (100) < ceiling (140): primary_budget = primary_len = 100
    # used = 110; each neighbour: header 18 + room 150-110-2-18=20 < neighbor_min_chars(120)
    # -> all four neighbours dropped with markers
    assert "…[邻居省略 200 字符]" in text


def test_zero_budget_returns_empty():
    assert build_presentation("anything", [], [], budget=0) == ""
    assert build_presentation("anything", [{"content": "x"}], [], budget=0) == ""


def test_output_never_exceeds_budget():
    """Hard invariant: build_presentation is a *budgeted* window.  Downstream
    slices are unnecessary."""

    primary = "P" * 2500
    prev = [_part("p1", "A" * 400), _part("p2", "B" * 400)]
    nxt = [_part("n1", "C" * 400), _part("n2", "D" * 400)]
    text = build_presentation(primary, prev, nxt, budget=1200)
    assert len(text) <= 1200
