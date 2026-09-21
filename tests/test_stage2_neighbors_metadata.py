"""Tests for stage-2 neighbor evidence window helpers and Chroma metadata projection."""

from __future__ import annotations

from src.retrieval.chroma_store import _metadata
from src.retrieval.neighbor_expand import compose_evidence_window


def test_compose_evidence_window_orders_prev_primary_next():
    text = compose_evidence_window(
        [{"content": "prev-a"}, {"content": "prev-b"}],
        "TABLE BODY",
        [{"content": "next-a"}],
    )
    assert text.index("prev-a") < text.index("TABLE BODY") < text.index("next-a")
    assert "[primary]" in text
    assert "[context:prev #2]" in text


def test_metadata_projects_neighbors_and_bools_as_ints():
    projected = _metadata(
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "atomic": True,
            "degraded": False,
            "parent_id": "table:e1",
            "neighbor_prev_chunk_ids": ["a", "b"],
            "neighbor_next_chunk_ids": ["c"],
            "pack_ordinal": 2,
            "chunk_type": "table_pack",
        }
    )
    assert projected["atomic"] == 1
    assert projected["degraded"] == 0
    assert projected["parent_id"] == "table:e1"
    assert '"a"' in projected["neighbor_prev_chunk_ids"]
    assert projected["pack_ordinal"] == 2
