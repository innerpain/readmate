"""Unit tests for OrderedChunker (no Docling / no embedding required)."""

from __future__ import annotations

from src.ingestion.ordered_chunker import CHUNKER_VERSION, chunk_ordered_document


def _el(
    element_id: str,
    type_: str,
    *,
    text: str,
    ordinal: int,
    page: int = 1,
    role: str = "body",
    index_policy: str = "embed",
    heading_path=None,
    structure=None,
    atomic: bool = False,
    parse_status: str = "parsed",
    search_text: str | None = None,
):
    return {
        "element_id": element_id,
        "ordinal": ordinal,
        "type": type_,
        "role": role,
        "index_policy": index_policy,
        "page": page,
        "heading_path_norm": heading_path or ["Section"],
        "text": text,
        "search_text": search_text if search_text is not None else text,
        "structure": structure or {},
        "atomic": atomic,
        "parse_status": parse_status,
        "payload": {},
    }


def test_prose_respects_max_and_skips_drop_and_byline():
    long_a = "Alpha sentence. " * 80  # well over 512 tokens by char/4 heuristic
    long_b = "Beta sentence. " * 80
    ordered = {
        "schema": "ordered_document_v1",
        "document_id": "doc1",
        "revision": "r1",
        "source_pdf": "doc1.pdf",
        "sequence": [
            _el("e1", "heading", text="Title", ordinal=0, heading_path=["Title"]),
            _el("e2", "paragraph", text="Nils Reimers", ordinal=1, heading_path=["Title"]),
            _el("e3", "paragraph", text="www.example.com", ordinal=2, heading_path=["Title"]),
            _el("e4", "heading", text="Abstract", ordinal=3, heading_path=["Abstract"]),
            _el("e5", "paragraph", text=long_a, ordinal=4, heading_path=["Abstract"]),
            _el("e6", "paragraph", text=long_b, ordinal=5, heading_path=["Abstract"]),
            _el(
                "e7",
                "list_item",
                text="Dropped ref",
                ordinal=6,
                role="reference",
                index_policy="drop",
                heading_path=["References"],
            ),
        ],
    }
    chunks, report = chunk_ordered_document(ordered, filename="doc1.pdf")
    assert CHUNKER_VERSION == "ordered-aware-1"
    assert report["demoted_byline"] >= 2
    # D17 (2026-09-20): the policy pass used to drop elements silently -- the
    # report said nothing, so "how much text never made it into the corpus" was
    # unanswerable.  ``skip`` is also the honest name for what ``metadata_only``
    # did (the text does not reach the corpus at all).
    assert report["dropped_by_role"] == {"reference": 1}
    assert report["dropped_total"] == 1
    prose = [c for c in chunks if c["chunk_type"] == "prose"]
    assert prose
    for c in prose:
        assert c["index_policy_effective"] == "embed"
        assert estimate_ok(c["retrieval_text"])
        assert "Dropped ref" not in c["retrieval_text"]
        assert "Nils Reimers" not in c["retrieval_text"]
        assert "www.example.com" not in c["retrieval_text"]
    assert all(c["chunker_version"] == CHUNKER_VERSION for c in chunks)


def estimate_ok(text: str, limit: int = 512) -> bool:
    return max(1, len(text) // 4) <= limit + 8  # small slack for path prefix rounding


def test_formula_stays_atomic_single_chunk():
    latex = r"o = \mathrm{softmax}(W_t(u,v,|u-v|))"
    ordered = {
        "schema": "ordered_document_v1",
        "document_id": "doc1",
        "revision": "r1",
        "sequence": [
            _el("e1", "heading", text="3 Model", ordinal=0, heading_path=["3 Model"]),
            _el(
                "e2",
                "paragraph",
                text="We introduce the architecture.",
                ordinal=1,
                heading_path=["3 Model"],
            ),
            _el(
                "e3",
                "formula",
                text=latex,
                ordinal=2,
                heading_path=["3 Model"],
                atomic=True,
                parse_status="partial",
                search_text=f"3 Model [formula] {latex} Classification Objective Function.",
                structure={"latex": latex, "context": "Classification Objective Function."},
            ),
            _el(
                "e4",
                "paragraph",
                text="Then we discuss inference.",
                ordinal=3,
                heading_path=["3 Model"],
            ),
        ],
    }
    chunks, _report = chunk_ordered_document(ordered, filename="x.pdf")
    formulas = [c for c in chunks if c["chunk_type"] == "formula"]
    assert len(formulas) == 1
    assert formulas[0]["atomic"] == 1
    assert latex in formulas[0]["retrieval_text"]
    assert formulas[0]["element_ids"] == ["e3"]


def test_table_summary_and_packs_with_neighbors():
    facts = []
    rows = []
    for i in range(40):
        row = [f"Model-{i}", str(50 + i), str(60 + i), str(70 + i)]
        rows.append(row)
        for col, val in zip(["STS12", "STS13", "STS14"], row[1:]):
            facts.append(
                f"[table-fact] Table 1: caption long " + ("x" * 40) + f" | {row[0]} | {col} | {val}"
            )
    # Deliberately huge compiled search_text (must NOT become one retrieval vector)
    bloated = "\n".join(facts * 3)
    ordered = {
        "schema": "ordered_document_v1",
        "document_id": "doc1",
        "revision": "r1",
        "sequence": [
            _el(
                "e1",
                "paragraph",
                text="Previous context about evaluation setup and metrics used.",
                ordinal=0,
                heading_path=["4 Experiments"],
            ),
            _el(
                "e2",
                "paragraph",
                text="More previous prose describing the STS benchmark protocol.",
                ordinal=1,
                heading_path=["4 Experiments"],
            ),
            _el(
                "e3",
                "table",
                text="Table 1",
                ordinal=2,
                heading_path=["4 Experiments"],
                atomic=True,
                search_text=bloated,
                structure={
                    "caption": "Table 1: Spearman rank correlation for STS tasks. " + ("detail " * 30),
                    "headers": ["Model", "STS12", "STS13", "STS14"],
                    "rows": rows,
                    "facts": facts,
                },
            ),
            _el(
                "e4",
                "paragraph",
                text="Following discussion interprets the table averages.",
                ordinal=3,
                heading_path=["4 Experiments"],
            ),
            _el(
                "e5",
                "paragraph",
                text="Another following paragraph on ablation intent.",
                ordinal=4,
                heading_path=["4 Experiments"],
            ),
        ],
    }
    chunks, report = chunk_ordered_document(ordered, filename="paper.pdf")
    summaries = [c for c in chunks if c["chunk_type"] == "table_summary"]
    packs = [c for c in chunks if c["chunk_type"] == "table_pack"]
    prose = [c for c in chunks if c["chunk_type"] == "prose"]
    assert len(summaries) == 1
    assert len(packs) >= 2
    assert summaries[0]["parent_id"].startswith("table:")
    assert summaries[0]["table_id"]
    for c in summaries + packs:
        assert estimate_ok(c["retrieval_text"])
        assert len(c["retrieval_text"]) < len(bloated) // 2
        assert len(c["neighbor_prev_chunk_ids"]) >= 1
        assert len(c["neighbor_prev_chunk_ids"]) <= 2
        assert len(c["neighbor_next_chunk_ids"]) >= 1
        assert len(c["neighbor_next_chunk_ids"]) <= 2
        # neighbors must be prose chunk ids
        prose_ids = {p["chunk_id"] for p in prose}
        assert set(c["neighbor_prev_chunk_ids"]) <= prose_ids
        assert set(c["neighbor_next_chunk_ids"]) <= prose_ids
    assert report["n_chunks_by_type"]["table_summary"] == 1
    assert report["n_chunks_by_type"]["table_pack"] == len(packs)


def test_figure_caption_chunk_binds_neighbors():
    ordered = {
        "schema": "ordered_document_v1",
        "document_id": "doc1",
        "revision": "r1",
        "sequence": [
            _el(
                "e1",
                "paragraph",
                text="Architecture overview comes first in the section.",
                ordinal=0,
                heading_path=["3 Model"],
            ),
            _el(
                "e2",
                "figure",
                text="Figure 1",
                ordinal=1,
                heading_path=["3 Model"],
                parse_status="skipped_image",
                atomic=True,
                search_text="3 Model [image] Figure 1: SBERT architecture with classification objective.",
                structure={
                    "caption": "Figure 1: SBERT architecture with classification objective.",
                    "image_path": "figures/fig1.png",
                },
            ),
            _el(
                "e3",
                "paragraph",
                text="We then formalize the objective function.",
                ordinal=2,
                heading_path=["3 Model"],
            ),
        ],
    }
    chunks, _report = chunk_ordered_document(ordered, filename="x.pdf")
    figs = [c for c in chunks if c["chunk_type"] == "figure"]
    assert len(figs) == 1
    assert "Figure 1" in figs[0]["retrieval_text"]
    assert figs[0]["image_path"].endswith("fig1.png")
    assert figs[0]["neighbor_prev_chunk_ids"]
    assert figs[0]["neighbor_next_chunk_ids"]


def test_legacy_metadata_only_policy_is_still_dropped():
    """D17 renamed the policy to ``skip`` on the producer side only.

    The ``ordered.json`` files already on disk were written *before* that rename and
    still say ``metadata_only`` -- and ``index_builder`` re-chunks from those very
    files on every rebuild, so the consumer has to keep recognising the old name.
    Without this, the elements silently start being embedded (measured on the real
    corpus: 9 body-role elements across 5 documents) while
    ``document_quality._dropped_by_role`` -- which mirrors this rule -- would report
    drops that never happened.
    """

    ordered = {
        "document_id": "doc_legacy",
        "sequence": [
            _el("e1", "paragraph", text="legacy policy row", ordinal=1, index_policy="metadata_only"),
            _el("e2", "paragraph", text="renamed policy row", ordinal=2, index_policy="skip"),
            _el("e3", "paragraph", text="indexed row", ordinal=3),
        ],
    }
    chunks, report = chunk_ordered_document(ordered, filename="legacy.pdf")

    indexed = {element_id for chunk in chunks for element_id in chunk.get("element_ids", [])}
    assert "e1" not in indexed, "the pre-rename policy name must still mean 'do not embed'"
    assert "e2" not in indexed
    assert "e3" in indexed
    assert report["dropped_by_role"] == {"body": 2}
