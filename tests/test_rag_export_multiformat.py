"""D61: rag_export changes -- PDF-only cross-page merge, sheet anchor injection,
empty-table drop, and the level passthrough the section rule needs.

The items below use Docling's **real** label enums: ``map_type`` gates on
``isinstance(label, DocItemLabel)``, so a stub label would silently drop every
element and the test would assert on an empty stream (the first version of this
file did exactly that).
"""

from __future__ import annotations

import pandas as pd
import pytest
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import GroupLabel

from src.ingestion.rag_export import build_rag_document


class Prov:
    def __init__(self, page_no: int) -> None:
        self.page_no = page_no
        self.bbox = None


class Item:
    """A Docling-shaped item: label, text, optional group name, page provenance."""

    def __init__(
        self,
        label,
        text: str = "",
        *,
        name: str | None = None,
        pages: list[int] | None = None,
        frame: pd.DataFrame | None = None,
    ) -> None:
        self.label = label
        self.text = text
        self.name = name
        self.prov = [Prov(page) for page in (pages or [])]
        self.captions = []
        self.image = None
        self._frame = frame

    def caption_text(self, doc=None) -> str:
        return ""

    def export_to_markdown(self, doc=None) -> str:
        return self.text

    def export_to_dataframe(self, doc=None):
        if self._frame is None:
            raise ValueError("no table data")
        return self._frame


class FakeDoc:
    """Minimal stand-in for DoclingDocument: only what build_rag_document reads."""

    def __init__(self, items: list[tuple]) -> None:
        self._items = items

    def iterate_items(self, **kwargs):
        return iter(self._items)


def build(items, *, file_type: str, document_id: str = "d") -> dict:
    return build_rag_document(
        FakeDoc(items),
        document_id=document_id,
        source_pdf=f"{document_id}{file_type}",
        file_type=file_type,
    )


def test_level_rides_through_to_elements():
    rag = build(
        [
            (Item(DocItemLabel.TITLE, "Top"), 1),
            (Item(DocItemLabel.TEXT, "body"), 1),
        ],
        file_type=".docx",
    )
    assert [e["type"] for e in rag["elements"]] == ["heading", "paragraph"]
    assert [e.get("level") for e in rag["elements"]] == [1, 1]


def test_source_label_rides_through_for_the_section_rule():
    rag = build([(Item(DocItemLabel.SECTION_HEADER, "H"), 1)], file_type=".docx")
    assert rag["elements"][0]["source_label"] == "section_header"


def test_cross_page_merge_is_pdf_only():
    # Two lowercase-continuation paragraphs on adjacent pages: PDF merges them
    # (a sentence broken across a page column); a non-PDF must not, because its
    # page numbers are sections, and adjacent sections are different content.
    def stream() -> list[tuple]:
        return [
            (Item(DocItemLabel.TEXT, "first half of the sentence", pages=[3]), 1),
            (Item(DocItemLabel.TEXT, "and it continues here", pages=[4]), 1),
        ]

    pdf = build(stream(), file_type=".pdf")
    assert len(pdf["elements"]) == 1, "PDF keeps its cross-page merge"

    docx = build(stream(), file_type=".docx")
    assert len(docx["elements"]) == 2, "non-PDF must not run the merge"


def test_legacy_call_shape_still_merges():
    # A caller that does not pass file_type keeps the pre-D61 behaviour (PDF).
    rag = build_rag_document(
        FakeDoc([(Item(DocItemLabel.TEXT, "abc", pages=[1]), 1),
                 (Item(DocItemLabel.TEXT, "def", pages=[2]), 1)]),
        document_id="d",
        source_pdf="d.pdf",
    )
    assert rag.get("file_type") == ".pdf"


def test_sheet_group_becomes_a_heading_anchor():
    frame = pd.DataFrame({"region": ["north"], "total": [10]})
    rag = build(
        [
            (Item(GroupLabel.SHEET, name="Sales-Q3"), 1),
            (Item(DocItemLabel.TABLE, pages=[1], frame=frame), 2),
        ],
        file_type=".xlsx",
    )
    types = [e["type"] for e in rag["elements"]]
    assert types == ["heading", "table"]
    sheet_heading, table = rag["elements"]
    assert sheet_heading["text"] == "Sales-Q3"
    assert sheet_heading["source_label"] == "sheet"
    # the sheet name must be citable context for the rows under it
    assert table["heading_path"] == ["Sales-Q3"]
    assert "Sales-Q3" in table["search_text"]


def test_unnamed_sheet_group_is_skipped():
    rag = build(
        [
            (Item(GroupLabel.SHEET, name=""), 1),
            (Item(DocItemLabel.TEXT, "body"), 1),
        ],
        file_type=".xlsx",
    )
    assert [e["type"] for e in rag["elements"]] == ["paragraph"]


def test_empty_table_is_dropped_and_counted():
    rag = build(
        [
            # An XLSX region without table structure comes back as an empty grid
            # (measured) -- pure noise, and it used to become a "shape: 0 rows"
            # chunk that could be cited.
            (Item(DocItemLabel.TABLE, pages=[2], frame=pd.DataFrame()), 2),
            (Item(DocItemLabel.TEXT, "still here"), 1),
        ],
        file_type=".xlsx",
    )
    assert [e["type"] for e in rag["elements"]] == ["paragraph"]
    assert rag["stats"]["dropped_empty_tables"] == 1


def test_table_with_content_survives():
    frame = pd.DataFrame({"region": ["north"], "total": [10]})
    rag = build([(Item(DocItemLabel.TABLE, pages=[1], frame=frame), 2)], file_type=".xlsx")
    assert [e["type"] for e in rag["elements"]] == ["table"]
    assert rag["stats"]["dropped_empty_tables"] == 0


def test_header_only_table_survives_and_is_not_counted_as_empty():
    # D61 follow-up (measured): a worksheet that holds a single row comes back
    # from Docling with that row as the *header* and zero data rows.  Dropping it
    # as "empty" deleted real data -- the whole second sheet disappeared from the
    # index.  Only a table with neither columns nor rows is noise.
    frame = pd.DataFrame(columns=["memo", "carried"])
    rag = build([(Item(DocItemLabel.TABLE, pages=[2], frame=frame), 2)], file_type=".xlsx")
    assert [e["type"] for e in rag["elements"]] == ["table"]
    assert rag["stats"]["dropped_empty_tables"] == 0
    assert rag["elements"][0]["payload"]["grid"]["headers"] == ["memo", "carried"]


def test_unparseable_table_is_not_mistaken_for_an_empty_one():
    # A table whose dataframe raised keeps its error payload and must survive --
    # "could not parse" is reported, whereas "no content" is dropped.
    rag = build([(Item(DocItemLabel.TABLE, pages=[1], frame=None), 2)], file_type=".xlsx")
    kinds = [e["type"] for e in rag["elements"]]
    assert "table" in kinds
    assert rag["stats"]["dropped_empty_tables"] == 0


def test_body_layer_filter_still_applies():
    rag = build([(Item(DocItemLabel.PAGE_FOOTER, "footer text", pages=[1]), 1)], file_type=".pdf")
    assert rag["elements"] == []
