"""D61: page_map unit tests -- DOCX page-break transfer, §section numbering,
encoding sniff/transcode.

The DOCX fixtures are built by hand (a minimal OOXML zip) rather than with
python-docx: ``_docx_units`` reads ``word/document.xml`` directly, so the test
exercises *our* parser, not a third-party one -- and it runs anywhere the repo's
plain stdlib python runs.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.ingestion.page_map import (
    DOCX_ALIGN_MIN_RATIO,
    PageMapError,
    _docx_units,
    assign_docx_pages,
    assign_pages,
    fill_page_gaps,
    sniff_encoding,
    transcode_to_utf8,
)

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(path: Path, paragraphs: list[str], *, page_breaks: int = 0) -> Path:
    """Write a minimal DOCX whose first ``page_breaks`` paragraphs end in a
    page break (``<w:br w:type="page"/>``)."""

    body = []
    for index, text in enumerate(paragraphs):
        run = f'<w:r><w:t xml:space="preserve">{text}</w:t>'
        if index < page_breaks:
            run += '<w:br w:type="page"/>'
        run += "</w:r>"
        body.append(f"<w:p>{run}</w:p>")
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W}"><w:body>{"".join(body)}'
        "<w:sectPr/></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
    return path


def elements(texts: list[str], pages: list[int | None] | None = None) -> list[dict]:
    pages = pages or [None] * len(texts)
    return [
        {"type": "paragraph", "text": text, "page": page, "page_end": page}
        for text, page in zip(texts, pages)
    ]


class TestDocxUnits:
    def test_counts_explicit_page_breaks(self, tmp_path: Path) -> None:
        path = make_docx(tmp_path / "b.docx", ["one", "two", "three"], page_breaks=2)
        units = _docx_units(path)
        # paragraphs: one(page1) two(page2) three(page3); sectPr is not a w:p
        assert [(text, page) for text, page, _ in units if text] == [
            ("one", 1),
            ("two", 2),
            ("three", 3),
        ]

    def test_no_breaks_is_one_page(self, tmp_path: Path) -> None:
        path = make_docx(tmp_path / "flat.docx", ["a", "b"])
        assert {page for _, page, _ in _docx_units(path)} == {1}

    def test_page_break_before_starts_this_paragraph_on_the_new_page(self, tmp_path: Path) -> None:
        # <w:pageBreakBefore/> means THIS paragraph opens the new page, unlike a
        # trailing <w:br w:type="page"/>, which leaves the paragraph behind.
        path = tmp_path / "bb.docx"
        document = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<w:document xmlns:w="{_W}"><w:body>'
            "<w:p><w:r><w:t>first</w:t></w:r></w:p>"
            "<w:p><w:pPr><w:pageBreakBefore/></w:pPr><w:r><w:t>second</w:t></w:r></w:p>"
            "<w:sectPr/></w:body></w:document>"
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", document)
        assert [(t, page) for t, page, _ in _docx_units(path) if t] == [
            ("first", 1),
            ("second", 2),
        ]

    def test_trailing_break_keeps_its_paragraph_on_the_earlier_page(self, tmp_path: Path) -> None:
        # A break-only paragraph trailing a heading advances the counter AFTER
        # itself, so "one" is page 1 and "two" is page 2 -- the common Word shape
        # where the break is its own paragraph.
        path = make_docx(tmp_path / "trail.docx", ["one", "two"], page_breaks=1)
        assert [(t, page) for t, page, _ in _docx_units(path) if t] == [("one", 1), ("two", 2)]


class TestDocxPageTransfer:
    def test_pages_transfer_in_reading_order(self, tmp_path: Path) -> None:
        path = make_docx(tmp_path / "b.docx", ["one", "two", "three"], page_breaks=2)
        els = elements(["one", "two", "three"])
        ratio = assign_docx_pages(els, _docx_units(path))
        assert ratio >= DOCX_ALIGN_MIN_RATIO
        assert [e["page"] for e in els] == [1, 2, 3]

    def test_unmatched_stream_falls_back_to_sections(self, tmp_path: Path) -> None:
        path = make_docx(tmp_path / "b.docx", ["alpha", "beta"], page_breaks=1)
        els = elements(["totally", "different", "text", "here"])
        assert assign_docx_pages(els, _docx_units(path)) < DOCX_ALIGN_MIN_RATIO
        assert assign_pages(els, file_type=".docx", source_path=path) == "section"

    def test_flat_docx_has_no_real_pages(self, tmp_path: Path) -> None:
        path = make_docx(tmp_path / "flat.docx", ["a", "b", "c"])
        els = elements(["a", "b", "c"])
        assert assign_pages(els, file_type=".docx", source_path=path) == "section"
        assert [e["page"] for e in els] == [1, 1, 1]

    def test_table_inherits_the_surrounding_page(self, tmp_path: Path) -> None:
        # A table's Docling text is a rendered grid, never the source paragraph
        # text -- it must inherit its neighbours' page, not reset the counter.
        path = make_docx(tmp_path / "tb.docx", ["before table", "after table"], page_breaks=1)
        els = [
            {"type": "paragraph", "text": "before table", "page": None, "page_end": None},
            {"type": "table", "text": "| x |  |", "page": None, "page_end": None},
            {"type": "paragraph", "text": "after table", "page": None, "page_end": None},
        ]
        assert assign_pages(els, file_type=".docx", source_path=path) == "page"
        assert [e["page"] for e in els] == [1, 1, 2]

    def test_unreadable_docx_falls_back_instead_of_raising(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.docx"
        path.write_bytes(b"not a zip at all")
        els = elements(["a", "b"])
        assert assign_pages(els, file_type=".docx", source_path=path) == "section"


class TestSectionNumbering:
    def test_title_label_marks_the_top_in_a_markdown_like_stream(self) -> None:
        els = [
            {"type": "heading", "source_label": "title", "level": 1, "text": "A"},
            {"type": "paragraph", "source_label": "text", "level": 1, "text": "a"},
            {"type": "heading", "source_label": "section_header", "level": 1, "text": "Sub"},
            {"type": "paragraph", "source_label": "text", "level": 1, "text": "s"},
            {"type": "heading", "source_label": "title", "level": 1, "text": "B"},
            {"type": "paragraph", "source_label": "text", "level": 1, "text": "b"},
        ]
        assert assign_pages(els, file_type=".md") == "section"
        assert [e["page"] for e in els] == [1, 1, 1, 1, 2, 2]

    def test_shallowest_level_marks_the_top_without_titles(self) -> None:
        els = [
            {"type": "heading", "source_label": "section_header", "level": 2, "text": "H1"},
            {"type": "paragraph", "source_label": "text", "text": "p"},
            {"type": "heading", "source_label": "section_header", "level": 3, "text": "H2"},
            {"type": "paragraph", "source_label": "text", "text": "q"},
            {"type": "heading", "source_label": "section_header", "level": 2, "text": "H1b"},
        ]
        assert assign_pages(els, file_type=".html") == "section"
        assert [e["page"] for e in els] == [1, 1, 1, 1, 2]

    def test_content_before_the_first_heading_is_section_one(self) -> None:
        els = [
            {"type": "paragraph", "source_label": "text", "text": "lead-in"},
            {"type": "heading", "source_label": "title", "level": 1, "text": "A"},
            {"type": "paragraph", "source_label": "text", "text": "a"},
        ]
        assign_pages(els, file_type=".md")
        assert [e["page"] for e in els] == [1, 1, 1]

    def test_no_headings_is_one_section(self) -> None:
        els = [{"type": "paragraph", "text": "a"}, {"type": "table", "text": "b"}]
        assert assign_pages(els, file_type=".csv") == "section"
        assert [e["page"] for e in els] == [1, 1]

    def test_sheet_heading_anchor_does_not_start_a_new_section(self) -> None:
        # A workbook's sheet anchors are headings; they must not double-count as
        # sections (the document is numbered by sheet, not by §).
        els = [
            {"type": "heading", "source_label": "sheet", "level": 1, "text": "Sales", "page": 1},
            {"type": "table", "text": "grid", "page": 1},
        ]
        assert assign_pages(els, file_type=".xlsx") == "sheet"


class TestProvenanceKinds:
    def test_pdf_numbers_are_never_touched(self) -> None:
        pdf = elements(["x"], [7])
        assert assign_pages(pdf, file_type=".pdf") == "page"
        assert pdf[0]["page"] == 7

    def test_pptx_keeps_slide_numbers(self) -> None:
        deck = [
            {"type": "heading", "source_label": "title", "text": "S1", "page": 1, "page_end": 1},
            {"type": "list_item", "text": "n", "page": 1, "page_end": 1},
            {"type": "heading", "source_label": "title", "text": "S2", "page": 2, "page_end": 2},
        ]
        assert assign_pages(deck, file_type=".pptx") == "slide"
        assert [e["page"] for e in deck] == [1, 1, 2]

    def test_gap_fill_prefers_the_next_page(self) -> None:
        els = [
            {"type": "heading", "text": "Sales", "page": None, "page_end": None},
            {"type": "table", "text": "grid", "page": 1, "page_end": 1},
            {"type": "heading", "text": "Notes", "page": None, "page_end": None},
            {"type": "table", "text": "grid2", "page": 2, "page_end": 2},
        ]
        assert assign_pages(els, file_type=".xlsx") == "sheet"
        assert [e["page"] for e in els] == [1, 1, 2, 2]

    def test_all_none_stream_is_left_alone(self) -> None:
        els = [{"type": "heading", "text": "A", "page": None, "page_end": None}]
        fill_page_gaps(els)
        assert els[0]["page"] is None

    def test_extension_without_dot_is_accepted(self) -> None:
        els = elements(["a"])
        assert assign_pages(els, file_type="md") == "section"


class TestEncoding:
    def test_utf8_passes_through(self, tmp_path: Path) -> None:
        p = tmp_path / "a.csv"
        p.write_text("名称,数量\n苹果,3\n", encoding="utf-8")
        assert sniff_encoding(p) == "utf-8"
        feed, original = transcode_to_utf8(p, tmp_path / "out")
        assert feed == p and original is None

    def test_bom_is_recognised_and_left_alone(self, tmp_path: Path) -> None:
        p = tmp_path / "b.csv"
        p.write_bytes(b"\xef\xbb\xbfcol1,col2\nx,1\n")
        assert sniff_encoding(p) == "utf-8-sig"
        feed, original = transcode_to_utf8(p, tmp_path / "out")
        assert feed == p and original is None

    def test_gbk_is_transcoded_and_the_original_kept(self, tmp_path: Path) -> None:
        # Realistic size on purpose: charset_normalizer cannot tell a 20-byte GBK
        # sample from big5 (measured), and a real CSV is never 20 bytes.
        payload = "名称,数量\n" + "".join(
            f"水果{i:04d},{i}\n" for i in range(50)
        ) + "苹果,3\n香蕉,5\n"
        p = tmp_path / "c.csv"
        p.write_bytes(payload.encode("gbk"))
        encoding = sniff_encoding(p)
        assert encoding.lower().replace("-", "") in {"gbk", "gb18030", "gb2312"}
        feed, original = transcode_to_utf8(p, tmp_path / "out")
        assert feed != p
        text = feed.read_text(encoding="utf-8")
        assert "苹果" in text and "香蕉" in text and "\ufffd" not in text
        assert original == encoding
        assert p.read_bytes() == payload.encode("gbk")  # original untouched

    def test_utf16_is_a_real_format_not_a_misread(self, tmp_path: Path) -> None:
        # Excel's "Unicode Text" export is UTF-16LE with a BOM.  A detector that
        # sees the BOM is right, so this must be transcoded, not refused.
        p = tmp_path / "u16.csv"
        p.write_bytes(b"\xff\xfe" + "名称,数量\n苹果,3\n".encode("utf-16-le"))
        assert sniff_encoding(p) == "utf-16-le"
        feed, original = transcode_to_utf8(p, tmp_path / "out")
        assert feed != p and "苹果" in feed.read_text(encoding="utf-8")

    def test_unrecognisable_bytes_are_refused(self, tmp_path: Path) -> None:
        p = tmp_path / "d.csv"
        p.write_bytes(bytes(range(256)) * 8)  # no BOM, no consistent text
        with pytest.raises(PageMapError):
            sniff_encoding(p)

    def test_two_detectors_disagreeing_is_refused_not_guessed(self, tmp_path: Path) -> None:
        # A 17-byte GBK fragment: charset_normalizer says cp949, chardet says
        # GB18030 (measured) -- a real ambiguity where guessing wrong yields
        # clean-looking text with different ideographs.  Refuse it.
        p = tmp_path / "tiny.csv"
        p.write_bytes("名称,数量\n苹果,3\n".encode("gbk"))
        with pytest.raises(PageMapError):
            sniff_encoding(p)

    def test_ambiguous_single_byte_text_is_refused_not_guessed(self, tmp_path: Path) -> None:
        # Western-European single-byte text is genuinely undecidable from a small
        # sample: charset_normalizer says cp1250, chardet says Windows-1252, and
        # the two decode the same byte differently (0xE8 -> č vs è -- measured in
        # the image).  Guessing would bake the wrong letter into the corpus with
        # no detectable symptom, so this is refused and surfaces as
        # ``unsupported_or_unreadable``; re-saving as UTF-8 is the fix.
        p = tmp_path / "e.csv"
        p.write_bytes("café,naïve\nCrème brûlée,1\n".encode("latin1") * 40)
        with pytest.raises(PageMapError):
            sniff_encoding(p)

    def test_utf8_is_always_accepted_for_the_same_content(self, tmp_path: Path) -> None:
        # The refusal above is about ambiguity, not about accented text: the same
        # content in UTF-8 (what Excel writes as "CSV UTF-8") ingests normally.
        p = tmp_path / "f.csv"
        p.write_text("café,naïve\nCrème brûlée,1\n", encoding="utf-8")
        assert sniff_encoding(p) == "utf-8"
        feed, original = transcode_to_utf8(p, tmp_path / "out")
        assert feed == p and original is None
