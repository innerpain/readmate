"""Multi-format page numbering for the ingestion chain (D61).

Docling gives a real page number for PDF (text provenance), PPTX (slide number)
and XLSX (sheet number) -- and **none at all** for DOCX / Markdown / HTML / CSV,
not even when the source carries explicit page breaks.  Measured: ``w:br
type="page"``, ``<hr>``, ``<section>`` and CSS page-break divs all come back with
empty ``prov`` (see ``tmp/fmt-probe/probe_page_units.py``).

So a page number for those formats is derived from the source file itself:

* **DOCX with explicit page breaks** -> the real page, counted from the source
  XML.  The source body order and Docling's non-container item order were
  measured to line up one-to-one, tables included (``probe_page_map2.py``), so the
  numbering can be transferred onto the Docling stream by text alignment.
* **everything else** -> the top-level heading section (``§N``).  That is a real,
  checkable location ("open the file, go to the second heading"), never a
  fabricated page number.  A file with no headings is one section (``§1``).

Assignment deliberately happens **after** ``build_rag_document`` and **before**
``refine_rag_document``: the export stage's ``merge_cross_page_fragments`` joins
paragraphs whose page numbers differ by one, and it must only ever see real PDF
provenance (numbering first would merge paragraphs from two different sections
into one passage; the merge is now gated to PDFs).  Between those two stages the
elements still carry ``level``/``source_label`` -- the refiner carries
``page``/``page_end`` onward but drops everything else.

Pure apart from reading the source file, so it is unit-testable without Docling.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

# The page number's meaning.  ``page`` is a physical page, ``slide``/``sheet`` are
# the container position in a deck/workbook, ``section`` is a top-level heading
# section.  The UI labels them differently (p3 / 幻灯片 3 / 工作表 3 / §2) -- a
# section number must never be shown as "p2", because the user would go looking
# for page 2 of a Word file that has no such page.
PAGE_KINDS = ("page", "slide", "sheet", "section")

#: Formats whose page number is a physical page or a container position Docling
#: reports itself -- nothing to compute, only the kind has to be recorded.
PROVENANCE_KINDS = {".pdf": "page", ".pptx": "slide", ".xlsx": "sheet"}

#: Text formats that need an encoding check before Docling reads them: the CSV
#: backend has no encoding option and fails outright on a GBK file (measured).
TEXT_FORMATS = {".csv", ".md", ".html", ".htm"}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class PageMapError(Exception):
    """The source file cannot be given a trustworthy page number."""


# --------------------------------------------------------------------------- DOCX


def _docx_units(path: Path) -> list[tuple[str, int, str]]:
    """``[(text, page, tag)]`` for every direct child of ``w:body``, in order.

    ``page`` counts explicit page breaks, and the two forms have different
    semantics (both measured against Word's own layout):

    * ``<w:pageBreakBefore/>`` in paragraph properties -- *this* paragraph
      starts a new page, so the counter advances **before** the row is recorded;
    * ``<w:br w:type="page"/>`` inside a run -- the break sits *within* the
      paragraph, so the paragraph keeps the page it started on and the counter
      advances **after** the row (the common shape: a break-only paragraph
      trailing a heading).

    A paragraph holding nothing but a break yields an empty ``text`` entry -- it
    is kept, because it is what advances the counter.
    """

    with zipfile.ZipFile(path) as archive:
        document = archive.read("word/document.xml")
    root = ElementTree.fromstring(document)
    body = root.find(f"{_W}body")
    if body is None:
        raise PageMapError("docx has no body")

    units: list[tuple[str, int, str]] = []
    page = 1
    for child in body:
        tag = child.tag.split("}")[-1]
        text = "".join(node.text or "" for node in child.iter(f"{_W}t")).strip()
        breaks = [node.get(f"{_W}type") or "line" for node in child.iter(f"{_W}br")]
        break_before = child.find(f".//{_W}pageBreakBefore") is not None
        if break_before:
            page += 1
        units.append((text, page, tag))
        if "page" in breaks:
            page += 1
    return units


def _norm(text: str) -> str:
    return " ".join((text or "").split()).strip()


def assign_docx_pages(elements: list[dict], units: list[tuple[str, int, str]]) -> float:
    """Copy the source page numbers onto ``elements``; return the match ratio.

    Both sides are walked in reading order.  ``units`` is first reduced to the
    text-bearing paragraphs: a break-only paragraph produces no Docling text item
    (empty non-structural rows are dropped in rag_export), yet it is still what
    advanced the page counter -- keeping those rows would mis-align the walk by
    one item per page break (measured on the real stream shape).  Tables never
    text-match the source grid, so they inherit the page of the preceding item.

    The ratio is the caller's guard: below :data:`DOCX_ALIGN_MIN_RATIO` the
    alignment is not trustworthy and the caller falls back to heading sections
    instead of labelling passages with pages that may be off by one.
    """

    wanted = [(_norm(text), page) for text, page, _ in units if _norm(text)]
    cursor = 0
    matched = 0
    considered = 0
    current = wanted[0][1] if wanted else 1

    for element in elements:
        text = _norm(str(element.get("text") or ""))
        if not text:
            element["page"] = current
            element["page_end"] = current
            continue
        considered += 1
        # Look ahead from the cursor for this element's source paragraph.  The
        # search never runs backwards (a repeated sentence must not be matched to
        # an earlier page) and an unmatched row must not consume the cursor: a
        # table sits between two paragraphs and must inherit, not swallow them.
        found = next(
            (index for index in range(cursor, len(wanted)) if wanted[index][0] == text),
            None,
        )
        if found is not None:
            current = wanted[found][1]
            cursor = found + 1
            matched += 1
        element["page"] = current
        element["page_end"] = current

    return (matched / considered) if considered else 0.0


#: Below this share of matched items the DOCX page transfer is treated as
#: unreliable and heading sections are used instead (honest, coarse, never wrong).
DOCX_ALIGN_MIN_RATIO = 0.5


# ------------------------------------------------------------------ heading sections


def top_heading_predicate(elements: list[dict]):
    """Build the "is this a top-level heading?" predicate for one document.

    The label/level semantics are *not* uniform across formats (measured):
    Markdown puts both ``#`` and ``##`` at ``level=1`` and leaves both with an
    empty ``heading_path`` -- only ``source_label`` separates them (``title`` vs
    ``section_header``).  DOCX labels every heading ``section_header`` and
    separates them only by ``level`` (2 vs 3).  So:

    * a document that has any ``title`` heading -> ``title`` marks the top;
    * otherwise -> the shallowest ``level`` among its headings marks the top;
    * otherwise (no level info at all) -> every heading is top (coarse but honest).

    Requires ``level`` and ``source_label`` on the elements, which is why page
    assignment runs on the rag elements (post-``build_rag_document``,
    pre-refine): the refiner carries ``page``/``page_end`` onward but drops
    ``level``.

    D62: the heading-path prefix chain reuses this same predicate for the
    non-PDF formats, so ``§N`` and the retrieval prefix can no longer disagree.
    """

    headings = [e for e in elements if e.get("type") == "heading"]
    has_title = any(e.get("source_label") == "title" for e in headings)
    if has_title:

        def is_top(element: dict) -> bool:
            return element.get("type") == "heading" and element.get("source_label") == "title"

        return is_top

    ranks = [e.get("level") for e in headings if isinstance(e.get("level"), int)]
    if not ranks:

        def is_top(element: dict) -> bool:  # pragma: no cover - trivial fallback
            return element.get("type") == "heading"

        return is_top

    shallowest = min(ranks)

    def is_top(element: dict) -> bool:
        return element.get("type") == "heading" and element.get("level") == shallowest

    return is_top


def assign_section_pages(elements: list[dict]) -> None:
    """Number elements by top-level heading section (``§1``, ``§2``, ...).

    Content before the first top-level heading belongs to section 1, so a
    document whose first heading is at the very top still starts at ``§1``.  A
    document with no headings at all is a single section -- which is the truth
    for a CSV, not a placeholder.
    """

    is_top = top_heading_predicate(elements)
    section = 1
    seen_top = False
    for element in elements:
        if is_top(element):
            if seen_top:
                section += 1
            seen_top = True
        element["page"] = section
        element["page_end"] = section


# ------------------------------------------------------------------------ encoding


_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)


def _family(encoding: str) -> str:
    """Coerce an encoding name to the family two detectors can be compared by."""

    name = encoding.lower().replace("-", "").replace("_", "").replace(" ", "")
    table = {
        "gb18030": "cjk-gb", "gbk": "cjk-gb", "gb2312": "cjk-gb",
        "big5": "cjk-big5", "big5hkscs": "cjk-big5",
        "cp949": "cjk-kr", "euckr": "cjk-kr",
        "shiftjis": "cjk-jp", "eucjp": "cjk-jp", "eucjis2004": "cjk-jp", "cp51932": "cjk-jp",
        "utf8": "unicode", "utf8sig": "unicode",
        "utf16": "utf16", "utf16le": "utf16", "utf16be": "utf16",
        "ascii": "ascii",
    }
    for prefix, family in table.items():
        if name.startswith(prefix):
            return family
    # cp125x / iso-8859-x / windows-125x are *not* folded together: their byte
    # tables differ.  Equivalence, when it holds, is proved by decoding instead.
    return name


def sniff_encoding(path: Path) -> str:
    """Best-effort encoding for a text file; raises when nothing is confident.

    A wrong guess is worse than a failure here: decoding a GBK CSV as big5 yields
    valid *different* ideographs -- clean output, silently wrong content, which no
    downstream stage can detect.  Measured detector behaviour (charset_normalizer
    x chardet, in the project image):

    * UTF-8 / BOM'd / ASCII files are decided outright by the fast path;
    * a Chinese CSV of any realistic size is reported as GB18030 by **both**;
    * a ~17-byte GBK fragment splits the detectors (e.g. cp949 vs GB18030) --
      genuinely ambiguous, so it is refused rather than guessed;
    * single-byte Western European text also splits them (cp1250 vs
      Windows-1252), and those two decode the *same byte to different letters*
      (0xE8 -> č vs è), so decoded-text equivalence is not a usable agreement
      signal here -- the file is refused and the user re-saves as UTF-8;
    * UTF-16 is a real format (Excel's "Unicode Text" export), not a misread.

    So: accept when the two detectors agree -- by family or byte-for-byte -- or
    when chardet is confident (>= 0.7) and the decode is clean.  Everything else
    raises :class:`PageMapError`, which the ingestion reports as
    ``unsupported_or_unreadable``.
    """

    raw = path.read_bytes()
    for bom, encoding in _BOMS:
        if raw.startswith(bom):
            return encoding
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    sample = raw[:200_000]
    candidates: list[str] = []
    try:  # charset_normalizer ships with the image (measured)
        from charset_normalizer import from_bytes

        best = from_bytes(sample).best()
        if best is not None and best.encoding:
            candidates.append(str(best.encoding))
    except Exception:  # pragma: no cover - optional dependency
        pass
    confident: str | None = None
    try:
        import chardet

        detected = chardet.detect(sample)
        encoding = detected.get("encoding")
        if encoding:
            candidates.append(str(encoding))
            if float(detected.get("confidence") or 0) >= 0.7:
                confident = str(encoding)
    except Exception:  # pragma: no cover - optional dependency
        pass

    def decodes(encoding: str) -> str | None:
        """The sample decoded by ``encoding``, or None when it cannot be trusted."""

        try:
            text = sample.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            return None
        if "\ufffd" in text:
            return None
        return text

    # Two detectors, one answer: same family, or byte-for-byte the same text.
    decoded: dict[str, str | None] = {}
    for encoding in candidates:
        decoded[encoding] = decodes(encoding)
    for i, first in enumerate(candidates):
        if decoded[first] is None:
            continue
        for second in candidates[i + 1:]:
            if decoded[second] is not None and (
                _family(first) == _family(second) or decoded[first] == decoded[second]
            ):
                return first
        # A single detector that agrees with itself is still a guess; only a
        # confident chardet answer passes alone.
    if confident and decodes(confident) is not None:
        return confident
    raise PageMapError("encoding_undetermined")


def transcode_to_utf8(path: Path, dest_dir: Path) -> tuple[Path, str | None]:
    """Return ``(path to feed Docling, original encoding or None)``.

    ``None`` means the file already is UTF-8 and the original can be used as is.
    A converted copy lands in ``dest_dir`` and is the caller's to delete -- the
    uploaded original is never modified.
    """

    encoding = sniff_encoding(path)
    if encoding in {"utf-8", "utf-8-sig"}:
        return path, None
    text = path.read_text(encoding=encoding)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{path.stem}.utf8{path.suffix}"
    target.write_text(text, encoding="utf-8")
    return target, encoding


# --------------------------------------------------------------------------- entry


def fill_page_gaps(elements: list[dict]) -> None:
    """Give page-less elements the page of the *next* numbered element.

    D61: the XLSX sheet-heading anchor is injected from a group node that carries
    no provenance page, while the table right after it does.  "Belongs to the
    upcoming page" is the only reading that keeps a sheet title on its own sheet
    (a plain forward-fill would pin sheet 2's title to sheet 1).  A gap with no
    following page falls back to the previous one; an all-None stream is left
    alone -- the section rule numbers that case elsewhere.
    """

    known = [element.get("page") for element in elements]

    # suffix[i] = nearest known page at or after i; prefix[i] = nearest before i.
    suffix: list[int | None] = [None] * len(elements)
    last: int | None = None
    for index in range(len(elements) - 1, -1, -1):
        if isinstance(known[index], int):
            last = known[index]
        suffix[index] = last
    prefix: list[int | None] = [None] * len(elements)
    last = None
    for index, page in enumerate(known):
        if isinstance(page, int):
            last = page
        prefix[index] = last

    for element, before, after in zip(elements, prefix, suffix):
        if isinstance(element.get("page"), int):
            continue
        chosen = after if after is not None else before
        if chosen is None:
            continue
        element["page"] = chosen
        if element.get("page_end") is None:
            element["page_end"] = chosen


def assign_pages(
    elements: list[dict],
    *,
    file_type: str,
    source_path: Path | str | None = None,
) -> str:
    """Number ``elements`` in place and return the page kind that was applied.

    ``file_type`` is a lower-case extension including the dot.  PDF keeps the
    provenance page untouched -- nothing here may move a PDF number, existing
    artifacts depend on that.  PPTX/XLSX keep their provenance pages too, but
    page-less anchors (the injected sheet headings) get gap-filled.  DOCX gets
    real pages when its explicit breaks align, and everything else -- including a
    DOCX whose alignment was not trustworthy -- gets heading sections.
    """

    extension = (file_type or "").lower()
    if not extension.startswith("."):
        extension = f".{extension}"

    provenance = PROVENANCE_KINDS.get(extension)
    if extension == ".pdf":
        return "page"
    if provenance in {"slide", "sheet"}:
        fill_page_gaps(elements)
        return provenance

    if extension == ".docx" and source_path is not None:
        try:
            units = _docx_units(Path(source_path))
        except (OSError, KeyError, ElementTree.ParseError, PageMapError, zipfile.BadZipFile):
            # A corrupt/renamed container must degrade to sections, never take
            # the ingestion down: the section rule always produces an answer.
            units = []
        pages_present = len({page for _, page, _ in units}) > 1
        if pages_present:
            ratio = assign_docx_pages(elements, units)
            if ratio >= DOCX_ALIGN_MIN_RATIO:
                return "page"

    assign_section_pages(elements)
    return "section"
