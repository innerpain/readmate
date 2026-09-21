"""Export a DoclingDocument into an ordered rag_document_v1 element stream."""

from __future__ import annotations

import re
from pathlib import Path

from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument

FURNITURE = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}


def collapse_spaced_letters(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\b(?:[A-Za-z] ){2,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), text)
    text = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", text)
    return text


def demote_pre_figure_titles(seq: list[dict]) -> list[dict]:
    for i, row in enumerate(seq[:-1]):
        if row["type"] != "heading":
            continue
        text = row.get("text") or ""
        if re.match(r"^\d+(?:\.\d+)*\b", text) or len(text) >= 80:
            continue
        if seq[i + 1]["type"] in {"figure", "caption"}:
            row["type"] = "figure_title_candidate"
            row["search_text"] = text
    return seq


def merge_cross_page_fragments(seq: list[dict]) -> list[dict]:
    skip_between = {"figure", "caption", "figure_title_candidate", "footnote"}
    out: list[dict] = []
    for row in seq:
        joined = False
        if row["type"] == "paragraph" and row.get("page") is not None:
            cur = (row.get("text") or "").lstrip()
            if cur and cur[0].islower():
                for j in range(len(out) - 1, -1, -1):
                    prev = out[j]
                    if prev["type"] in skip_between:
                        continue
                    if prev["type"] == "heading" and not re.match(
                        r"^\d+(?:\.\d+)*\b", prev.get("text") or ""
                    ):
                        continue
                    if (
                        prev["type"] == "paragraph"
                        and prev.get("page") is not None
                        and row["page"] == prev["page"] + 1
                    ):
                        prev_t = (prev.get("text") or "").rstrip()
                        if prev_t.endswith("-") and cur[:1].islower():
                            merged = prev_t[:-1] + cur
                        elif prev_t and not re.search(r"[.!?\"”']$", prev_t) and not prev_t.endswith(":"):
                            merged = prev_t + " " + cur
                        else:
                            break
                        prev["text"] = merged
                        prev["search_text"] = merged
                        prev["page_end"] = row["page"]
                        joined = True
                    break
        if not joined:
            out.append(row)
    return out


def resolve_captions(doc: DoclingDocument, item) -> list[str]:
    caps: list[str] = []
    try:
        ct = item.caption_text(doc=doc)
        if ct and str(ct).strip():
            return [collapse_spaced_letters(str(ct).strip())]
    except Exception:
        pass
    for cap in getattr(item, "captions", None) or []:
        if hasattr(cap, "text") and cap.text:
            caps.append(collapse_spaced_letters(cap.text))
            continue
        cref = getattr(cap, "cref", None) or (cap.get("$ref") if isinstance(cap, dict) else None)
        if isinstance(cref, str) and cref.startswith("#/texts/"):
            idx = int(cref.rsplit("/", 1)[-1])
            t = doc.texts[idx].text if idx < len(doc.texts) else ""
            if t:
                caps.append(collapse_spaced_letters(t))
    return caps


def table_payload(doc: DoclingDocument, table) -> tuple[str, dict]:
    caps = resolve_captions(doc, table)
    caption = caps[0] if caps else ""
    try:
        df = table.export_to_dataframe(doc=doc)
        headers = [str(c) for c in df.columns.tolist()]
        rows = []
        for _, series in df.iterrows():
            cells = [str(v) for v in series.tolist()]
            facts = [f"{h}: {v}" for h, v in zip(headers, cells) if v and v.lower() != "nan"]
            rows.append(" | ".join(facts) if facts else " | ".join(cells))
        grid = {
            "headers": headers,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "rows": df.astype(str).values.tolist(),
        }
        body = "\n".join(rows)
    except Exception as error:
        grid = {"error": f"{type(error).__name__}: {error}"}
        body = table.export_to_markdown(doc=doc) if hasattr(table, "export_to_markdown") else ""
        body = body or ""
    text = (caption + "\n" + body).strip() if caption else body
    return text, {
        "caption": caption,
        "captions": caps,
        "grid": grid,
        "markdown": table.export_to_markdown(doc=doc) if hasattr(table, "export_to_markdown") else None,
    }


def figure_payload(doc: DoclingDocument, picture) -> tuple[str, dict]:
    caps = resolve_captions(doc, picture)
    caption = caps[0] if caps else ""
    uri = None
    img = getattr(picture, "image", None)
    if img is not None:
        uri = str(getattr(img, "uri", None) or "")
    return caption or "[figure]", {
        "caption": caption,
        "captions": caps,
        "image_path": uri,
    }


def map_type(label) -> str | None:
    if label in FURNITURE:
        return None
    if not isinstance(label, DocItemLabel):
        return None
    mapping = {
        DocItemLabel.TITLE: "heading",
        DocItemLabel.SECTION_HEADER: "heading",
        DocItemLabel.PARAGRAPH: "paragraph",
        DocItemLabel.TEXT: "paragraph",
        DocItemLabel.CAPTION: "caption",
        DocItemLabel.FORMULA: "formula",
        DocItemLabel.TABLE: "table",
        DocItemLabel.PICTURE: "figure",
        DocItemLabel.LIST_ITEM: "list_item",
        DocItemLabel.FOOTNOTE: "footnote",
        DocItemLabel.CODE: "code",
        DocItemLabel.REFERENCE: "reference",
    }
    return mapping.get(label, "other")


def build_rag_document(
    doc: DoclingDocument,
    *,
    document_id: str,
    source_pdf: Path | str,
    source_docling_json: Path | str | None = None,
    parser_options: dict | None = None,
) -> dict:
    raw_items: list[dict] = []
    for item, level in doc.iterate_items(
        with_groups=True,
        traverse_pictures=False,
        included_content_layers={ContentLayer.BODY},
    ):
        typ = map_type(getattr(item, "label", None))
        if typ is None:
            continue
        prov = getattr(item, "prov", None) or []
        page = prov[0].page_no if prov else None
        # A3: a single Docling text item can straddle a page break and ``prov``
        # keeps every page it touches.  Recording only the first page made a
        # citation claim "p3" for a sentence that physically sits on p4, so the
        # end page is carried alongside the start page from here on.
        page_end = None
        if prov:
            last_page = getattr(prov[-1], "page_no", None)
            page_end = last_page if last_page != page else None
        bbox = None
        if prov:
            bb = prov[0].bbox
            if bb is not None:
                bbox = {
                    "l": getattr(bb, "l", None),
                    "t": getattr(bb, "t", None),
                    "r": getattr(bb, "r", None),
                    "b": getattr(bb, "b", None),
                    "coord_origin": str(getattr(bb, "coord_origin", "")),
                }
        payload: dict = {}
        if typ == "table":
            text, payload = table_payload(doc, item)
        elif typ == "figure":
            text, payload = figure_payload(doc, item)
        elif typ == "formula":
            text = collapse_spaced_letters(getattr(item, "text", "") or "")
            payload = {"latex": text}
        else:
            text = collapse_spaced_letters(getattr(item, "text", "") or "")
            payload = {}
        if not text and typ not in {"figure", "table"}:
            continue
        lab = getattr(item, "label", None)
        raw_items.append(
            {
                "type": typ,
                "level": level,
                "page": page,
                "page_end": page_end,
                "bbox": bbox,
                "text": text,
                "search_text": text,
                "payload": payload,
                "label": lab.value if hasattr(lab, "value") else str(lab),
            }
        )

    raw_items = merge_cross_page_fragments(raw_items)
    raw_items = demote_pre_figure_titles(raw_items)

    bound_caps = set()
    for it in raw_items:
        if it["type"] in {"figure", "table"}:
            for c in it.get("payload", {}).get("captions") or []:
                bound_caps.add(c.strip())
            cap = it.get("payload", {}).get("caption") or ""
            if cap:
                bound_caps.add(cap.strip())

    elements: list[dict] = []
    heading_stack: list[str] = []
    eid = 0
    for it in raw_items:
        if it["type"] == "caption" and (it.get("text") or "").strip() in bound_caps:
            continue
        if it["type"] == "heading":
            lvl = max(int(it.get("level") or 1), 1)
            heading_stack = heading_stack[: max(lvl - 1, 0)]
            heading_stack.append(it["text"])
            heading_path = heading_stack[:-1]
        else:
            heading_path = list(heading_stack)
        eid += 1
        search = it["search_text"]
        if heading_path and it["type"] in {"paragraph", "formula", "table", "figure", "list_item"}:
            prefix = " > ".join(heading_path)
            if prefix and not search.startswith(prefix):
                search = f"{prefix}\n{search}"
        elements.append(
            {
                "element_id": f"e{eid:04d}",
                "type": it["type"],
                "page": it.get("page"),
                "page_end": it.get("page_end"),
                "bbox": it.get("bbox"),
                "heading_path": heading_path,
                "text": it["text"],
                "search_text": search,
                "payload": it.get("payload") or {},
                "source_label": it.get("label"),
            }
        )

    pages = sorted({e["page"] for e in elements if e.get("page") is not None})
    return {
        "schema": "rag_document_v1",
        "document_id": document_id,
        "source_pdf": str(source_pdf),
        "source_docling_json": str(source_docling_json) if source_docling_json else None,
        "parser": {
            "name": "docling",
            "pipeline": "standard",
            "options": parser_options or {},
        },
        "stats": {
            "n_elements": len(elements),
            "n_pages": len(pages),
            "pages": pages,
            "by_type": {
                t: sum(1 for e in elements if e["type"] == t)
                for t in sorted({e["type"] for e in elements})
            },
        },
        "elements": elements,
    }
