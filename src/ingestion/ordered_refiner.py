"""Refine rag_document_v1 into ordered_document_v1 (parse-stage final artifact)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

# Same estimator the chunker uses for its pack ceilings: keeping one definition of
# "a token" inside the ingestion layer matters more than avoiding the import.
from src.ingestion.ordered_chunker import estimate_tokens, take_within_budget

NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")
APPENDIX_HEADING = re.compile(r"^(Appendix\s+[A-Z]|[A-Z])\s+(\S.*)$", re.I)
PERMISSION_RE = re.compile(r"\b(permission|license|creativecommons|grants permission)\b", re.I)
EQ_LABEL_RE = re.compile(r"(?:\(|\[)\s*(\d+[a-zA-Z]?)\s*(?:\)|\])\s*$")
MULTIHEAD_RE = re.compile(r"\bMutiHead\b")
SPACED_ATTN_RE = re.compile(r"Attention\s*\(\s*Q\s*,\s*K\s*,\s*V\s*\)")

# D13 (2026-09-20): table facts used to be cut by *count* -- ``facts[:200]`` stored in
# the structure, ``facts[:80]`` joined into ``search_text``, and ``facts[:8]`` in the
# table pack.  A count says nothing about cost (80 short facts are cheap, 80 long ones
# are not) and the cut was silent.  These budgets replace the counts, and whatever is
# left out is now reported on the element's ``user_notice``.
TABLE_FACTS_MAX_TOKENS = 400        # kept in structure["facts"] (provenance + chunker input)
TABLE_SEARCH_FACTS_MAX_TOKENS = 320  # joined into the element's search_text



def _numbered_depth(text: str) -> tuple[int, str] | None:
    m = NUMBERED_HEADING.match(text.strip())
    if not m:
        return None
    return m.group(1).count(".") + 1, text.strip()


def deepen_heading_paths(elements: list[dict[str, Any]]) -> list[list[str]]:
    stack: list[tuple[int, str]] = []
    paths: list[list[str]] = []
    for el in elements:
        text = (el.get("text") or "").strip()
        if el.get("type") == "heading":
            numbered = _numbered_depth(text)
            if numbered:
                depth, title = numbered
                while stack and stack[-1][0] >= depth:
                    stack.pop()
                stack.append((depth, title))
                paths.append([t for _, t in stack[:-1]])
            else:
                paths.append([t for _, t in stack])
                # Conservative: treat unnumbered heading as child title without popping.
                stack.append((stack[-1][0] + 1 if stack else 1, text))
        else:
            paths.append([t for _, t in stack])
    return paths


def assign_role(element: dict[str, Any], *, seen_abstract: bool, in_references: bool) -> tuple[str, str]:
    typ = element.get("type")
    text = element.get("text") or ""
    path = " ".join(element.get("heading_path") or [])
    if typ == "footnote":
        return "footnote", "skip"
    if typ == "figure_title_candidate":
        if re.match(r"^[A-Z]\s+\S", text) and "appendix" not in text.lower():
            return "noise", "drop"
        return "noise", "drop"
    if in_references and typ == "list_item":
        return "reference", "drop"
    if typ == "heading" and re.search(r"\breferences\b", text, re.I):
        return "body", "skip"
    if not seen_abstract:
        if typ == "paragraph" and PERMISSION_RE.search(text):
            return "preamble", "drop"
        if typ == "paragraph" and ("@" in text or "University" in text or "Google" in text):
            return "byline", "skip"
        if typ == "paragraph" and len(text) < 8:
            return "byline", "skip"
    if typ == "reference":
        return "reference", "drop"
    return "body", "embed"


def clean_formula(latex: str) -> tuple[str, str | None, list[str]]:
    fixes: list[str] = []
    text = latex or ""
    if MULTIHEAD_RE.search(text):
        text = MULTIHEAD_RE.sub("MultiHead", text)
        fixes.append("MutiHead->MultiHead")
    if SPACED_ATTN_RE.search(text):
        text = SPACED_ATTN_RE.sub("Attention(Q,K,V)", text)
        fixes.append("collapse_attention_args")
    # Collapse "Q , K" style spaces around commas inside simple calls.
    text2 = re.sub(r"\s*,\s*", ",", text)
    if text2 != text:
        fixes.append("collapse_comma_spaces")
        text = text2
    eq_label = None
    m = EQ_LABEL_RE.search(text)
    if m:
        eq_label = m.group(1)
        text = text[: m.start()].rstrip(" &\\")
        fixes.append(f"eq_label={eq_label}")
    # Strip trailing alignment markers common in enrichment output.
    text = re.sub(r"\s*&\s*$", "", text).strip()
    return text, eq_label, fixes


def _split_multi_values(cell: str) -> list[str]:
    cell = (cell or "").strip()
    if not cell or cell in {"-", "—", "nan", "None"}:
        return []
    if "/" in cell and re.search(r"\d", cell):
        parts = [p.strip() for p in cell.split("/")]
        return [p for p in parts if p and p != "-"]
    if re.fullmatch(r"[\d\.\s]+", cell) and cell.count(" ") >= 1:
        return [p for p in cell.split() if p]
    return [cell]


def build_table_structure(payload: dict[str, Any], caption: str) -> tuple[dict[str, Any], str, str]:
    grid = payload.get("grid") if isinstance(payload.get("grid"), dict) else {}
    headers = [str(h) for h in (grid.get("headers") or [])]
    rows = grid.get("rows") if isinstance(grid.get("rows"), list) else []
    if grid.get("error") or not rows:
        return (
            {"raw_grid": grid, "caption": caption, "markdown": payload.get("markdown")},
            "unparsed",
            "未能可靠解析该表结构，保留原始 grid/markdown。",
        )

    # Forward-fill empty first column group labels.
    filled_rows: list[list[str]] = []
    last_group = ""
    for row in rows:
        cells = ["" if str(c).lower() == "nan" else str(c) for c in row]
        if headers and headers[0] == "" and cells:
            if cells[0].strip():
                last_group = cells[0].strip()
            elif last_group:
                cells[0] = last_group
        filled_rows.append(cells)

    facts: list[str] = []
    for cells in filled_rows:
        if not any(c.strip() for c in cells):
            continue
        row_label_parts = []
        start_col = 0
        if headers and headers[0] == "" and cells:
            if cells[0].strip():
                row_label_parts.append(cells[0].strip())
            start_col = 1
            if len(cells) > 1 and cells[1].strip() and (len(headers) < 2 or headers[1].lower() == "model" or headers[1] == "Model"):
                row_label_parts.append(cells[1].strip())
                start_col = 2
        elif cells:
            row_label_parts.append(cells[0].strip())
            start_col = 1
        row_name = " | ".join(p for p in row_label_parts if p) or "row"
        for idx in range(start_col, len(cells)):
            header = headers[idx] if idx < len(headers) else f"col{idx+1}"
            if not header:
                continue
            for value in _split_multi_values(cells[idx]):
                prefix = f"[table-fact] {caption}" if caption else "[table-fact]"
                facts.append(f"{prefix} | {row_name} | {header} | {value}")

    status = "parsed" if facts else "partial"
    facts_taken, facts_dropped = take_within_budget(facts, TABLE_FACTS_MAX_TOKENS)
    structure = {
        "caption": caption,
        "headers": headers,
        "rows": filled_rows,
        "facts": facts_taken,
        # D13: how many facts the budget left out -- provenance for the UI (D8/D11)
        # and the reason the element may carry a user_notice.
        "facts_dropped": facts_dropped,
        "markdown": payload.get("markdown"),
    }
    return structure, status, ""


def refine_rag_document(
    rag: dict[str, Any],
    *,
    output_dir: Path,
    copy_figures: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    elements = list(rag.get("elements") or [])
    norm_paths = deepen_heading_paths(elements)

    seen_abstract = False
    in_references = False
    sequence: list[dict[str, Any]] = []
    quality = {
        "role_counts": {},
        "parse_status_counts": {},
        "formula_fixes": [],
        "dropped_embed_candidates": 0,
        "figure_path_fixes": 0,
        "table_unparsed": 0,
        "formula_unparsed": 0,
    }

    for idx, el in enumerate(elements):
        text = el.get("text") or ""
        if el.get("type") == "heading" and re.search(r"\babstract\b", text, re.I):
            seen_abstract = True
        if el.get("type") == "heading" and re.search(r"\breferences\b", text, re.I):
            in_references = True

        role, index_policy = assign_role(el, seen_abstract=seen_abstract, in_references=in_references)
        heading_path_norm = norm_paths[idx]
        payload = dict(el.get("payload") or {})
        parse_status = "parsed"
        structure: dict[str, Any] = {}
        user_notice = None
        search_text = el.get("search_text") or text
        atomic = el.get("type") in {"table", "formula", "figure", "code"}
        degraded = False
        fixes: list[str] = []

        if el.get("type") == "table":
            caption = payload.get("caption") or ""
            structure, parse_status, user_notice = build_table_structure(payload, caption)
            if parse_status == "unparsed":
                quality["table_unparsed"] += 1
                degraded = True
                search_text = (caption + "\n" + (payload.get("markdown") or text))[:4000]
            else:
                facts = structure.get("facts") or []
                if facts:
                    search_facts, search_dropped = take_within_budget(
                        facts, TABLE_SEARCH_FACTS_MAX_TOKENS
                    )
                    search_text = "\n".join(search_facts)
                    if search_dropped:
                        # D13: say it rather than silently cutting -- the old count cap
                        # of 80 dropped facts with no trace anywhere in the artifacts.
                        user_notice = (
                            f"表格事实超出检索预算，已按预算截取（省略 {search_dropped} 条）。"
                        )
                else:
                    search_text = text
                if heading_path_norm:
                    search_text = " > ".join(heading_path_norm) + "\n" + search_text

        elif el.get("type") == "formula":
            latex_in = payload.get("latex") or text
            latex, eq_label, fixes = clean_formula(latex_in)
            # Neighbor contexts: previous/next body paragraphs in original list.
            prev_ctx = ""
            next_ctx = ""
            for j in range(idx - 1, -1, -1):
                if elements[j].get("type") == "paragraph":
                    prev_ctx = (elements[j].get("text") or "")[:300]
                    break
            for j in range(idx + 1, len(elements)):
                if elements[j].get("type") == "paragraph":
                    next_ctx = (elements[j].get("text") or "")[:300]
                    break
            if not latex.strip():
                parse_status = "unparsed"
                user_notice = "未能解析为结构化公式/LaTeX，仅保留上下文。"
                quality["formula_unparsed"] += 1
                degraded = True
            elif fixes:
                parse_status = "partial" if "MutiHead" in latex_in or " , " in latex_in else "parsed"
            structure = {
                "latex": latex,
                "eq_label": eq_label,
                "context_before": prev_ctx,
                "context_after": next_ctx,
                "fixes": fixes,
            }
            label = f"({eq_label}) " if eq_label else ""
            search_text = f"[formula] {label}{latex}"
            if heading_path_norm:
                search_text = " > ".join(heading_path_norm) + "\n" + search_text
            if prev_ctx:
                search_text += "\n" + prev_ctx
            if fixes:
                quality["formula_fixes"].append({"element_id": el.get("element_id"), "fixes": fixes})

        elif el.get("type") == "figure":
            parse_status = "skipped_image"
            src = payload.get("image_path") or ""
            rel = None
            if src and copy_figures:
                src_path = Path(str(src).replace("file:///", "").replace("file://", ""))
                # Handle Windows paths that may come from URI form.
                if re.match(r"^[A-Za-z]:/", src_path.as_posix()) or src_path.exists():
                    candidate = Path(str(src).split("://")[-1]) if "://" in str(src) else Path(src)
                    # file:///d:/... -> /d:/... on some parsers; normalize.
                    s = str(src)
                    if s.startswith("file:///"):
                        s = s[8:]
                    elif s.startswith("file://"):
                        s = s[7:]
                    candidate = Path(s)
                    if candidate.exists():
                        dest = figures_dir / candidate.name
                        if candidate.resolve() != dest.resolve():
                            shutil.copy2(candidate, dest)
                        rel = f"figures/{dest.name}"
                        quality["figure_path_fixes"] += 1
            caption = payload.get("caption") or ""
            structure = {
                "caption": caption,
                "image_path": rel or src,
                "understanding": None,
            }
            search_text = f"[image] {caption}" if caption else f"[image] page={el.get('page')}"
            if heading_path_norm and caption:
                search_text = " > ".join(heading_path_norm) + "\n" + search_text
            if not caption:
                index_policy = "skip"

        elif role != "body":
            # Keep text but mark policy for stage-2.
            pass

        if index_policy != "embed":
            quality["dropped_embed_candidates"] += 1

        quality["role_counts"][role] = quality["role_counts"].get(role, 0) + 1
        quality["parse_status_counts"][parse_status] = quality["parse_status_counts"].get(parse_status, 0) + 1

        sequence.append(
            {
                "ordinal": len(sequence),
                "element_id": el.get("element_id"),
                "type": el.get("type"),
                "role": role,
                "index_policy": index_policy,
                "page": el.get("page"),
                "page_end": el.get("page_end"),
                "bbox": el.get("bbox"),
                "heading_path": el.get("heading_path") or [],
                "heading_path_norm": heading_path_norm,
                "parse_status": parse_status,
                "text": text,
                "search_text": search_text,
                "structure": structure,
                "payload": payload,
                "user_notice": user_notice,
                "atomic": atomic,
                "degraded": degraded,
                "source_label": el.get("source_label"),
            }
        )

    return {
        "schema": "ordered_document_v1",
        "document_id": rag.get("document_id"),
        "source_pdf": rag.get("source_pdf"),
        "source_rag_document": True,
        "parser": rag.get("parser") or {},
        "stats": {
            "n_elements": len(sequence),
            "by_type": {
                t: sum(1 for e in sequence if e["type"] == t)
                for t in sorted({e["type"] for e in sequence})
            },
            "by_role": dict(quality["role_counts"]),
            "by_parse_status": dict(quality["parse_status_counts"]),
        },
        "quality": quality,
        "sequence": sequence,
    }
