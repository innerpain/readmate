"""Chunk ordered_document_v1 into embeddable retrieval units (stage 2)."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

CHUNKER_VERSION = "ordered-aware-1"
PROSE_TARGET_TOKENS = 320
PROSE_MAX_TOKENS = 512
PROSE_OVERLAP_TOKENS = 64
TABLE_PACK_MAX_TOKENS = 512
TABLE_SUMMARY_MAX_TOKENS = 512
SHORT_CAPTION_CHARS = 200
MAX_NEIGHBORS = 2

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？；;])\s+")
_BYLINE_URL = re.compile(r"(https?://|www\.)", re.I)
_BYLINE_EMAIL = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
_AUTHORISH = re.compile(
    r"\b(university|laboratory|department|institute|gmbh|ltd\.?|inc\.?)\b",
    re.I,
)


def estimate_tokens(text: str) -> int:
    """Match EmbeddingGenerator's cheap fallback: ~4 chars/token."""

    return max(1, len(text) // 4) if text else 1


def take_within_budget(items: list[str], max_tokens: int) -> tuple[list[str], int]:
    """Longest prefix of ``items`` that fits ``max_tokens``, plus how many were left out.

    D13 (2026-09-20): shared by the refiner (table facts / search_text) and this module
    (table-pack highlights), so both sides measure a budget the same way.  Lives here
    because this module owns :func:`estimate_tokens`.

    At least one item is always taken: a budget smaller than a single item must not
    empty the list.
    """

    taken: list[str] = []
    used = 0
    for item in items:
        cost = estimate_tokens(item) + 1
        if taken and used + cost > max_tokens:
            break
        taken.append(item)
        used += cost
    return taken, len(items) - len(taken)


def chunk_ordered_document(
    ordered: dict[str, Any],
    *,
    filename: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert an ordered_document_v1 mapping into chunk dicts + report."""

    document_id = str(ordered.get("document_id") or "unknown")
    revision = str(ordered.get("revision") or "legacy")
    source_name = filename or str(ordered.get("source_pdf") or document_id)
    sequence = list(ordered.get("sequence") or [])

    demoted_byline = 0
    # D17 (2026-09-20): how many elements the policy pass threw away, per role.
    # The drops used to be silent -- the report said nothing, so "how much text
    # never made it into the corpus" was unanswerable.
    dropped_by_role: dict[str, int] = {}
    truncated = 0
    # D15 (2026-09-20): the report used to carry ``"table_packs_over_cap": 0`` as a
    # *literal* -- a fake statistic.  The real ceiling is TABLE_PACK_MAX_TOKENS and
    # over-long packs were truncated silently, so "how often did we cut a table" was
    # unobservable from the artifacts.  This counts it for real; ``document_quality``
    # turns a non-zero value into the ``tables_truncated`` flag the UI shows (the
    # chunker runs after ``ordered.json`` is written, so it cannot retro-add an
    # element-level ``user_notice`` -- the report is the honest channel).
    table_packs_over_cap = 0
    pending_prose: list[dict[str, Any]] = []
    structural_jobs: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    element_to_prose_chunk_ids: dict[str, list[str]] = {}

    def flush_prose() -> None:
        nonlocal pending_prose
        if not pending_prose:
            return
        produced = _chunk_prose_group(
            pending_prose,
            document_id=document_id,
            revision=revision,
            filename=source_name,
        )
        for chunk in produced:
            chunks.append(chunk)
            for eid in chunk.get("element_ids") or []:
                element_to_prose_chunk_ids.setdefault(str(eid), []).append(chunk["chunk_id"])
        pending_prose = []

    for element in sequence:
        if not isinstance(element, dict):
            continue
        etype = str(element.get("type") or "")
        policy = str(element.get("index_policy") or "embed")
        role = str(element.get("role") or "body")

        if policy == "drop" or role in {"reference", "preamble", "noise"}:
            dropped_by_role[role or "unknown"] = dropped_by_role.get(role or "unknown", 0) + 1
            continue
        if policy in {"skip", "metadata_only"} or role == "footnote":
            # D17 renamed this policy to ``skip`` on the producer side
            # (``ordered_refiner.assign_role``), but the artifacts already on disk
            # were written before the rename and still say ``metadata_only``.
            # ``index_builder`` re-chunks from those very files on every rebuild, so
            # without the legacy name here those elements would silently start being
            # embedded -- and ``document_quality._dropped_by_role`` (which mirrors
            # this rule) would report drops that never happened.  Measured on the
            # current corpus: 9 body-role elements across 5 documents.
            dropped_by_role[role or "unknown"] = dropped_by_role.get(role or "unknown", 0) + 1
            continue

        if etype == "heading":
            flush_prose()
            continue

        if etype in {"paragraph", "list_item"}:
            if _should_demote_byline(element, sequence):
                demoted_byline += 1
                continue
            pending_prose.append(element)
            continue

        if etype in {"table", "figure", "formula"}:
            flush_prose()
            structural_jobs.append(element)
            continue

    flush_prose()

    # Assign structural chunks with neighbor binding against prose produced so far.
    # Structural elements later in the doc still need neighbors from prose around them;
    # recompute neighbors using ordinal positions among all chunks after insertion.
    structural_chunks: list[dict[str, Any]] = []
    for element in structural_jobs:
        etype = str(element.get("type") or "")
        if etype == "formula":
            structural_chunks.append(
                _make_formula_chunk(
                    element,
                    document_id=document_id,
                    revision=revision,
                    filename=source_name,
                )
            )
        elif etype == "table":
            table_chunks, packs_over_cap = _make_table_chunks(
                element,
                document_id=document_id,
                revision=revision,
                filename=source_name,
            )
            structural_chunks.extend(table_chunks)
            table_packs_over_cap += packs_over_cap
        elif etype == "figure":
            structural_chunks.append(
                _make_figure_chunk(
                    element,
                    document_id=document_id,
                    revision=revision,
                    filename=source_name,
                )
            )

    # Merge structural chunks into reading order by source ordinal / pack order.
    merged = _merge_by_reading_order(chunks, structural_chunks)
    _bind_neighbors(merged, sequence)

    for chunk in merged:
        if estimate_tokens(chunk["retrieval_text"]) > PROSE_MAX_TOKENS + 8:
            chunk["retrieval_text"] = _truncate_to_tokens(chunk["retrieval_text"], PROSE_MAX_TOKENS)
            chunk["truncated"] = 1
            truncated += 1
        else:
            chunk["truncated"] = int(chunk.get("truncated") or 0)

    report = {
        "chunker_version": CHUNKER_VERSION,
        "document_id": document_id,
        "n_chunks": len(merged),
        "n_chunks_by_type": dict(Counter(c["chunk_type"] for c in merged)),
        "demoted_byline": demoted_byline,
        # D17: dropped elements by role (the policy pass above).
        "dropped_by_role": dict(sorted(dropped_by_role.items())),
        "dropped_total": sum(dropped_by_role.values()),
        "table_packs_over_cap": table_packs_over_cap,
        "max_retrieval_tokens": max((estimate_tokens(c["retrieval_text"]) for c in merged), default=0),
        "tables_missing_neighbors": sum(
            1
            for c in merged
            if c["chunk_type"] in {"table_summary", "table_pack", "figure"}
            and not c.get("neighbor_prev_chunk_ids")
            and not c.get("neighbor_next_chunk_ids")
        ),
        "truncated": truncated,
    }
    return merged, report


def write_chunks_artifacts(
    out_dir,
    chunks: list[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, str]:
    """Persist chunks.jsonl and chunk_report.json under a parsed doc directory."""

    import json
    from pathlib import Path

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    chunks_path = directory / "chunks.jsonl"
    report_path = directory / "chunk_report.json"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"chunks_path": str(chunks_path), "report_path": str(report_path)}


def _should_demote_byline(element: dict[str, Any], sequence: list[dict[str, Any]]) -> bool:
    role = str(element.get("role") or "")
    if role == "byline":
        return True
    text = str(element.get("text") or "").strip()
    if not text:
        return True
    # Before first Abstract/Introduction heading → treat authorish short lines as byline.
    ordinal = int(element.get("ordinal") or 0)
    saw_body_heading = False
    for other in sequence:
        if int(other.get("ordinal") or 0) >= ordinal:
            break
        if other.get("type") == "heading":
            heading = str(other.get("text") or "").strip().lower()
            if heading.startswith("abstract") or re.match(r"^\d+(\.\d+)*\s+", heading):
                saw_body_heading = True
                break
    if saw_body_heading:
        return False
    if _BYLINE_URL.search(text) or _BYLINE_EMAIL.search(text):
        return True
    if len(text) <= 180 and (
        _AUTHORISH.search(text)
        or re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text)
    ):
        return True
    return False


def _heading_path(element: dict[str, Any]) -> list[str]:
    path = element.get("heading_path_norm") or element.get("heading_path") or []
    if isinstance(path, list):
        return [str(p) for p in path if str(p).strip()]
    return [str(path)] if path else []


def _path_prefix(path: list[str]) -> str:
    return " > ".join(path) if path else ""


def _pages_of(element: dict[str, Any]) -> list[int]:
    """Every page the element occupies.

    A3: one element can straddle a page break (``page`` + ``page_end``), so a
    chunk built from it must not claim the start page alone -- that is how a
    quote that sits on p4 was cited as p3.
    """

    pages: list[int] = []
    for key in ("page", "page_end"):
        value = element.get(key)
        if value is None or value == "":
            continue
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(pages)) or [1]


def _chunk_prose_group(
    elements: list[dict[str, Any]],
    *,
    document_id: str,
    revision: str,
    filename: str,
) -> list[dict[str, Any]]:
    if not elements:
        return []
    # Group by heading path string.
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_key = None
    for element in elements:
        key = tuple(_heading_path(element))
        if current and key != current_key:
            groups.append(current)
            current = []
        current.append(element)
        current_key = key
    if current:
        groups.append(current)

    out: list[dict[str, Any]] = []
    for group in groups:
        path = _heading_path(group[0])
        prefix = _path_prefix(path)
        units: list[tuple[str, list[str], int]] = []
        for element in group:
            body = str(element.get("text") or element.get("search_text") or "").strip()
            if not body:
                continue
            units.append((body, [str(element.get("element_id"))], _pages_of(element)))

        # Merge units toward target, then split oversized.
        merged_units: list[tuple[str, list[str], list[int]]] = []
        buf_text = ""
        buf_ids: list[str] = []
        buf_pages: list[int] = []
        for body, ids, element_pages in units:
            candidate = f"{buf_text}\n{body}".strip() if buf_text else body
            if buf_text and estimate_tokens(_with_prefix(prefix, candidate)) > PROSE_TARGET_TOKENS:
                merged_units.append((buf_text, buf_ids, buf_pages))
                buf_text, buf_ids, buf_pages = body, list(ids), list(element_pages)
            else:
                buf_text = candidate
                buf_ids.extend(ids)
                buf_pages.extend(element_pages)
        if buf_text:
            merged_units.append((buf_text, buf_ids, buf_pages))

        pieces: list[tuple[str, list[str], list[int]]] = []
        for text, ids, pages in merged_units:
            if estimate_tokens(_with_prefix(prefix, text)) <= PROSE_MAX_TOKENS:
                pieces.append((text, ids, pages))
            else:
                pieces.extend(_split_long_prose(text, ids, pages, prefix))

        # Apply overlap across consecutive pieces in the same path.
        overlapped = _apply_overlap(pieces, prefix)
        for text, ids, pages in overlapped:
            retrieval = _with_prefix(prefix, text).strip()
            if not retrieval:
                continue
            out.append(
                _base_chunk(
                    chunk_type="prose",
                    document_id=document_id,
                    revision=revision,
                    filename=filename,
                    element_ids=ids,
                    pages=pages,
                    heading_path=path,
                    retrieval_text=retrieval,
                    atomic=0,
                    content_kind="text",
                )
            )
    return out


def _with_prefix(prefix: str, body: str) -> str:
    body = body.strip()
    if not prefix:
        return body
    return f"{prefix}\n{body}"


def _split_long_prose(
    text: str,
    ids: list[str],
    pages: list[int],
    prefix: str,
) -> list[tuple[str, list[str], list[int]]]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        sentences = [text]
    pieces: list[tuple[str, list[str], list[int]]] = []
    buf = ""
    for sentence in sentences:
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if buf and estimate_tokens(_with_prefix(prefix, candidate)) > PROSE_MAX_TOKENS:
            pieces.append((buf, list(ids), list(pages)))
            buf = sentence
        else:
            buf = candidate
    if buf:
        # Hard fallback: character windows if still too long.
        while estimate_tokens(_with_prefix(prefix, buf)) > PROSE_MAX_TOKENS:
            budget_chars = max(32, PROSE_MAX_TOKENS * 4 - len(prefix) - 1)
            pieces.append((buf[:budget_chars].strip(), list(ids), list(pages)))
            buf = buf[budget_chars:].strip()
        if buf:
            pieces.append((buf, list(ids), list(pages)))
    return pieces


def _apply_overlap(
    pieces: list[tuple[str, list[str], list[int]]],
    prefix: str,
) -> list[tuple[str, list[str], list[int]]]:
    if len(pieces) <= 1 or PROSE_OVERLAP_TOKENS <= 0:
        return pieces
    out: list[tuple[str, list[str], list[int]]] = []
    prev_text = ""
    for index, (text, ids, pages) in enumerate(pieces):
        if index == 0:
            out.append((text, ids, pages))
            prev_text = text
            continue
        overlap = _tail_tokens(prev_text, PROSE_OVERLAP_TOKENS)
        merged = f"{overlap} {text}".strip() if overlap else text
        # Keep under max after overlap; if overflow, skip overlap.
        if estimate_tokens(_with_prefix(prefix, merged)) > PROSE_MAX_TOKENS:
            merged = text
        out.append((merged, ids, pages))
        prev_text = text
    return out


def _tail_tokens(text: str, token_budget: int) -> str:
    if not text:
        return ""
    char_budget = max(1, token_budget * 4)
    if len(text) <= char_budget:
        return text
    clipped = text[-char_budget:]
    parts = clipped.split()
    return " ".join(parts[1:]) if len(parts) > 1 else clipped


def _make_formula_chunk(
    element: dict[str, Any],
    *,
    document_id: str,
    revision: str,
    filename: str,
) -> dict[str, Any]:
    path = _heading_path(element)
    retrieval = str(element.get("search_text") or element.get("text") or "").strip()
    if not retrieval:
        structure = element.get("structure") or {}
        latex = str(structure.get("latex") or element.get("text") or "")
        context = str(structure.get("context") or "")
        retrieval = _with_prefix(_path_prefix(path), f"[formula] {latex} {context}".strip())
    degraded = 1 if str(element.get("parse_status") or "") == "unparsed" else 0
    return _base_chunk(
        chunk_type="formula",
        document_id=document_id,
        revision=revision,
        filename=filename,
        element_ids=[str(element.get("element_id"))],
        pages=_pages_of(element),
        heading_path=path,
        retrieval_text=retrieval,
        atomic=1,
        content_kind="formula",
        parse_status=str(element.get("parse_status") or "parsed"),
        degraded=degraded,
        source_ordinal=int(element.get("ordinal") or 0),
    )


def _make_figure_chunk(
    element: dict[str, Any],
    *,
    document_id: str,
    revision: str,
    filename: str,
) -> dict[str, Any]:
    path = _heading_path(element)
    structure = element.get("structure") or {}
    caption = str(structure.get("caption") or element.get("text") or "").strip()
    image_path = str(structure.get("image_path") or element.get("image_path") or "")
    body = f"[figure] {caption}".strip()
    retrieval = _with_prefix(_path_prefix(path), body)
    figure_id = str(element.get("element_id"))
    chunk = _base_chunk(
        chunk_type="figure",
        document_id=document_id,
        revision=revision,
        filename=filename,
        element_ids=[figure_id],
        pages=_pages_of(element),
        heading_path=path,
        retrieval_text=retrieval,
        atomic=1,
        content_kind="figure",
        parse_status=str(element.get("parse_status") or "skipped_image"),
        figure_id=figure_id,
        image_path=image_path,
        parent_id=f"figure:{figure_id}",
        source_ordinal=int(element.get("ordinal") or 0),
    )
    return chunk


def _make_table_chunks(
    element: dict[str, Any],
    *,
    document_id: str,
    revision: str,
    filename: str,
) -> list[dict[str, Any]]:
    path = _heading_path(element)
    prefix = _path_prefix(path)
    structure = element.get("structure") or {}
    caption = str(structure.get("caption") or element.get("text") or "table").strip()
    short_caption = caption if len(caption) <= SHORT_CAPTION_CHARS else caption[: SHORT_CAPTION_CHARS - 1] + "…"
    headers = [str(h) for h in (structure.get("headers") or [])]
    rows = structure.get("rows") or []
    facts = [str(f) for f in (structure.get("facts") or [])]
    table_id = str(element.get("element_id"))
    parent_id = f"table:{table_id}"
    page = int(element.get("page") or 1)
    ordinal = int(element.get("ordinal") or 0)

    summary_lines = [
        f"[table-summary] {short_caption}",
        "columns: " + " | ".join(headers) if headers else "columns:",
    ]
    row_labels = []
    for row in rows[:20]:
        if isinstance(row, (list, tuple)) and row:
            row_labels.append(str(row[0]))
    if row_labels:
        summary_lines.append("row_labels: " + "; ".join(row_labels[:15]))
    # D13: the highlight list used to be ``facts[:8]`` -- a count, not a cost.
    highlights_facts, _highlights_dropped = take_within_budget(facts, TABLE_SUMMARY_MAX_TOKENS // 2)
    highlights = []
    for fact in highlights_facts:
        highlights.append(_compact_fact(fact, short_caption))
    if highlights:
        summary_lines.append("highlights: " + " ; ".join(highlights))
    summary_lines.append(f"shape: {len(rows)} rows × {len(headers)} cols")
    summary_body = "\n".join(summary_lines)
    summary_retrieval = _with_prefix(prefix, summary_body)
    if estimate_tokens(summary_retrieval) > TABLE_SUMMARY_MAX_TOKENS:
        summary_retrieval = _truncate_to_tokens(summary_retrieval, TABLE_SUMMARY_MAX_TOKENS)

    summary = _base_chunk(
        chunk_type="table_summary",
        document_id=document_id,
        revision=revision,
        filename=filename,
        element_ids=[table_id],
        pages=_pages_of(element),
        heading_path=path,
        retrieval_text=summary_retrieval,
        atomic=1,
        content_kind="table",
        table_id=table_id,
        parent_id=parent_id,
        source_ordinal=ordinal,
        pack_ordinal=-1,
    )

    packs: list[dict[str, Any]] = []
    # Prefer row packs; fall back to facts.
    pack_sources: list[str] = []
    if rows and headers:
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            cells = [str(c) for c in row]
            line = " | ".join(
                f"{headers[i]}: {cells[i]}" if i < len(headers) else cells[i]
                for i in range(len(cells))
            )
            pack_sources.append(line)
    elif facts:
        pack_sources = [_compact_fact(fact, short_caption) for fact in facts]
    else:
        pack_sources = [str(element.get("text") or "table")]

    header_line = "headers: " + " | ".join(headers) if headers else ""
    current_lines: list[str] = []
    pack_ordinal = 0
    # D15: how many packs the token ceiling actually cut (reported, not swallowed).
    packs_over_cap = 0

    def emit_pack(lines: list[str]) -> None:
        nonlocal pack_ordinal, packs_over_cap
        body = "\n".join(
            [
                f"[table] {short_caption}",
                header_line,
                *lines,
            ]
        ).strip()
        retrieval = _with_prefix(prefix, body)
        if estimate_tokens(retrieval) > TABLE_PACK_MAX_TOKENS:
            retrieval = _truncate_to_tokens(retrieval, TABLE_PACK_MAX_TOKENS)
            packs_over_cap += 1
        packs.append(
            _base_chunk(
                chunk_type="table_pack",
                document_id=document_id,
                revision=revision,
                filename=filename,
                element_ids=[table_id],
                pages=_pages_of(element),
                heading_path=path,
                retrieval_text=retrieval,
                atomic=1,
                content_kind="table",
                table_id=table_id,
                parent_id=parent_id,
                source_ordinal=ordinal,
                pack_ordinal=pack_ordinal,
            )
        )
        pack_ordinal += 1

    for line in pack_sources:
        tentative = current_lines + [line]
        body = "\n".join([f"[table] {short_caption}", header_line, *tentative]).strip()
        if current_lines and estimate_tokens(_with_prefix(prefix, body)) > TABLE_PACK_MAX_TOKENS:
            emit_pack(current_lines)
            current_lines = [line]
        else:
            current_lines = tentative
    if current_lines:
        emit_pack(current_lines)
    if not packs:
        emit_pack([str(element.get("text") or "table")])

    return [summary, *packs], packs_over_cap


def _compact_fact(fact: str, short_caption: str) -> str:
    text = fact.strip()
    # Drop leading [table-fact] and duplicated caption prefix when present.
    text = re.sub(r"^\[table-fact\]\s*", "", text)
    if short_caption and text.startswith(short_caption[:60]):
        # keep trailing metric part after last useful pipe clusters
        parts = [p.strip() for p in text.split("|")]
        if len(parts) >= 3:
            return " | ".join(parts[-3:])
    parts = [p.strip() for p in text.split("|")]
    if len(parts) >= 3:
        return " | ".join(parts[-3:])
    return text[:240]


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    budget = max(16, max_tokens * 4)
    if len(text) <= budget:
        return text
    return text[: budget - 1].rstrip() + "…"


def _base_chunk(
    *,
    chunk_type: str,
    document_id: str,
    revision: str,
    filename: str,
    element_ids: list[str],
    pages: list[int],
    heading_path: list[str],
    retrieval_text: str,
    atomic: int,
    content_kind: str,
    parse_status: str = "parsed",
    degraded: int = 0,
    table_id: str | None = None,
    figure_id: str | None = None,
    image_path: str = "",
    parent_id: str = "",
    source_ordinal: int = 10**9,
    pack_ordinal: int = -1,
) -> dict[str, Any]:
    pages_clean = sorted({int(p) for p in pages if isinstance(p, int) or str(p).isdigit()})
    page = pages_clean[0] if pages_clean else 1
    section = heading_path[-1] if heading_path else "Unknown"
    payload = {
        "document_id": document_id,
        "revision": revision,
        "filename": filename,
        "chunk_type": chunk_type,
        "parent_id": parent_id,
        "element_ids": list(element_ids),
        "pack_ordinal": pack_ordinal,
        "pages": pages_clean,
        "page": page,
        "page_end": pages_clean[-1] if pages_clean else page,
        "heading_path": list(heading_path),
        "section": section,
        "atomic": int(atomic),
        "degraded": int(degraded),
        "parse_status": parse_status,
        "table_id": table_id,
        "figure_id": figure_id,
        "content_kind": content_kind,
        "image_path": image_path,
        "neighbor_prev_chunk_ids": [],
        "neighbor_next_chunk_ids": [],
        "retrieval_text": retrieval_text,
        "index_policy_effective": "embed",
        "chunker_version": CHUNKER_VERSION,
        "source_ordinal": source_ordinal,
    }
    payload["chunk_id"] = _stable_chunk_id(payload)
    return payload


def _stable_chunk_id(payload: dict[str, Any]) -> str:
    material = "|".join(
        [
            str(payload.get("document_id")),
            str(payload.get("revision")),
            str(payload.get("chunk_type")),
            ",".join(payload.get("element_ids") or []),
            str(payload.get("pack_ordinal")),
            _norm(str(payload.get("retrieval_text") or "")),
        ]
    )
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:20]
    return f"chk_{digest}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _merge_by_reading_order(
    prose_chunks: list[dict[str, Any]],
    structural_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tagged = []
    for chunk in prose_chunks:
        # Approximate ordinal via min element id number if present, else large.
        ordinal = _min_element_ordinal(chunk.get("element_ids") or [])
        tagged.append((ordinal, -1, chunk))
    for chunk in structural_chunks:
        ordinal = int(chunk.get("source_ordinal") or _min_element_ordinal(chunk.get("element_ids") or []))
        pack = int(chunk.get("pack_ordinal") if chunk.get("pack_ordinal") is not None else -1)
        # summary (-1) before packs (0..)
        tagged.append((ordinal, pack, chunk))
    tagged.sort(key=lambda item: (item[0], item[1]))
    return [chunk for _, __, chunk in tagged]


def _min_element_ordinal(element_ids: list[str]) -> int:
    best = 10**9
    for eid in element_ids:
        match = re.search(r"(\d+)$", str(eid))
        if match:
            best = min(best, int(match.group(1)))
    return best


def _bind_neighbors(chunks: list[dict[str, Any]], sequence: list[dict[str, Any]]) -> None:
    prose = [c for c in chunks if c.get("chunk_type") == "prose"]
    if not prose:
        return

    # Map element_id -> ordinal from sequence
    ordinal_by_id = {
        str(el.get("element_id")): int(el.get("ordinal") or 0)
        for el in sequence
        if isinstance(el, dict) and el.get("element_id") is not None
    }

    def prose_anchor(chunk: dict[str, Any]) -> int:
        ids = chunk.get("element_ids") or []
        vals = [ordinal_by_id[i] for i in ids if i in ordinal_by_id]
        return min(vals) if vals else 10**9

    prose_sorted = sorted(prose, key=prose_anchor)

    for chunk in chunks:
        if chunk.get("chunk_type") not in {"table_summary", "table_pack", "figure"}:
            continue
        anchor_ids = chunk.get("element_ids") or []
        anchors = [ordinal_by_id[i] for i in anchor_ids if i in ordinal_by_id]
        if not anchors:
            continue
        anchor = min(anchors)
        prevs = [c for c in prose_sorted if prose_anchor(c) < anchor]
        nexts = [c for c in prose_sorted if prose_anchor(c) > anchor]
        chunk["neighbor_prev_chunk_ids"] = [c["chunk_id"] for c in prevs[-MAX_NEIGHBORS:]]
        chunk["neighbor_next_chunk_ids"] = [c["chunk_id"] for c in nexts[:MAX_NEIGHBORS]]
