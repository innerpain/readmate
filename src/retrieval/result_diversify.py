"""Collapse near-duplicate retrieval hits and enforce the table gate."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

# Metric / table-seeking cues (ZH + EN). Bare digits alone do NOT open tables.
# D21 (2026-09-20): the demo-corpus dataset names (nq / trivia / webquestions /
# curatedtrec) were removed -- they only worked for the three RAG papers.  The
# gate is a generic cue list now; the per-channel quota pick is the second half
# of the safety net.
_TABLE_METRIC_CUES = re.compile(
    r"(?i)("
    r"table|bleu|sts|spearman|score|scores|exact\s*match|em|"
    r"avg\.?|correlation|"
    r"表|分数|准确率|相关系数|基准|benchmark"
    r")"
)


def query_wants_tables(query: str) -> bool:
    """Whether the question is likely asking for a result table / metric cell."""

    text = query if isinstance(query, str) else ""
    return bool(_TABLE_METRIC_CUES.search(text))


def drops_unwanted_table(
    metadata: Mapping[str, Any],
    *,
    allow_tables: bool,
) -> bool:
    """The table gate's enforcement: table chunks are dropped when the question
    does not look table-seeking.

    D21 (2026-09-20): the B/C noise heuristics that used to live here were
    deleted.  They keyed off the demo corpus -- a hardcoded list of the three
    papers' titles decided which ``table_summary`` rows counted as
    "front-matter noise" -- so they failed open on any other document and could
    silently drop real result tables.  What remains is the gate itself
    (``query_wants_tables``) plus the per-channel quota pick, neither of which
    depends on the corpus.
    """

    chunk_type = str(metadata.get("chunk_type") or "")
    if chunk_type in {"table_summary", "table_pack"} and not allow_tables:
        return True
    return False


def apply_score_floor(
    hits: Sequence[Any],
    score_fn: Callable[[Any], float],
    *,
    ratio: float,
) -> list[Any]:
    """Drop hits whose score is far below the best hit (weak B/C tails)."""

    if not hits or ratio <= 0:
        return list(hits)
    scores = [float(score_fn(hit)) for hit in hits]
    top = max(scores) if scores else 0.0
    if top <= 0:
        return list(hits)
    threshold = top * ratio
    kept = [hit for hit, score in zip(hits, scores) if score >= threshold]
    return kept or list(hits[:1])


def collapse_key(metadata: Mapping[str, Any], *, chunk_id: str = "") -> tuple:
    """Group key for diversity: one slot per table / figure / formula, and one per
    prose chunk.

    ``chunk_id`` is passed explicitly by the caller because it is the *hit's*
    identity, not necessarily a metadata field: relying on
    ``metadata["chunk_id"]`` alone silently fell back to the old per-page grouping
    for any snapshot or caller whose metadata omits it (found by an independent
    review of D22).
    """

    chunk_type = str(metadata.get("chunk_type") or "")
    document_id = str(metadata.get("document_id") or "")
    page = int(metadata.get("page") or 0)
    table_id = metadata.get("table_id")
    figure_id = metadata.get("figure_id")
    if table_id:
        return ("table", str(table_id))
    if figure_id:
        return ("figure", str(figure_id))
    if chunk_type == "formula":
        return ("formula", document_id, page)
    # D22 (2026-09-20): prose no longer collapses per (document, page).  One page
    # usually carries several distinct passages, and keeping only the first cost
    # real evidence; every chunk id is its own slot now.  Table / figure / formula
    # keep their slot constraint (one slot per table, figure or formula).
    return ("prose", chunk_id or str(metadata.get("chunk_id") or "") or f"{document_id}:{page}")


def prefers_over(new_meta: Mapping[str, Any], old_meta: Mapping[str, Any]) -> bool:
    """Prefer table_pack over table_summary when both map to the same table_id."""

    new_type = str(new_meta.get("chunk_type") or "")
    old_type = str(old_meta.get("chunk_type") or "")
    return new_type == "table_pack" and old_type == "table_summary"


def diversify_hits(
    hits: Sequence[Any],
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    *,
    final_k: int,
    query: str = "",
    score_fn: Callable[[Any], float] | None = None,
    score_floor_ratio: float = 0.82,
) -> list[Any]:
    """Greedily keep diverse hits; the table gate drops tables the query did not ask for."""

    if final_k < 1 or not hits:
        return []

    allow_tables = query_wants_tables(query)
    ranked = list(hits)
    if score_fn is not None and score_floor_ratio > 0:
        ranked = apply_score_floor(ranked, score_fn, ratio=score_floor_ratio)

    selected: list[Any] = []
    key_index: dict[tuple, int] = {}
    for hit in ranked:
        if len(selected) >= final_k:
            break
        chunk_id = str(getattr(hit, "chunk_id", "") or "")
        if not chunk_id:
            continue
        meta = metadata_by_id.get(chunk_id) or {}
        if drops_unwanted_table(meta, allow_tables=allow_tables):
            continue
        key = collapse_key(meta, chunk_id=chunk_id)
        if key in key_index:
            idx = key_index[key]
            old_id = str(getattr(selected[idx], "chunk_id", "") or "")
            old_meta = metadata_by_id.get(old_id) or {}
            if prefers_over(meta, old_meta):
                selected[idx] = hit
            continue
        key_index[key] = len(selected)
        selected.append(hit)
    return selected[:final_k]


def ensure_hit_present(selected: Sequence[Any], required: Any | None, *, final_k: int) -> list[Any]:
    """Force ``required`` into the window (front) without exceeding ``final_k``."""

    if required is None or final_k < 1:
        return list(selected)[:final_k]
    required_id = str(getattr(required, "chunk_id", "") or "")
    if not required_id:
        return list(selected)[:final_k]
    if any(str(getattr(hit, "chunk_id", "") or "") == required_id for hit in selected):
        return list(selected)[:final_k]
    merged = [required, *[hit for hit in selected if str(getattr(hit, "chunk_id", "") or "") != required_id]]
    return merged[:final_k]
