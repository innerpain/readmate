"""D8 + D11: expose what the parse stage degraded, on the document surfaces.

Both fields the UI needs were already being *written* and never *read*:

* ``ordered.json`` carries per-element ``degraded`` / ``user_notice`` (the
  "未能可靠解析该表结构，保留原始 grid/markdown" note produced by
  ``ordered_refiner.build_table_structure``) -- D8: no reader anywhere in the repo.
* ``quality_report.json`` carries the role distribution, the dropped-embed
  candidate count and the table/formula degradation counters -- D11: written only.

This module is the single reader for both.  It is deliberately defensive: a
missing or corrupted artifact must answer ``quality: null``, never a 500 -- the
library page has to keep working while an ingestion is still running or after a
half-written parse directory.

Wire shape (frozen with the frontend, see ``frontend/src/types.ts``)::

    {
      "degraded": bool,                 # any element degraded, or any table/formula unparsed
      "user_notice": str | null,        # first real notice, de-duplicated, most common first
      "quality": {                      # null when neither artifact could be read
        "parse_quality": str,           # ok | partial | degraded
        "table_packs_over_cap": int,    # from chunk_report.json (0 until D15 lands)
        "tables_truncated": bool,       # any chunk truncated, or table_packs_over_cap > 0
        "dropped_elements": {"total": int, "by_role": {...}} | null
      } | null
    }
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# How many element rows we are willing to walk when looking for a user_notice.
# ``sequence`` is already parsed in memory by then, so this only bounds the scan
# on a pathological document; the notices live on atomic elements (tables,
# formulas) which sit in reading order, not necessarily at the front.
_NOTICE_SCAN_LIMIT = 2000


def _read_json(path: Path) -> object | None:
    """Parse one artifact, returning ``None`` for absent / unreadable / corrupt.

    A malformed artifact is logged at warning level (never raised): the caller's
    contract is "no quality data", and a broken JSON file is exactly the case the
    frontend must survive.
    """

    try:
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        logger.warning("quality artifact unreadable: %s (%s)", path, error, exc_info=True)
        return None


def _counts(value: object) -> dict[str, int]:
    """Coerce a ``{role: count}`` mapping, dropping anything not int-like."""

    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw in value.items():
        try:
            counts[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return counts


def _first_notice(ordered: object) -> str | None:
    """The most common user-facing notice among the document's degraded elements.

    Most-common rather than first-seen on purpose: a document with six degraded
    tables should not lead with whatever paragraph happened to be scanned first,
    and the per-element wording is a small, closed set (table / formula).
    """

    if not isinstance(ordered, dict):
        return None
    sequence = ordered.get("sequence")
    if not isinstance(sequence, list):
        return None
    notices: Counter[str] = Counter()
    for element in sequence[:_NOTICE_SCAN_LIMIT]:
        if not isinstance(element, dict):
            continue
        notice = element.get("user_notice")
        if isinstance(notice, str) and notice.strip():
            notices[notice.strip()] += 1
    if not notices:
        return None
    return notices.most_common(1)[0][0]


def _dropped_by_role(ordered: object) -> dict[str, int] | None:
    """Elements that never reached a chunk, split by role.

    Derived from ``ordered.json`` rather than from ``quality_report.json``, which
    only carries the flat ``dropped_embed_candidates`` total.  The rule mirrors
    ``ordered_chunker.chunk_ordered_document`` exactly (the function that decides
    what is dropped): a role of ``reference`` / ``preamble`` / ``noise``, a policy of
    ``drop`` / ``skip`` (plus the legacy ``metadata_only`` still present in artifacts
    written before D17's rename), a ``footnote``, and the byline demotion are all
    skipped, and everything else (paragraphs, list items, tables, figures,
    formulas, headings) lands in a chunk.

    A role the chunker *could* have demoted is not a claim we can make from
    ``ordered.json`` alone: the byline demotion looks at where an element sits
    relative to the first body heading, so those rows are attributed to the role
    the chunker reports in ``chunk_report.demoted_byline`` instead of being
    counted here.  ``None`` when the sequence is missing, so the UI can tell
    "nothing dropped" (``total: 0``) from "cannot compute".
    """

    if not isinstance(ordered, dict):
        return None
    sequence = ordered.get("sequence")
    if not isinstance(sequence, list):
        return None
    by_role: Counter[str] = Counter()
    for element in sequence:
        if not isinstance(element, dict):
            continue
        policy = str(element.get("index_policy") or "embed")
        role = str(element.get("role") or "body")
        if policy in {"drop", "skip", "metadata_only"} or role in {"reference", "preamble", "noise", "footnote"}:
            by_role[role] += 1
    return dict(by_role)


def _parse_quality(
    *,
    report: object | None,
    ordered: object | None,
    unparsed: int,
    notices: int,
) -> str:
    """``ok`` | ``partial`` | ``degraded`` for one document.

    ``degraded`` is reserved for a *structural* failure (a table or formula that
    could not be turned into structure at all); a document that merely skipped
    figures or lost some embed candidates is ``partial``.
    """

    if unparsed:
        return "degraded"
    quality = report.get("quality") if isinstance(report, dict) else None
    stats = report.get("stats") if isinstance(report, dict) else None
    if not isinstance(stats, dict) and isinstance(ordered, dict):
        stats = ordered.get("stats")
    by_status = stats.get("by_parse_status") if isinstance(stats, dict) else None
    skipped = 0
    if isinstance(by_status, dict):
        for key, raw in by_status.items():
            if key in {"parsed", "partial"}:
                continue
            try:
                skipped += int(raw)
            except (TypeError, ValueError):
                continue
    dropped = 0
    if isinstance(quality, dict):
        try:
            dropped = int(quality.get("dropped_embed_candidates") or 0)
        except (TypeError, ValueError):
            dropped = 0
    if notices or skipped or dropped:
        return "partial"
    return "ok"


def document_quality(document_id: str, parsed_root: Path) -> dict[str, object]:
    """D8/D11 payload for one document; never raises.

    ``parsed_root`` is the ``data/parsed`` directory (i.e. ``registry.data_dir /
    "parsed"``).  Both artifacts live under ``parsed_root/<document_id>/``.
    """

    directory = Path(parsed_root) / str(document_id)
    report = _read_json(directory / "quality_report.json")
    ordered = _read_json(directory / "ordered.json")
    chunk_report = _read_json(directory / "chunk_report.json")

    # No parse artifacts at all (queued / failed / deleted document): the honest
    # answer is "we know nothing", not a fabricated all-zero summary.
    if report is None and ordered is None and chunk_report is None:
        return {"degraded": False, "user_notice": None, "quality": None}

    quality_block = report.get("quality") if isinstance(report, dict) else None
    if not isinstance(quality_block, dict):
        quality_block = ordered.get("quality") if isinstance(ordered, dict) else None
    if not isinstance(quality_block, dict):
        quality_block = {}

    unparsed = 0
    for key in ("table_unparsed", "formula_unparsed"):
        try:
            unparsed += int(quality_block.get(key) or 0)
        except (TypeError, ValueError):
            continue

    by_role = _dropped_by_role(ordered)
    dropped_total = sum(by_role.values()) if by_role is not None else 0
    if by_role is not None and not by_role:
        try:
            dropped_total = int(quality_block.get("dropped_embed_candidates") or 0)
        except (TypeError, ValueError):
            dropped_total = 0
    if by_role is not None and isinstance(chunk_report, dict):
        try:
            by_role["byline"] = by_role.get("byline", 0) + int(chunk_report.get("demoted_byline") or 0)
        except (TypeError, ValueError):
            pass
        dropped_total = sum(by_role.values())

    try:
        packs_over_cap = int((chunk_report or {}).get("table_packs_over_cap") or 0)
    except (TypeError, ValueError):
        packs_over_cap = 0
    try:
        truncated_chunks = int((chunk_report or {}).get("truncated") or 0)
    except (TypeError, ValueError):
        truncated_chunks = 0

    notice = _first_notice(ordered)
    degraded = bool(unparsed) or notice is not None

    return {
        "degraded": degraded,
        "user_notice": notice,
        "quality": {
            "parse_quality": _parse_quality(
                report=report, ordered=ordered, unparsed=unparsed, notices=0 if notice is None else 1
            ),
            "table_packs_over_cap": packs_over_cap,
            "tables_truncated": bool(truncated_chunks or packs_over_cap),
            "dropped_elements": None if by_role is None else {"total": dropped_total, "by_role": by_role},
        },
    }
