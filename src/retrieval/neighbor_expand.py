"""Pure helpers for table/figure neighbor evidence windows."""

from __future__ import annotations


def neighbor_payload(chunk_id: str, payload: dict | None) -> dict[str, str]:
    payload = payload or {}
    metadata = dict(payload.get("metadata") or {})
    return {
        "chunk_id": chunk_id,
        "content": str(payload.get("document") or ""),
        "page": str(metadata.get("page") or ""),
        "section": str(metadata.get("section") or ""),
    }


def compose_evidence_window(
    prev_parts: list[dict[str, str]],
    primary: str,
    next_parts: list[dict[str, str]],
) -> str:
    blocks: list[str] = []
    for index, part in enumerate(prev_parts, start=1):
        blocks.append(f"[context:prev #{index}]\n{part.get('content') or ''}".strip())
    blocks.append(f"[primary]\n{primary}".strip())
    for index, part in enumerate(next_parts, start=1):
        blocks.append(f"[context:next #{index}]\n{part.get('content') or ''}".strip())
    return "\n\n".join(block for block in blocks if block)


def build_presentation(
    primary: str,
    prev: list[dict[str, str]] | None,
    next: list[dict[str, str]] | None,
    budget: int,
    *,
    primary_min_ratio: float = 0.6,
    neighbor_min_chars: int = 120,
) -> str:
    """Compose a budgeted presentation window with **primary-first priority**.

    ``primary`` (表体/正文本体) is the body; ``prev``/``next`` are
    ``neighbor_payload`` dicts.  ``budget`` caps the returned string length.
    Truncation semantics: ``…[本体省略 N 字符]`` means the primary body is
    incomplete; ``…[邻居省略 N 字符]`` marks a neighbour dropped or cut.  A
    ``presentation`` produced here is already bounded — callers must not
    re-slice with a raw ``[:budget]`` prefix (that reverts A5).
    """

    if budget <= 0:
        return ""
    primary_text = primary or ""
    prev_list = list(prev or [])[:2]
    next_list = list(next or [])[:2]
    header = "[primary]\n"
    sep = "\n\n"
    ceiling = max(0, budget - len(header))

    ordered: list[tuple[str, int, dict[str, str]]] = []
    for i in range(max(len(prev_list), len(next_list))):
        if i < len(prev_list):
            ordered.append(("prev", i + 1, prev_list[i]))
        if i < len(next_list):
            ordered.append(("next", i + 1, next_list[i]))

    if not ordered:
        body = primary_text[:ceiling]
        out = header + body
        if len(primary_text) > ceiling:
            out += f"…[本体省略 {len(primary_text) - ceiling} 字符]"
        return out

    floor = max(1, int(budget * primary_min_ratio))
    if len(primary_text) <= floor:
        primary_budget = len(primary_text)
    else:
        primary_budget = min(len(primary_text), ceiling)
    body = primary_text[:primary_budget]
    primary_block = header + body
    if len(primary_text) > primary_budget:
        primary_block += f"…[本体省略 {len(primary_text) - primary_budget} 字符]"
    parts: list[str] = [primary_block]
    used = len(primary_block)
    for kind, idx, part in ordered:
        content = str((part or {}).get("content") or "")
        if not content:
            continue
        neighbor_header = f"[context:{kind} #{idx}]\n"
        room = budget - used - len(sep) - len(neighbor_header)
        if room < neighbor_min_chars:
            parts.append(f"…[邻居省略 {len(content)} 字符]")
            used = len(sep.join(parts))
            continue
        kept = content[:room]
        block = neighbor_header + kept
        if len(kept) < len(content):
            block += f"…[邻居省略 {len(content) - len(kept)} 字符]"
        parts.append(block)
        used = len(sep.join(parts))
    out = sep.join(parts)
    if len(out) > budget:
        out = out[:budget]
    return out

