"""The answer contract: blocks, verifiable citations, and non-source marks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.llm.types import ToolCall


@dataclass(frozen=True)
class CitationOut:
    chunk_id: str
    document_id: str = ""
    filename: str = ""
    page: int | None = None
    # A3: last page the cited chunk covers; equal to ``page`` (or None) when the
    # passage does not straddle a page break.
    page_end: int | None = None
    quote: str = ""


@dataclass
class AgentAnswer:
    text: str = ""
    citations: list[CitationOut] = field(default_factory=list)
    non_source: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)
    # A12: the model's own statement that it declined to answer because the
    # material does not support one.  Declared in the envelope rather than
    # guessed from the prose -- a Chinese refusal used to arrive as a normal
    # answer, so nothing downstream could tell it apart from a sourced one.
    refused: bool = False


# The envelope key the model is asked to produce, used to recover the answer text
# when the envelope itself does not parse (see ``_partial_answer_field``).
_ANSWER_KEY_RE = re.compile(r'"answer"\s*:\s*"')

_BACKSLASH = chr(92)
# JSON string escapes other than \uXXXX, built with chr() so this mapping holds
# no escape sequences of its own.
_JSON_ESCAPES = {
    "n": chr(10),
    "t": chr(9),
    "r": chr(13),
    "b": chr(8),
    "f": chr(12),
    '"': '"',
    "/": "/",
    _BACKSLASH: _BACKSLASH,
}


def parse_agent_answer(raw: str) -> AgentAnswer:
    """Parse the JSON answer envelope; tolerant of markdown fences."""

    text = (raw or "").strip()
    payload = _json_object(text)
    if payload is None:
        # A truncated or slightly malformed envelope used to be handed back
        # verbatim, so the wire format itself reached the user: the answer bubble
        # -- and the stored row -- showed ``{"answer": ..., "citations": [...],
        # "non_source": []}``.  Recover the field instead, and only fall back to
        # the raw text when there is no envelope to recover from.
        recovered = _partial_answer_field(text)
        if recovered is not None:
            return AgentAnswer(text=recovered, warnings=["answer_recovered_partial"])
        return AgentAnswer(text=text, warnings=["answer_not_json"])
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        # An object that parses but carries no ``answer`` key (or an empty one) is
        # not a usable response.  Keep whatever prose surrounded it and never fall
        # back to the envelope itself -- that leak is what this module prevents.
        return AgentAnswer(text=_strip_envelope(text), warnings=["answer_missing_in_envelope"])
    citations = [
        CitationOut(
            chunk_id=str(item.get("chunk_id") or ""),
            page=int(item["page"]) if isinstance(item.get("page"), (int, float, str)) and str(item["page"]).isdigit() else None,
            quote=str(item.get("quote") or ""),
        )
        for item in (payload.get("citations") or [])
        if isinstance(item, dict) and str(item.get("chunk_id") or "").strip()
    ]
    non_source = [str(item).strip() for item in (payload.get("non_source") or []) if str(item).strip()]
    return AgentAnswer(
        text=answer,
        citations=citations,
        non_source=non_source,
        refused=_as_bool(payload.get("refused")),
    )


def validate_citations(answer: AgentAnswer, observed: dict[str, dict]) -> AgentAnswer:
    """Drop any citation whose chunk_id was not observed through a tool this turn.

    ``observed`` maps chunk_id -> {"document_id", "filename", "page"} captured
    from this run's search/read observations.
    """

    kept: list[CitationOut] = []
    for citation in answer.citations:
        meta = observed.get(citation.chunk_id)
        if meta is None:
            answer.dropped_citations.append(citation.chunk_id)
            continue
        kept.append(
            CitationOut(
                chunk_id=citation.chunk_id,
                document_id=str(meta.get("document_id") or ""),
                filename=str(meta.get("filename") or ""),
                page=_page_of(citation.page, meta),
                page_end=_page_end_of(meta),
                quote=citation.quote,
            )
        )
    answer.citations = kept
    if answer.dropped_citations:
        answer.warnings.append("citations_dropped_unobserved")
    return answer


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_bool(value: object) -> bool:
    """Tolerant truthiness for a model-supplied flag (``true`` / ``"true"`` / ``1``)."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _observed_span(meta: dict) -> tuple[int | None, int | None]:
    """(start, end) of the pages the observed chunk covers; end defaults to start."""

    start = _as_int(meta.get("page"))
    end = _as_int(meta.get("page_end"))
    if start is None:
        return None, end
    return start, end if end is not None else start


def _page_of(model_page: int | None, meta: dict) -> int | None:
    """A3: which page a citation shows.

    A chunk's own page is the *start* of a span (a passage can run over a page
    break).  A model that names a page inside that span is naming a page the
    passage really covers -- the page its quote sits on -- so it is kept.  A page
    outside the span is a recollection rather than evidence, so the observed span
    start wins, as it does when the model names no page at all.
    """

    start, end = _observed_span(meta)
    if start is None:
        return model_page
    if model_page is not None and start <= model_page <= (end or start):
        return model_page
    return start


def _page_end_of(meta: dict) -> int | None:
    """A3: the chunk's last page, or None when unknown."""

    _start, end = _observed_span(meta)
    return end


def _strip_envelope(text: str) -> str:
    """Return ``text`` with its JSON object removed (the prose around it kept)."""

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return ""
    return (text[:start] + text[end + 1 :]).strip()


def _partial_answer_field(text: str) -> str | None:
    """Best-effort read of the ``answer`` value from an unparsable envelope.

    Scans the string literal that follows ``"answer":`` and stops at its closing
    quote -- or at the end of the text when the response was cut off mid-value --
    so a truncated stream still yields the answer instead of the envelope.
    Returns ``None`` when there is no envelope to recover from.
    """

    match = _ANSWER_KEY_RE.search(text)
    if match is None:
        return None
    recovered = _unescape_json_fragment(text[match.end() :]).strip()
    return recovered or None


def _unescape_json_fragment(fragment: str) -> str:
    """Unescape a JSON string body, tolerating an incomplete escape at the tail."""

    out: list[str] = []
    index = 0
    length = len(fragment)
    while index < length:
        char = fragment[index]
        if char == _BACKSLASH:
            if index + 1 >= length:
                break  # the stream stopped inside an escape sequence
            nxt = fragment[index + 1]
            if nxt == "u":
                hexdigits = fragment[index + 2 : index + 6]
                if len(hexdigits) < 4:
                    break
                try:
                    out.append(chr(int(hexdigits, 16)))
                except ValueError:
                    break
                index += 6
                continue
            out.append(_JSON_ESCAPES.get(nxt, nxt))
            index += 2
            continue
        if char == '"':
            break  # end of the value
        out.append(char)
        index += 1
    return "".join(out)


def _json_object(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def tool_call_signature(call: ToolCall) -> str:
    """Stable key for the per-turn plan cache and tool traces."""

    return f"{call.name}:{json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)}"
