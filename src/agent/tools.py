"""Expose the RAG Adapter (and memory) as LLM tools.

Two rules matter here:
  * no upload/reindex/delete tool exists -- document management stays in the
    existing HTTP+Celery path (RAG接入Agent方案.md 5.5);
  * ``memory_note`` only writes a *pending candidate*; nothing reaches the
    long-term tables until the user confirms (ReadMate-Agent 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.adapter.contracts import AdapterError
from src.llm.types import ToolCall


@dataclass
class Observation:
    ok: bool
    content: str
    error_code: str | None = None
    hits: list[dict] = field(default_factory=list)
    read: dict | None = None
    route: dict | None = None      # which retrieval track ran (direct|planned)
    signals: dict = field(default_factory=dict)  # D24/D25/D31/D37: retrieval diagnostics
    memory_auto_written: bool = False  # 用户明说"记住"→ 已直写长期库（见 MEMORY_NOTE_TOOL）


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the user's current collection for evidence passages. Call this before answering any question about the material.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for; keep the user's key terms and numbers verbatim."},
                "top_k": {"type": "integer", "description": "How many hits to return.", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    },
}

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": (
            "Open the full text of one passage by chunk_id, or every passage on a page of a document. "
            "document_id must be the UUID of one document (from a search hit or list_docs) -- NOT the "
            "collection id (col_...), and NOT a filename (a bare filename is accepted only as a "
            "last-resort fallback and may be ambiguous). "
            "A long passage (a big table, a full page) comes back in windows: when the result ends "
            "with a remaining-chars note, call read again with the same chunk_id/page and the "
            "suggested offset to continue reading where it stopped."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "document_id": {"type": "string", "description": "Document UUID from search hits or list_docs (avoid filenames)."},
                "page": {"type": "integer", "minimum": 1},
                "element_id": {"type": "string"},
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Character offset to continue from; use the value in the remaining-chars note.",
                },
                "max_chars": {"type": "integer", "minimum": 1, "description": "Window size override, in characters."},
            },
        },
    },
}

LIST_DOCS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_docs",
        "description": "List the documents in the current collection with ingestion status.",
        "parameters": {"type": "object", "properties": {}},
    },
}

MEMORY_GET_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_get",
        "description": "Read the learner profile, preferences and progress digest.",
        "parameters": {"type": "object", "properties": {}},
    },
}

MEMORY_NOTE_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_note",
        "description": (
            "Record a durable fact about the user (kind=profile|preference|progress). "
            "Normally it stays pending until the user confirms in the UI. "
            "Set explicit=true ONLY when the user explicitly asked you to remember it "
            "(e.g. \"记住…\") -- such a note is written to long-term memory immediately "
            "and you should tell the user it is saved, not that it awaits confirmation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["profile", "preference", "progress"]},
                "key": {"type": "string"},
                "value": {"type": "string"},
                "explicit": {
                    "type": "boolean",
                    "description": "true only when the user explicitly asked to remember this.",
                },
            },
            "required": ["kind", "key", "value"],
        },
    },
}


class ToolRunner:
    """Executes tool calls against the adapter plus the memory service."""

    def __init__(self, *, adapter, collections, memory, settings) -> None:
        self.adapter = adapter
        self.collections = collections
        self.memory = memory
        self.settings = settings
        # C5: consecutive empty searches per collection, scoped to one turn
        # (a ToolRunner is built fresh by ``build_runtime`` on every request,
        # so this naturally resets between turns).
        self._empty_search_streak: dict[str, int] = {}

    def specs(self) -> list[dict]:
        return [SEARCH_TOOL, READ_TOOL, LIST_DOCS_TOOL, MEMORY_GET_TOOL, MEMORY_NOTE_TOOL]

    def run(self, call: ToolCall, *, collection_id: str | None, session_id: str | None, user_text: str | None = None, collection_ids: list[str] | None = None, explicit_memory: bool = False) -> Observation:
        try:
            if call.name == "search":
                return self._search(call, collection_id, user_text, session_id, collection_ids)
            if call.name == "read":
                return self._read(call)
            if call.name == "list_docs":
                return self._list_docs(collection_id)
            if call.name == "memory_get":
                return Observation(ok=True, content=self.memory.digest(collection_id))
            if call.name == "memory_note":
                return self._memory_note(call, session_id, collection_id, explicit_memory)
        except AdapterError as error:
            return Observation(ok=False, content=f"tool failed: {error.code}", error_code=error.code)
        except Exception as error:  # pragma: no cover - defensive
            return Observation(ok=False, content=f"tool failed: {error}", error_code="tool_error")
        return Observation(ok=False, content=f"unknown tool: {call.name}", error_code="unknown_tool")

    # ------------------------------------------------------------------ tools
    def _search(self, call: ToolCall, collection_id: str | None, user_text: str | None = None, session_id: str | None = None, collection_ids: list[str] | None = None) -> Observation:
        query = str(call.arguments.get("query") or "").strip()
        if not query:
            return Observation(ok=False, content="search needs a query", error_code="invalid_arguments")
        # R7 / D-5b: ``collection_ids`` is the authoritative per-message scope.
        # A multi-collection turn searches the *union*; a single one keeps the
        # adapter's existing ``collection_id`` resolution.
        scope_ids = [cid for cid in (collection_ids or []) if cid] or ([collection_id] if collection_id else [])
        if not scope_ids:
            return Observation(ok=False, content="no collection is bound to this session", error_code="collection_empty")
        document_ids: set[str] | None = self.collections.document_ids_multi(scope_ids) if len(scope_ids) > 1 else None
        representative = scope_ids[0]
        top_k = call.arguments.get("top_k")
        streak = self._empty_search_streak.get(representative, 0)
        hits = self.adapter.search(
            representative,
            query,
            top_k=int(top_k) if top_k else None,
            document_ids=document_ids,
            # A4/D5: the raw user question drives the table gate, not the
            # model's (often English/abstract) search string.
            gate_text=user_text or query,
            # C5: feed the router the session mode and the number of prior empty
            # searches this turn, so a repeat of a zero-hit query escalates to
            # the planned track instead of looping on the same direct result.
            mode=self._session_mode(session_id),
            previous_empty_searches=streak,
        )
        self._empty_search_streak[representative] = 0 if hits else streak + 1
        encoded: list[dict] = []
        lines: list[str] = []
        for hit in hits[: self.settings.tool_max_hits]:
            # The adapter already composes ``presentation_content`` with
            # primary-first budgeting (neighbor_expand.build_presentation); a
            # second ``[:tool_text_chars]`` here would re-truncate from the
            # head and revert A5, so we pass the window through untouched.
            text = hit.presentation_content
            encoded.append(
                {
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.document_id,
                    "filename": hit.filename,
                    "page": hit.page,
                    # A3: the chunk's last page.  Equal to ``page`` unless the
                    # passage straddles a page break; the model sees the span so
                    # it can cite the page the quote really sits on.
                    "page_end": int(getattr(hit, "page_end", 0) or hit.page),
                    "chunk_type": hit.chunk_type,
                }
            )
            span = getattr(hit, "page_end", 0) or hit.page
            page_label = f"p{hit.page}" if int(span) == int(hit.page) else f"p{hit.page}-{span}"
            lines.append(
                f"- chunk_id={hit.chunk_id} file={hit.filename} {page_label} type={hit.chunk_type}\n{text}"
            )
        route = getattr(self.adapter, "last_route", None)
        route_payload = route.as_dict() if hasattr(route, "as_dict") else None
        if not lines:
            return Observation(ok=True, content="no hits", hits=[], route=route_payload)
        return Observation(ok=True, content="\n".join(lines), hits=encoded, route=route_payload)

    def _read(self, call: ToolCall) -> Observation:
        result = self.adapter.read(
            chunk_id=call.arguments.get("chunk_id"),
            document_id=call.arguments.get("document_id"),
            page=int(call.arguments["page"]) if call.arguments.get("page") else None,
            element_id=call.arguments.get("element_id"),
        )
        # D6: a table is one evidence object whose value is in its rows, so it gets
        # the table budget; everything else keeps the prose one.  ``offset`` lets
        # the model walk past either ceiling instead of hitting a wall (a long
        # table used to be cut mid-body with no way to ask for the rest).
        is_table = str(result.chunk_type or "") in {"table_summary", "table_pack"}
        default_budget = self.settings.tool_table_chars if is_table else self.settings.tool_text_chars
        try:
            offset = max(0, int(call.arguments.get("offset") or 0))
        except (TypeError, ValueError):
            offset = 0
        try:
            budget = int(call.arguments.get("max_chars") or default_budget)
        except (TypeError, ValueError):
            budget = default_budget
        budget = max(1, budget)
        body = result.text or ""
        text = body[offset : offset + budget]
        remaining = max(0, len(body) - (offset + len(text)))
        # D34 (2026-09-20): a page read is cut by the adapter at its own ceiling,
        # so ``body`` is already short and the arithmetic above reports 0 -- the
        # model could not tell there was more of the page.  Trust the adapter's
        # numbers when it says text was left behind.
        next_hint = offset + len(text)
        adapter_remaining = int(getattr(result, "remaining_chars", 0) or 0)
        if adapter_remaining > remaining:
            remaining = adapter_remaining
            next_hint = int(getattr(result, "next_offset", 0) or 0) or next_hint
        # A read carries no neighbours, so a head slice already keeps the body
        # first (primary-priority holds by construction); no window rebuild here.
        base = {
            "document_id": result.document_id,
            "filename": result.filename,
            "page": result.page,
            # A3: keep the span on read observations too -- a read is the other
            # path that feeds ``observed_chunks`` and therefore citations.
            "page_end": int(getattr(result, "page_end", 0) or result.page),
            "chunk_type": result.chunk_type,
            "text": text,
        }
        # A chunk read has a single citable id; a page read carries one per row.
        # Every one must reach ``observed_chunks`` or :func:`validate_citations`
        # silently drops the citation.
        citable: list[str] = []
        for chunk_id in [result.chunk_id, *result.chunk_ids]:
            if chunk_id and chunk_id not in citable:
                citable.append(chunk_id)
        payload = {"chunk_id": result.chunk_id, **base}
        hits = [{"chunk_id": chunk_id, **base} for chunk_id in citable]
        suffix = " (truncated)" if result.truncated else ""
        page_note = f" page_chunks={len(result.chunk_ids)}" if not result.chunk_id else ""
        window_note = ""
        if offset:
            window_note += f" offset={offset}"
        if remaining:
            window_note += f"\n…[剩余 {remaining} 字符，用 offset={next_hint} 继续读]"
        return Observation(
            ok=True,
            content=f"chunk_id={result.chunk_id or '-'}{page_note} p{result.page or '-'}{suffix}{window_note}\n{text}",
            read=payload,
            hits=hits,
        )

    def _list_docs(self, collection_id: str | None) -> Observation:
        if not collection_id:
            return Observation(ok=False, content="no collection bound", error_code="collection_empty")
        docs = self.collections.list_docs(collection_id)
        lines = [f"- {doc.filename} status={doc.status} pages={doc.page_count} chunks={doc.chunk_count}" for doc in docs]
        return Observation(ok=True, content="\n".join(lines) or "collection is empty")

    def _memory_note(self, call: ToolCall, session_id: str | None, collection_id: str | None = None, explicit_memory: bool = False) -> Observation:
        """``memory_note`` -- pending candidate by default.

        M1=c (2026-09-19): an *explicit* request is recognised two ways and
        either one is enough -- the runtime's regex over the user's own message
        (authoritative, deterministic) or the model's ``explicit`` argument
        (catches phrasings the regex misses).  Explicit notes are written to the
        long-term tables straight away and the observation says so, so the model
        reports "已保存" instead of asking the user to confirm again.
        """

        if not session_id:
            return Observation(ok=False, content="no session", error_code="invalid_arguments")
        kind = str(call.arguments.get("kind") or "profile")
        key = str(call.arguments.get("key") or "")
        value = str(call.arguments.get("value") or "")
        explicit = bool(explicit_memory or call.arguments.get("explicit"))
        if explicit:
            result = self.memory.note_confirmed(
                session_id=session_id,
                kind=kind,
                key=key,
                value=value,
                collection_id=collection_id,
            )
            if result.get("written"):
                return Observation(
                    ok=True,
                    content=f"saved to long-term memory immediately (candidate #{result['candidate_id']}); tell the user it is remembered",
                    memory_auto_written=True,
                )
            # progress needs a collection to land -> stays pending, say so plainly.
            return Observation(
                ok=True,
                content=(
                    f"noted as pending candidate #{result['candidate_id']}; "
                    f"cannot auto-save ({result.get('reason') or 'nothing landed'}) -- ask the user to confirm"
                ),
            )
        candidate_id = self.memory.note(session_id=session_id, kind=kind, key=key, value=value)
        return Observation(ok=True, content=f"noted as pending candidate #{candidate_id}; confirm with the user")

    def _session_mode(self, session_id: str | None) -> str | None:
        """Best-effort lookup of the turn's mode (deep|chat) for the router's
        ``mode`` signal.  Returns ``None`` (adapter default ``deep``) when the
        session/db is unavailable, so tool paths without a bound DB are unaffected.
        ``mode`` only enriches the optional LLM route prompt; the deterministic
        rule track ignores it."""

        if not session_id:
            return None
        for holder in (self.collections, self.memory):
            db = getattr(holder, "db", None)
            if db is None:
                continue
            try:
                row = db.get_session(session_id)
            except Exception:  # unknown session / db shape varies -> keep routing
                continue
            if isinstance(row, dict):
                return row.get("mode")
        return None
