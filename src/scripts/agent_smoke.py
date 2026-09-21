"""Offline end-to-end smoke run for ReadMate: scripted model, mock adapter.

Run:  python -m src.scripts.agent_smoke

No network, no Chroma, no data/ dependency: this exercises the runtime gate, the
tool loop, citation validation, SQLite persistence and the memory write rules
against a temporary database.  Exits non-zero when any contract check fails.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.adapter.contracts import DocInfo, SearchHit
from src.adapter.mock_adapter import MockRagAdapter
from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.agent.runtime import MODE_CHAT, MODE_DEEP, AgentRuntime
from src.agent.tools import ToolRunner
from src.config.settings import get_agent_settings
from src.llm.fake import FakeLLMClient
from src.llm.types import LLMStep, ToolCall
from src.storage.agent_db import AgentDB


class _EmptyRegistry:
    """The smoke run needs no ingested documents; status is not under test."""

    def list_documents(self) -> list:
        return []


def _answer(chunk_id: str = "c1", page: int = 3) -> str:
    return json.dumps(
        {
            "answer": "The encoder is a stack of six identical layers.",
            "citations": [{"chunk_id": chunk_id, "page": page, "quote": "stack of N = 6"}],
            "non_source": [],
        }
    )


def main() -> int:
    settings = get_agent_settings()
    workdir = Path(tempfile.mkdtemp(prefix="readmate_smoke_"))
    db = AgentDB(workdir / "app.db")
    collection_id = db.upsert_collection("smoke")
    db.add_documents(collection_id, ["d1", "d2"])

    hits = [
        SearchHit(
            chunk_id="c1",
            document_id="d1",
            filename="attention.pdf",
            page=3,
            section="3 Model Architecture",
            score=0.81,
            chunk_type="prose",
            presentation_content="The encoder is composed of a stack of N = 6 identical layers.",
            excerpt="stack of N = 6",
        ),
        SearchHit(
            chunk_id="c2",
            document_id="d2",
            filename="bert.pdf",
            page=7,
            section="2 Related Work",
            score=0.66,
            chunk_type="prose",
            presentation_content="BERT uses a masked language model objective.",
            excerpt="masked language model",
        ),
    ]
    adapter = MockRagAdapter(
        hits=hits,
        docs=[DocInfo(document_id="d1", filename="attention.pdf", status="ready", page_count=15, chunk_count=42)],
    )
    collections = CollectionService(db=db, registry=_EmptyRegistry())
    memory = MemoryService(db=db, collections=collections, settings=settings)
    runner = ToolRunner(adapter=adapter, collections=collections, memory=memory, settings=settings)

    checks: dict[str, bool] = {}

    # ---- deep-read turn: answer first (gate must block), then search, then answer
    session_id = db.create_session(collection_id=collection_id, mode=MODE_DEEP)
    llm = FakeLLMClient(
        [
            LLMStep(content=_answer()),
            LLMStep(tool_calls=[ToolCall(id="1", name="search", arguments={"query": "encoder layers"})]),
            LLMStep(content=_answer()),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=runner, db=db, settings=settings)
    result = runtime.run(session_id, "Transformer 的编码器由多少层组成？")

    checks["deep gate blocked the unevidenced draft once"] = result.gate["blocks"] == 1
    checks["search counted as evidence"] = result.gate["searched"] is True
    checks["turn finished without failure"] = result.failure is None and result.refused is False
    checks["one citation survived validation"] = len(result.citations) == 1
    if result.citations:
        checks["citation document came from the observation"] = result.citations[0]["document_id"] == "d1"
        checks["citation filename came from the observation"] = result.citations[0]["filename"] == "attention.pdf"
        checks["citation page preserved"] = result.citations[0]["page"] == 3
    checks["tool trace recorded the search"] = [step["tool"] for step in result.tool_trace] == ["search"]
    checks["user message plus final answer persisted"] = db.count_messages(session_id) == 2
    checks["adapter saw one scoped search call"] = len(adapter.search_calls) == 1

    # ---- chat turn: a greeting must not touch the knowledge base
    chat_session = db.create_session(collection_id=collection_id, mode=MODE_CHAT)
    chat_llm = FakeLLMClient([LLMStep(content=_answer(chunk_id="c1"))])
    chat_runtime = AgentRuntime(llm=chat_llm, tools=runner, db=db, settings=settings)
    chat_result = chat_runtime.run(chat_session, "你好")
    checks["greeting skipped retrieval"] = len(adapter.search_calls) == 1  # unchanged from the deep turn
    checks["unobserved citation was dropped"] = chat_result.citations == []
    checks["drop was reported as a warning"] = "citations_dropped_unobserved" in chat_result.warnings

    # ---- memory rules: a note stays pending until the user confirms it
    candidate_id = memory.note(session_id=session_id, kind="preference", key="detail", value="先直觉后公式")
    checks["digest hides pending candidates"] = "先直觉后公式" not in memory.digest(collection_id)
    confirmed = memory.confirm(session_id=session_id, collection_id=collection_id)
    checks["confirm promoted exactly one candidate"] = confirmed == 1
    checks["digest shows the confirmed preference"] = "先直觉后公式" in memory.digest(collection_id)
    checks["candidate is no longer pending"] = [c["id"] for c in memory.pending(session_id)] == []
    checks["candidate id recorded"] = isinstance(candidate_id, int) and candidate_id > 0

    print(f"workdir:   {workdir}")
    print(f"answer:    {result.answer}")
    print(f"citations: {result.citations}")
    for name, ok in checks.items():
        print(f"[{'ok' if ok else 'FAIL'}] {name}")
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print(f"\nagent smoke FAILED ({len(failed)}/{len(checks)} checks)")
        return 1
    print(f"\nagent smoke ok ({len(checks)}/{len(checks)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
