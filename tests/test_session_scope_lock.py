"""问题.md 第二轮第 7/8 条: frozen prompt snapshot + session-level scope lock.

The two rules interlock: the material a conversation is allowed to use is chosen
*before* it starts and is declared inside a system prompt that is written once and
then reused byte-for-byte.  These tests pin down both halves -- the DB columns, the
upgrade path from a pre-change database, the digest that describes the material,
and the runtime behaviour that makes a mid-dialogue scope change impossible.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from src.adapter.contracts import DocInfo, SearchHit
from src.adapter.mock_adapter import MockRagAdapter
from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.agent.runtime import MODE_CHAT, MODE_DEEP, PROMPT_SNAPSHOT_VERSION, AgentRuntime
from src.agent.tools import ToolRunner
from src.config.settings import get_agent_settings
from src.llm.fake import FakeLLMClient
from src.llm.types import LLMStep, ToolCall
from src.storage.agent_db import _MIGRATION_1, SCHEMA_VERSION, AgentDB, session_scope

_ANSWER = json.dumps(
    {
        "answer": "好的，我在。",
        "citations": [],
        "non_source": [],
        "refused": False,
    }
)


_CITED_ANSWER = json.dumps(
    {
        "answer": "编码器由六层相同结构堆叠而成。",
        "citations": [{"chunk_id": "c1", "page": 3, "quote": "six identical layers"}],
        "non_source": [],
        "refused": False,
    }
)


class _EmptyRegistry:
    def list_documents(self) -> list:
        return []


class _Registry:
    """Minimal document registry: enough for listings plus the chunk digest."""

    def __init__(self, data_dir: Path, records: list) -> None:
        self.data_dir = Path(data_dir)
        self._records = list(records)

    def list_documents(self) -> list:
        return self._records


def _record(document_id: str, filename: str, pages: int, chunks: int) -> SimpleNamespace:
    return SimpleNamespace(
        document_id=document_id,
        original_filename=filename,
        status="completed",
        page_count=pages,
        chunk_count=chunks,
        failure_code=None,
    )


def _write_chunks(data_dir: Path, document_id: str, rows: list[dict]) -> None:
    path = data_dir / "parsed" / document_id / "chunks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id="c1",
        document_id="d1",
        filename="attention.pdf",
        page=3,
        section="3 Model Architecture",
        score=0.81,
        chunk_type="prose",
        presentation_content="encoders are a stack of six identical layers",
        excerpt="six identical layers",
    )


def _runtime(tmp_path: Path, *, script: list[LLMStep]):
    settings = get_agent_settings()
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("attention")
    db.add_documents(collection_id, ["d1"])
    adapter = MockRagAdapter(hits=[_hit()], docs=[])
    collections = CollectionService(db=db, registry=_EmptyRegistry())
    memory = MemoryService(db=db, collections=collections, settings=settings)
    runner = ToolRunner(adapter=adapter, collections=collections, memory=memory, settings=settings)
    runtime = AgentRuntime(llm=FakeLLMClient(list(script)), tools=runner, db=db, settings=settings)
    return runtime, db, collection_id


# --------------------------------------------------------------------- the column


def test_session_scope_reads_only_a_real_list():
    assert session_scope({}) is None
    assert session_scope({"scope_json": None}) is None
    assert session_scope({"scope_json": ""}) is None
    assert session_scope({"scope_json": "not json"}) is None
    assert session_scope({"scope_json": '{"a": 1}'}) is None  # a dict is not a scope
    # An empty list is a *decision*: this conversation is locked to the whole library.
    assert session_scope({"scope_json": "[]"}) == []
    assert session_scope({"scope_json": '["col_a", "col_b"]'}) == ["col_a", "col_b"]


def test_schema_version_matches_the_code_and_carries_every_column(tmp_path):
    """The stored marker must equal ``SCHEMA_VERSION`` and all migrated columns exist.

    Deliberately not pinned to a literal number: this test exists to catch a
    migration that was written but never appended to ``_MIGRATIONS`` (the version
    would lag), not to freeze the current revision.  D32 added ``summary_failures``.
    """

    db = AgentDB(tmp_path / "app.db")
    connection = sqlite3.connect(tmp_path / "app.db")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
    version = connection.execute("SELECT version FROM schema_version").fetchone()[0]
    connection.close()
    assert version == SCHEMA_VERSION
    assert {
        "scope_json",
        "system_prompt",
        "prompt_snapshot_version",
        "summary_failures",
    } <= columns


def test_a_database_from_before_the_change_upgrades_in_place(tmp_path):
    """A v1 file (no scope columns) must migrate the first time it is opened."""

    path = tmp_path / "app.db"
    connection = sqlite3.connect(path)
    connection.executescript(_MIGRATION_1)
    connection.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    connection.execute("INSERT INTO schema_version (version) VALUES (1)")
    connection.commit()
    connection.close()

    db = AgentDB(path)
    session_id = db.create_session(collection_id=None, mode="deep")
    db.set_session_scope(session_id, ["col_a"])
    assert session_scope(db.get_session(session_id)) == ["col_a"]
    rows = db.list_sessions()
    assert rows[0]["scope_json"] == '["col_a"]'


# -------------------------------------------------------------------- the digest


def test_digest_declares_the_material_the_way_a_tool_schema_would(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("attention 论文")
    db.add_documents(collection_id, ["d1"])
    data_dir = tmp_path / "data"
    _write_chunks(
        data_dir,
        "d1",
        [
            {
                "section": "1 Introduction",
                "context_text": "The dominant sequence transduction models are based on recurrent networks.",
            },
            {"section": "3 Model Architecture", "context_text": "The encoder is a stack of six identical layers."},
            {"section": "3 Model Architecture", "context_text": "Each layer has two sub-layers."},
            {"section": "Unknown", "context_text": "footer noise"},
        ],
    )
    service = CollectionService(db=db, registry=_Registry(data_dir, [_record("d1", "attention.pdf", 15, 61)]))

    digest = service.digest([collection_id])
    assert "attention.pdf (15 pages, 61 chunks)" in digest
    assert "sections: 1 Introduction | 3 Model Architecture" in digest  # de-duped, "Unknown" dropped
    assert "opens with: The dominant sequence transduction models" in digest
    assert "footer noise" not in digest


def test_digest_is_empty_for_an_empty_scope(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    service = CollectionService(db=db, registry=_EmptyRegistry())
    assert service.digest([]) == ""
    assert service.digest(["col_missing"]) == ""


# ------------------------------------------------------------------- the lock


def test_first_turn_locks_the_scope_and_later_turns_cannot_move_it(tmp_path):
    runtime, db, collection_id = _runtime(tmp_path, script=[LLMStep(content=_ANSWER), LLMStep(content=_ANSWER)])
    other = db.upsert_collection("other")
    session_id = db.create_session(collection_id=None, mode=MODE_CHAT)

    runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=[collection_id])
    assert session_scope(db.get_session(session_id)) == [collection_id]

    # Same scope again: no complaint.
    same = runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=[collection_id])
    assert "scope_overridden" not in same.warnings

    # A different scope mid-dialogue is ignored -- never silently obeyed.
    moved = runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=[other])
    assert session_scope(db.get_session(session_id)) == [collection_id]
    assert "scope_overridden" in moved.warnings


def test_the_whole_library_is_itself_a_lockable_choice(tmp_path):
    runtime, db, collection_id = _runtime(tmp_path, script=[LLMStep(content=_ANSWER), LLMStep(content=_ANSWER)])
    session_id = db.create_session(collection_id=None, mode=MODE_CHAT)

    runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=None)
    assert session_scope(db.get_session(session_id)) == []

    # Later asking for one collection does not narrow it: the choice was made.
    late = runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=[collection_id])
    assert session_scope(db.get_session(session_id)) == []
    assert "scope_overridden" in late.warnings


def test_the_session_never_retrieves_from_outside_its_locked_scope(tmp_path):
    """The lock is only real if the tools receive the session's scope, not the request's."""

    settings = get_agent_settings()
    db = AgentDB(tmp_path / "app.db")
    chosen = db.upsert_collection("chosen")
    other = db.upsert_collection("other")
    db.add_documents(chosen, ["d1"])
    adapter = MockRagAdapter(hits=[_hit()], docs=[])
    collections = CollectionService(db=db, registry=_EmptyRegistry())
    memory = MemoryService(db=db, collections=collections, settings=settings)
    runner = ToolRunner(adapter=adapter, collections=collections, memory=memory, settings=settings)
    script = [
        LLMStep(tool_calls=[ToolCall(id="1", name="search", arguments={"query": "encoder layers"})]),
        LLMStep(content=_CITED_ANSWER),
        LLMStep(content=_CITED_ANSWER),
    ]
    runtime = AgentRuntime(llm=FakeLLMClient(script), tools=runner, db=db, settings=settings)
    session_id = db.create_session(collection_id=None, mode=MODE_DEEP)

    runtime.run(session_id, "编码器有几层？", mode=MODE_DEEP, collection_ids=[chosen])
    runtime.run(session_id, "编码器有几层？", mode=MODE_DEEP, collection_ids=[other])

    assert adapter.search_calls, "the deep-mode turn must have searched"
    assert {call["collection_id"] for call in adapter.search_calls} == {chosen}


# ----------------------------------------------------------------- the snapshot


def test_the_snapshot_is_written_once_and_reused_byte_for_byte(tmp_path):
    runtime, db, collection_id = _runtime(tmp_path, script=[LLMStep(content=_ANSWER), LLMStep(content=_ANSWER)])
    session_id = db.create_session(collection_id=None, mode=MODE_CHAT)

    runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=[collection_id])
    first = db.get_session(session_id)
    assert first["system_prompt"]
    assert first["prompt_snapshot_version"] == PROMPT_SNAPSHOT_VERSION

    runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=[collection_id])
    second = db.get_session(session_id)
    assert second["system_prompt"] == first["system_prompt"]


def test_the_snapshot_declares_the_locked_material(tmp_path):
    runtime, db, collection_id = _runtime(tmp_path, script=[LLMStep(content=_ANSWER)])
    session_id = db.create_session(collection_id=None, mode=MODE_CHAT)

    runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=[collection_id])
    snapshot = db.get_session(session_id)["system_prompt"]
    assert "Session material" in snapshot
    assert f"attention ({collection_id})" in snapshot


def test_the_whole_library_scope_says_so_in_the_snapshot(tmp_path):
    runtime, db, _ = _runtime(tmp_path, script=[LLMStep(content=_ANSWER)])
    session_id = db.create_session(collection_id=None, mode=MODE_CHAT)

    runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=None)
    assert "collections: the whole library" in db.get_session(session_id)["system_prompt"]


def test_history_goes_back_to_real_turns_instead_of_one_system_blob(tmp_path):
    runtime, db, collection_id = _runtime(tmp_path, script=[LLMStep(content=_ANSWER), LLMStep(content=_ANSWER)])
    session_id = db.create_session(collection_id=None, mode=MODE_CHAT)

    runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=[collection_id])
    runtime.run(session_id, "你好呀", mode=MODE_CHAT, collection_ids=[collection_id])

    messages = runtime.llm.calls[-1]["messages"]  # what the model was actually sent
    assert messages[0]["role"] == "system"  # the frozen snapshot leads
    assert messages[-1] == {"role": "user", "content": "你好呀"}
    turns = [message["role"] for message in messages if message["role"] in {"user", "assistant"}]
    assert turns == ["user", "assistant", "user"]  # the earlier answer is a real turn
    assert all(not message["content"].startswith(("user: ", "assistant: ")) for message in messages)


def test_a_stale_snapshot_version_is_rebuilt(tmp_path):
    runtime, db, collection_id = _runtime(tmp_path, script=[LLMStep(content=_ANSWER)])
    session_id = db.create_session(collection_id=None, mode=MODE_CHAT)
    db.set_session_prompt_snapshot(session_id, "old rules", PROMPT_SNAPSHOT_VERSION - 1)

    runtime.run(session_id, "你好", mode=MODE_CHAT, collection_ids=[collection_id])
    rebuilt = db.get_session(session_id)["system_prompt"]
    assert "old rules" not in rebuilt
    assert "Session material" in rebuilt


def test_doc_info_is_still_the_listing_contract(tmp_path):
    """Belt-and-braces: the digest path must not have changed the listing shape."""

    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("c")
    db.add_documents(collection_id, ["d1"])
    service = CollectionService(db=db, registry=_Registry(tmp_path / "data", [_record("d1", "a.pdf", 3, 9)]))
    docs = service.list_docs_multi([collection_id])
    assert len(docs) == 1
    assert isinstance(docs[0], DocInfo)
    assert (docs[0].document_id, docs[0].filename, docs[0].page_count, docs[0].chunk_count) == ("d1", "a.pdf", 3, 9)
    # No chunk file on disk -> listed, but nothing to excerpt.
    assert service.digest([collection_id]) == "- a.pdf (3 pages, 9 chunks)"
