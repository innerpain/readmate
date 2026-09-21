"""Long-term memory candidates and session history compaction (A4 / A5)."""

from __future__ import annotations

from types import SimpleNamespace

from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.storage.agent_db import AgentDB


class _EmptyRegistry:
    def list_documents(self) -> list:
        return []


def _memory(db: AgentDB) -> MemoryService:
    collections = CollectionService(db=db, registry=_EmptyRegistry())
    settings = SimpleNamespace(digest_chars=600)
    return MemoryService(db=db, collections=collections, settings=settings)


def test_pending_candidate_is_absent_from_digest(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("papers")
    session_id = db.create_session(collection_id=collection_id, mode="chat")
    memory = _memory(db)

    memory.note(session_id=session_id, kind="preference", key="lang", value="zh")
    assert "zh" not in memory.digest(collection_id)
    assert "lang" not in memory.digest(collection_id)
    assert db.list_candidates(session_id, status="pending")


def test_confirm_promotes_candidate_into_long_term_tables(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("papers")
    session_id = db.create_session(collection_id=collection_id, mode="chat")
    memory = _memory(db)

    candidate_id = memory.note(
        session_id=session_id, kind="preference", key="style", value="bullet"
    )
    promoted = memory.confirm(session_id=session_id, candidate_ids=[candidate_id])

    assert promoted == 1
    digest = memory.digest(collection_id)
    assert "style=bullet" in digest
    assert db.list_candidates(session_id, status="pending") == []
    confirmed = db.list_candidates(session_id, status="confirmed")
    assert len(confirmed) == 1
    assert confirmed[0]["key"] == "style"
    assert db.get_preferences()["style"] == "bullet"


def test_new_session_reads_confirmed_preferences(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("papers")
    first = db.create_session(collection_id=collection_id, mode="chat")
    memory = _memory(db)

    candidate_id = memory.note(
        session_id=first, kind="preference", key="cite", value="page-first"
    )
    memory.confirm(session_id=first, candidate_ids=[candidate_id])

    second = db.create_session(collection_id=collection_id, mode="deep", title="next")
    assert second != first
    digest = memory.digest(collection_id)
    assert "cite=page-first" in digest
    assert db.get_preferences()["cite"] == "page-first"


def test_compact_messages_only_drops_rows_covered_by_summary(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    collection_id = db.upsert_collection("papers")
    session_id = db.create_session(collection_id=collection_id, mode="deep")

    message_ids = [
        db.append_message(session_id, "user" if index % 2 == 0 else "assistant", f"m{index}")
        for index in range(6)
    ]
    # Cover the first four rows; keep=2 among covered rows.
    upto = message_ids[3]
    db.set_session_summary(session_id, "earlier turns", upto_message_id=upto)

    deleted = db.compact_messages(session_id, keep=2)
    remaining = [row["id"] for row in db.list_messages(session_id)]
    deleted_ids = set(message_ids) - set(remaining)

    assert deleted > 0
    # Compaction never touches rows after the summary window.
    assert message_ids[4] in remaining
    assert message_ids[5] in remaining
    assert deleted_ids <= set(message_ids[:4])
    # Oldest covered rows are gone; a covered survivor remains near the window.
    assert message_ids[0] not in remaining
    assert message_ids[1] not in remaining
    assert any(message_id <= upto for message_id in remaining)
