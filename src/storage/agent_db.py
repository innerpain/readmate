"""SQLite store for ReadMate sessions, collections and long-term memory.

Same property as the document registry: no in-memory state, so an API restart
loses nothing.  Every call opens a short-lived connection; WAL keeps a polling
UI reader from blocking an agent turn.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence
from uuid import uuid4

SCHEMA_VERSION = 3

_MIGRATION_1 = """
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS collection_documents (
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (collection_id, document_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    collection_id TEXT REFERENCES collections(id) ON DELETE SET NULL,
    mode TEXT NOT NULL CHECK (mode IN ('deep', 'chat')),
    summary TEXT,
    summary_upto_message_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_name TEXT,
    tool_call_id TEXT,
    payload_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS memory_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'rejected')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_session ON memory_candidates(session_id, status);
CREATE TABLE IF NOT EXISTS user_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    identity TEXT NOT NULL DEFAULT '',
    major TEXT NOT NULL DEFAULT '',
    goal TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learning_progress (
    collection_id TEXT PRIMARY KEY,
    last_focus TEXT NOT NULL DEFAULT '',
    open_questions TEXT NOT NULL DEFAULT '',
    next_step TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
"""

# v2 (2026-09-19, 问题.md 第二轮第 7/8 条): a session keeps the scope it started
# with and the system prompt it was born with.
#
# ``scope_json`` holds the collection ids chosen *before* the conversation began
# (NULL = not decided yet, "[]" = deliberately the whole library).  ``system_prompt``
# is the frozen snapshot: rebuilding it every turn made the prompt prefix drift, so
# nothing about the model's instructions was cacheable or reproducible.
_MIGRATION_2 = """
ALTER TABLE sessions ADD COLUMN scope_json TEXT;
ALTER TABLE sessions ADD COLUMN system_prompt TEXT;
ALTER TABLE sessions ADD COLUMN prompt_snapshot_version INTEGER NOT NULL DEFAULT 0;
"""

# 批 D (D32) 2026-09-20: consecutive summary-generation failures for this session.
# The counter lives next to the summary it protects, so the breaker survives a
# worker restart (an in-process dict would forget and retry forever).
_MIGRATION_3 = """
ALTER TABLE sessions ADD COLUMN summary_failures INTEGER NOT NULL DEFAULT 0;
"""

_MIGRATIONS = (_MIGRATION_1, _MIGRATION_2, _MIGRATION_3)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_scope(session: dict) -> list[str] | None:
    """The collection ids a session is locked to, or ``None`` while undecided.

    ``None`` means "the conversation has not started yet, the caller still decides";
    an empty list means "locked to the whole library on purpose" (问题.md 第 8 条).
    """

    raw = session.get("scope_json")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [str(item) for item in parsed if item]


class AgentDB:
    """Thin, explicit data access; SQL lives here and nowhere else."""

    def __init__(self, path: Path | str | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.path = Path(path) if path is not None else project_root / "data" / "app.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ------------------------------------------------------------- plumbing
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = connection.execute("SELECT version FROM schema_version").fetchone()
            current = int(row["version"]) if row else 0
            for index, script in enumerate(_MIGRATIONS, start=1):
                if index <= current:
                    continue
                connection.execute("BEGIN IMMEDIATE")
                for statement in script.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                connection.execute("DELETE FROM schema_version")
                connection.execute("INSERT INTO schema_version (version) VALUES (?)", (index,))
                connection.execute("COMMIT")

    # ---------------------------------------------------------- collections
    def upsert_collection(self, name: str, collection_id: str | None = None) -> str:
        with self._connect() as connection:
            existing = connection.execute("SELECT id FROM collections WHERE name = ?", (name,)).fetchone()
            if existing:
                return str(existing["id"])
            new_id = collection_id or f"col_{uuid4().hex[:12]}"
            connection.execute(
                "INSERT INTO collections (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (new_id, name, _now(), _now()),
            )
            return new_id

    def list_collections(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT c.id, c.name, c.updated_at, COUNT(d.document_id) AS document_count "
                "FROM collections c LEFT JOIN collection_documents d ON d.collection_id = c.id "
                "GROUP BY c.id ORDER BY c.created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def rename_collection(self, collection_id: str, name: str) -> dict:
        """Rename one collection (FE-2).

        ``collections.name`` is UNIQUE, so a duplicate raises ``ValueError`` and the
        route turns it into a 409 rather than a 500 -- the caller can tell "that name
        is taken" from "the database is broken".
        """

        with self._connect() as connection:
            existing = connection.execute("SELECT id FROM collections WHERE name = ?", (name,)).fetchone()
            if existing and str(existing["id"]) != collection_id:
                raise ValueError("collection_name_taken")
            cursor = connection.execute(
                "UPDATE collections SET name = ?, updated_at = ? WHERE id = ?",
                (name, _now(), collection_id),
            )
            renamed = cursor.rowcount > 0
            row = connection.execute("SELECT id, name FROM collections WHERE id = ?", (collection_id,)).fetchone()
        return {"renamed": renamed, "collection_id": collection_id, "name": str(row["name"]) if row else name}

    def add_documents(self, collection_id: str, document_ids: Sequence[str]) -> int:
        with self._connect() as connection:
            for document_id in dict.fromkeys(document_ids):
                connection.execute(
                    "INSERT OR IGNORE INTO collection_documents (collection_id, document_id, added_at) VALUES (?, ?, ?)",
                    (collection_id, document_id, _now()),
                )
        return len(self.collection_document_ids(collection_id))

    def remove_document(self, collection_id: str, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM collection_documents WHERE collection_id = ? AND document_id = ?",
                (collection_id, document_id),
            )

    def delete_collection(self, collection_id: str) -> dict:
        """Delete one collection; report what went with it (C3).

        The schema already states the intent, and ``_connect`` turns foreign keys
        on, so the database does the work: ``collection_documents`` rows cascade
        away and ``sessions.collection_id`` becomes NULL.  Sessions are therefore
        *detached*, never deleted -- their history stays readable.
        """

        with self._connect() as connection:
            bound = connection.execute(
                "SELECT COUNT(*) AS n FROM sessions WHERE collection_id = ?", (collection_id,)
            ).fetchone()
            detached = int(bound["n"]) if bound else 0
            cursor = connection.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
            deleted = cursor.rowcount > 0
        return {"deleted": deleted, "collection_id": collection_id, "detached_sessions": detached if deleted else 0}

    def collection_document_ids(self, collection_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT document_id FROM collection_documents WHERE collection_id = ? ORDER BY added_at",
                (collection_id,),
            ).fetchall()
        return [str(row["document_id"]) for row in rows]

    # ------------------------------------------------------------- sessions
    def create_session(self, *, collection_id: str | None, mode: str = "deep", title: str = "") -> str:
        session_id = f"ses_{uuid4().hex[:12]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (id, title, collection_id, mode, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, title, collection_id, mode, _now(), _now()),
            )
        return session_id

    def get_session(self, session_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        return dict(row)

    def list_sessions(self, collection_id: str | None = None, limit: int = 20, offset: int = 0) -> list[dict]:
        """Recent sessions, newest first (FE-4 adds ``offset`` for 加载更多)."""

        query = "SELECT id, title, mode, collection_id, scope_json, created_at, updated_at FROM sessions"
        params: tuple = ()
        if collection_id:
            query += " WHERE collection_id = ?"
            params = (collection_id,)
        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params = params + (max(1, min(int(limit), 100)), max(0, int(offset)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def set_session_mode(self, session_id: str, mode: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE sessions SET mode = ?, updated_at = ? WHERE id = ?", (mode, _now(), session_id))

    def set_session_summary(self, session_id: str, summary: str, upto_message_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET summary = ?, summary_upto_message_id = ?, updated_at = ? WHERE id = ?",
                (summary, upto_message_id, _now(), session_id),
            )

    def summary_failures(self, session_id: str) -> int:
        """Consecutive failed summary attempts (批 D / D32 breaker input)."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary_failures FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return int(row["summary_failures"] or 0) if row else 0

    def bump_summary_failure(self, session_id: str) -> int:
        """Count one failure and return the new total."""

        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET summary_failures = COALESCE(summary_failures, 0) + 1 WHERE id = ?",
                (session_id,),
            )
            row = connection.execute(
                "SELECT summary_failures FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return int(row["summary_failures"] or 0) if row else 0

    def reset_summary_failures(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET summary_failures = 0 WHERE id = ?", (session_id,)
            )

    def set_session_title(self, session_id: str, title: str) -> None:
        """Rename a session (C4).

        ``create_session`` always took a ``title``; only the setter and the HTTP
        surface were missing, so every session kept the placeholder name it was
        born with.
        """

        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), session_id),
            )

    def set_session_scope(self, session_id: str, collection_ids: Sequence[str]) -> None:
        """Lock the session's material scope (问题.md 第二轮第 8 条).

        The scope is chosen *before* the conversation starts and is never rewritten
        afterwards.  An empty list is a deliberate choice ("the whole library"),
        which is exactly why the column distinguishes NULL (undecided) from ``[]``.
        """

        payload = json.dumps([str(cid) for cid in collection_ids if cid], ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET scope_json = ?, updated_at = ? WHERE id = ?",
                (payload, _now(), session_id),
            )

    def set_session_prompt_snapshot(self, session_id: str, prompt: str, version: int) -> None:
        """Freeze the system prompt this session was born with (第 7 条)."""

        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET system_prompt = ?, prompt_snapshot_version = ?, updated_at = ? WHERE id = ?",
                (prompt, int(version), _now(), session_id),
            )

    def delete_session(self, session_id: str) -> dict:
        """Delete a session together with its history (C4), reporting the damage.

        ``messages`` and ``memory_candidates`` both declare
        ``ON DELETE CASCADE`` and ``_connect`` enables foreign keys, so the
        transcript and any pending memory candidates go with the row.  The
        documents and the index are untouched -- a session is only a scope over
        evidence that lives elsewhere.
        """

        with self._connect() as connection:
            messages = connection.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
            candidates = connection.execute(
                "SELECT COUNT(*) AS n FROM memory_candidates WHERE session_id = ?", (session_id,)
            ).fetchone()
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            deleted = cursor.rowcount > 0
        return {
            "deleted": deleted,
            "session_id": session_id,
            "deleted_messages": int(messages["n"]) if messages and deleted else 0,
            "deleted_memory_candidates": int(candidates["n"]) if candidates and deleted else 0,
        }

    # ------------------------------------------------------------- messages
    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        payload_json: str | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages (session_id, role, content, tool_name, tool_call_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, role, content, tool_name, tool_call_id, payload_json, _now()),
            )
            connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))
            return int(cursor.lastrowid)

    def recent_messages(self, session_id: str, limit: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
                (session_id, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_messages(
        self,
        session_id: str,
        limit: int | None = None,
        *,
        before_id: int | None = None,
    ) -> list[dict]:
        """Messages of one session, oldest first.

        ``limit`` alone keeps the pre-D41 meaning (the *oldest* ``limit`` rows).
        D41 adds ``before_id``: with it, the **newest** ``limit`` rows strictly
        older than that message id are selected (``ORDER BY id DESC LIMIT``) and
        then reversed so the caller still receives ascending reading order -- the
        shape the chat view and the DB's other callers expect.  Both parameters
        are optional, so every existing caller (``db.list_messages(session_id)``)
        behaves exactly as before.
        """

        if before_id is None:
            query = "SELECT * FROM messages WHERE session_id = ? ORDER BY id"
            params: tuple = (session_id,)
            if limit:
                query += " LIMIT ?"
                params = (session_id, int(limit))
        elif limit:
            query = (
                "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? AND id < ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id"
            )
            params = (session_id, int(before_id), int(limit))
        else:
            query = "SELECT * FROM messages WHERE session_id = ? AND id < ? ORDER BY id"
            params = (session_id, int(before_id))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def count_messages(self, session_id: str, role: str | None = None) -> int:
        """Message count for a session.  ``role`` filters to e.g. ``"user"``,
        which is what the periodic memory review counts (it must count *user*
        turns, not every row, or the cadence would halve with each tool round)."""

        if role:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND role = ?",
                    (session_id, role),
                ).fetchone()
            return int(row["n"])
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)).fetchone()
        return int(row["n"])

    def compact_messages(self, session_id: str, keep: int) -> int:
        """Drop older rows once a summary covers them (history stays bounded)."""

        with self._connect() as connection:
            row = connection.execute("SELECT summary_upto_message_id FROM sessions WHERE id = ?", (session_id,)).fetchone()
            upto = int(row["summary_upto_message_id"] or 0) if row else 0
            if not upto:
                return 0
            rows = connection.execute(
                "SELECT id FROM messages WHERE session_id = ? AND id <= ? ORDER BY id DESC LIMIT ?",
                (session_id, upto, int(keep)),
            ).fetchall()
            boundary = int(rows[-1]["id"]) if rows else upto
            cursor = connection.execute("DELETE FROM messages WHERE session_id = ? AND id <= ?", (session_id, boundary))
            return int(cursor.rowcount or 0)

    # --------------------------------------------------------- long-term memory
    def get_profile(self) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM user_profile WHERE id = 1").fetchone()
        return dict(row) if row else {"identity": "", "major": "", "goal": "", "updated_at": ""}

    def set_profile(self, *, identity: str | None = None, major: str | None = None, goal: str | None = None) -> None:
        current = self.get_profile()
        merged = {
            "identity": identity if identity is not None else current.get("identity", ""),
            "major": major if major is not None else current.get("major", ""),
            "goal": goal if goal is not None else current.get("goal", ""),
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO user_profile (id, identity, major, goal, updated_at) VALUES (1, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET identity=excluded.identity, major=excluded.major, goal=excluded.goal, updated_at=excluded.updated_at",
                (merged["identity"], merged["major"], merged["goal"], _now()),
            )

    def set_preference(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, _now()),
            )

    def get_preferences(self) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value FROM user_preferences").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def set_progress(
        self,
        collection_id: str,
        *,
        last_focus: str | None = None,
        open_questions: str | None = None,
        next_step: str | None = None,
    ) -> None:
        current = self.get_progress(collection_id)
        merged = {
            "last_focus": last_focus if last_focus is not None else current["last_focus"],
            "open_questions": open_questions if open_questions is not None else current["open_questions"],
            "next_step": next_step if next_step is not None else current["next_step"],
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO learning_progress (collection_id, last_focus, open_questions, next_step, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(collection_id) DO UPDATE SET "
                "last_focus=excluded.last_focus, open_questions=excluded.open_questions, "
                "next_step=excluded.next_step, updated_at=excluded.updated_at",
                (collection_id, merged["last_focus"], merged["open_questions"], merged["next_step"], _now()),
            )

    def get_progress(self, collection_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM learning_progress WHERE collection_id = ?", (collection_id,)).fetchone()
        if row is None:
            return {"collection_id": collection_id, "last_focus": "", "open_questions": "", "next_step": "", "updated_at": ""}
        return dict(row)

    # ---------------------------------------------------------- memory candidates
    def add_candidate(self, session_id: str, kind: str, key: str, value: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memory_candidates (session_id, kind, key, value, status, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?)",
                (session_id, kind, key, value, _now()),
            )
            return int(cursor.lastrowid)

    def list_candidates(self, session_id: str | None = None, status: str = "pending") -> list[dict]:
        query = "SELECT * FROM memory_candidates WHERE status = ?"
        params: tuple = (status,)
        if session_id:
            query += " AND session_id = ?"
            params = (status, session_id)
        with self._connect() as connection:
            rows = connection.execute(query + " ORDER BY id", params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------- memory entries (FE-3: editable)
    def list_memory_entries(self, status: str = "confirmed") -> list[dict]:
        """Long-term memory rows by status (FE-3).

        "记忆条目" is just the confirmed view of ``memory_candidates`` -- the same rows
        the candidate flow writes -- so editing one is a plain UPDATE and the feature
        needs no new table.
        """

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_candidates WHERE status = ? ORDER BY id", (status,)
            ).fetchall()
        return [dict(row) for row in rows]

    def update_memory_entry(self, entry_id: int, *, key: str | None = None, value: str | None = None) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memory_candidates WHERE id = ?", (entry_id,)).fetchone()
            if row is None:
                raise KeyError(entry_id)
            connection.execute(
                "UPDATE memory_candidates SET key = ?, value = ? WHERE id = ?",
                (key if key is not None else row["key"], value if value is not None else row["value"], entry_id),
            )
            updated = connection.execute("SELECT * FROM memory_candidates WHERE id = ?", (entry_id,)).fetchone()
        return dict(updated)

    def delete_memory_entry(self, entry_id: int) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memory_candidates WHERE id = ?", (entry_id,))
        return int(cursor.rowcount or 0)

    def delete_preference(self, key: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM user_preferences WHERE key = ?", (key,))
        return int(cursor.rowcount or 0)

    def mark_candidates(self, candidate_ids: Sequence[int], status: str) -> int:
        if not candidate_ids:
            return 0
        placeholders = ",".join("?" for _ in candidate_ids)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE memory_candidates SET status = ? WHERE id IN ({placeholders}) AND status = 'pending'",
                (status, *candidate_ids),
            )
            return int(cursor.rowcount or 0)
