"""Long-term memory: small, explicit, and never written silently.

ReadMate-Agent 5.2: the model may only propose; the user (or an explicit
confirm call) writes.  Document text never enters these tables.
"""

from __future__ import annotations

from src.adapter.contracts import INVALID_ARGUMENTS, AdapterError
from src.agent.collections import CollectionService

MAX_DIGEST_CHARS = 600
MAX_VALUE_CHARS = 500

MEMORY_KINDS = frozenset({"profile", "preference", "progress"})


class MemoryService:
    def __init__(self, *, db, collections: CollectionService, settings) -> None:
        self.db = db
        self.collections = collections
        self.settings = settings

    def digest(self, collection_id: str | None = None, collection_ids: list[str] | None = None) -> str:
        """Compact always-injected context: profile + preferences + progress.

        R7 / D-6b: per-collection ``progress`` is only meaningful for a single
        collection, so it is injected **only** when the turn's scope resolves to
        exactly one collection.  A multi-collection turn still gets the shared
        profile/preferences but no progress line (which collection's progress
        would it be?).  ``collection_ids`` is the effective scope; ``collection_id``
        stays for the existing single-collection callers.
        """

        profile = self.db.get_profile()
        parts: list[str] = []
        identity = " / ".join(value for value in (profile.get("identity"), profile.get("major"), profile.get("goal")) if value)
        if identity:
            parts.append(f"Profile: {identity}")
        preferences = self.db.get_preferences()
        if preferences:
            parts.append("Preferences: " + "; ".join(f"{key}={value}" for key, value in preferences.items()))
        scope = [cid for cid in (collection_ids or ([collection_id] if collection_id else [])) if cid]
        if len(scope) == 1:
            progress = self.db.get_progress(scope[0])
            line = " | ".join(
                f"{label}: {progress[field]}"
                for label, field in (("last focus", "last_focus"), ("open questions", "open_questions"), ("next step", "next_step"))
                if progress.get(field)
            )
            if line:
                parts.append(f"Progress: {line}")
        return "\n".join(parts)[: self.settings.digest_chars]

    def note(self, *, session_id: str, kind: str, key: str, value: str) -> int:
        """Record a pending candidate.  Nothing durable changes yet.

        An unrecognised ``kind`` used to fall through silently to ``"profile"``
        and end up in the ``goal`` column; the caller now gets a stable
        ``invalid_arguments`` error code so the mistake surfaces instead of
        quietly polluting the learner profile.
        """

        if kind not in MEMORY_KINDS:
            raise AdapterError(INVALID_ARGUMENTS, f"unsupported memory kind: {kind!r}")
        return self.db.add_candidate(session_id, kind, key.strip()[:120], value.strip()[:MAX_VALUE_CHARS])

    def note_confirmed(
        self,
        *,
        session_id: str,
        kind: str,
        key: str,
        value: str,
        collection_id: str | None = None,
    ) -> dict:
        """User-explicit memory request: write through to the long-term tables
        **immediately** (2026-09-19, M1=c / M3=a).

        The candidate row is still created -- it is simply born ``confirmed``
        instead of ``pending`` -- so the write stays auditable and deletable
        from the memory page (M3).  ``progress`` needs a bound collection to
        land: without one the candidate deliberately stays *pending* and the
        caller is told, instead of pretending a durable write happened (the
        same honesty rule as ``confirm_with_details``'s landed/marked split).
        """

        candidate_id = self.note(session_id=session_id, kind=kind, key=key, value=value)
        if kind == "progress" and not collection_id:
            return {"candidate_id": candidate_id, "written": False, "landed": 0, "reason": "progress_needs_collection"}
        details = self.confirm_with_details(session_id=session_id, candidate_ids=[candidate_id], collection_id=collection_id)
        return {
            "candidate_id": candidate_id,
            "written": bool(details.get("landed")),
            "landed": int(details.get("landed") or 0),
        }

    def pending(self, session_id: str) -> list[dict]:
        return self.db.list_candidates(session_id, status="pending")

    def confirm(self, *, session_id: str, candidate_ids: list[int] | None = None, collection_id: str | None = None) -> int:
        """Promote pending candidates into the long-term tables.  Kept as the
        integer-marked-count API for existing callers (smoke / eval)."""

        return int(self.confirm_with_details(session_id=session_id, candidate_ids=candidate_ids, collection_id=collection_id)["marked"])

    def confirm_with_details(
        self,
        *,
        session_id: str,
        candidate_ids: list[int] | None = None,
        collection_id: str | None = None,
    ) -> dict:
        """Same promotion, but reports landed vs marked separately so a candidate
        that only got its status flipped (e.g. ``kind=progress`` without a
        bound collection) no longer looks like a durable write."""

        candidates = [
            item for item in self.pending(session_id)
            if candidate_ids is None or int(item["id"]) in set(candidate_ids)
        ]
        if not candidates:
            return {"marked": 0, "landed": 0}
        landed = 0
        for item in candidates:
            kind, key, value = str(item["kind"]), str(item["key"]), str(item["value"])
            if kind == "preference":
                self.db.set_preference(key, value)
                landed += 1
            elif kind == "progress":
                if not collection_id:
                    continue
                if key in {"last_focus", "open_questions", "next_step"}:
                    self.db.set_progress(collection_id, **{key: value})
                    landed += 1
            elif kind == "profile":
                field = key if key in {"identity", "major", "goal"} else "goal"
                self.db.set_profile(**{field: value})
                landed += 1
        marked = self.db.mark_candidates([int(item["id"]) for item in candidates], "confirmed")
        return {"marked": int(marked), "landed": landed}

    def reject(self, *, session_id: str, candidate_ids: list[int] | None = None) -> int:
        candidates = [
            item for item in self.pending(session_id)
            if candidate_ids is None or int(item["id"]) in set(candidate_ids)
        ]
        return self.db.mark_candidates([int(item["id"]) for item in candidates], "rejected")
