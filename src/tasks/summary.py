"""Async session summarisation (批 D 阶段 2 / D32).

The task is deliberately thin: planning, prompting and storage all live in
:mod:`src.agent.summary`, so the same code path serves the background job and the
manual ``POST /agent/sessions/{id}/compact`` route.  What lives *here* is the
failure budget -- a summary that keeps failing must stop costing a call per turn.

Why a background task at all: the summary is only needed by the *next* turn, so
paying for it inside the user's turn would add latency for nothing.  The turn
returns as soon as the answer is stored; the window is compressed afterwards.
"""

from __future__ import annotations

import logging

from src.agent.summary import MAX_SUMMARY_FAILURES, SessionSummaryService
from src.config.settings import get_agent_settings
from src.llm import LLMClient, ModelRegistry
from src.storage.agent_db import AgentDB

from .celery_app import celery_app

logger = logging.getLogger(__name__)


def build_summary_service() -> tuple[SessionSummaryService, AgentDB]:
    """The production wiring: agent settings -> db + ``summarizer`` role model."""

    settings = get_agent_settings()
    db = AgentDB(settings.db_path or None)
    registry = ModelRegistry.load(settings.models_config_path or None)
    return SessionSummaryService(db=db, llm=LLMClient(registry.resolve("summarizer"))), db


def run_summary(
    session_id: str,
    *,
    force: bool = False,
    service: SessionSummaryService | None = None,
    db: AgentDB | None = None,
) -> dict:
    """Summarise one session under the failure budget.  Never raises.

    ``force=True`` (the manual route) ignores the breaker: a human asking to
    compress right now has context the counter does not.
    """

    if service is None or db is None:
        service, db = build_summary_service()

    failures = db.summary_failures(session_id)
    if failures >= MAX_SUMMARY_FAILURES and not force:
        logger.warning(
            "summary breaker open for %s (%d consecutive failures); skipping",
            session_id,
            failures,
        )
        return {
            "stored": False,
            "reason": "breaker_open",
            "failures": failures,
            "upto_message_id": None,
        }

    outcome = service.summarize(session_id)
    if outcome.get("stored"):
        db.reset_summary_failures(session_id)
        outcome["failures"] = 0
        return outcome

    if outcome.get("reason") in {"nothing_to_summarize"}:
        # Not a failure: there was simply nothing new to fold in.
        outcome["failures"] = failures
        return outcome

    outcome["failures"] = db.bump_summary_failure(session_id)
    return outcome


@celery_app.task(bind=True, name="document_qa.summarize_session", max_retries=0)
def summarize_session_task(self, session_id: str) -> dict:
    """Celery entry point.  Reports the outcome instead of raising: a failed
    summary must never mark the document/turn pipeline as broken."""

    try:
        result = run_summary(session_id)
    except Exception as error:  # noqa: BLE001 - task must not raise
        logger.exception("summary task crashed for %s", session_id)
        return {"stored": False, "reason": "task_error", "error": str(error)}
    logger.info("summary task %s -> %s", session_id, result.get("reason"))
    return result
