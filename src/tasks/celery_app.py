"""Celery application configuration for local asynchronous jobs."""

import os

from celery import Celery


celery_app = Celery(
    "document_qa_assistant",
    broker=os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/1"),
    include=["src.tasks.ingestion", "src.tasks.summary"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
