"""C7: ``/tasks/{id}`` must not answer "pending" for an id that was never queued.

Celery reports an unknown id as PENDING because its result backend simply has no
key for it, so the endpoint used to claim a nonexistent task was waiting.  The
route now 404s ids this process never queued and no registry record claims, and
keeps answering normally for ids it did queue.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import importlib

# ``src/api/__init__.py`` re-exports ``app`` (the FastAPI instance), which
# shadows the submodule name, so ``import src.api.app as api_app`` would hand
# back the application object and monkeypatching its module-level helpers
# (AsyncResult, get_document_registry) would hit the wrong thing.
api_app = importlib.import_module("src.api.app")


class _FakeAsyncResult:
    """Celery's view of a task nobody ever queued: PENDING, no result."""

    def __init__(self, task_id: str, app=None) -> None:
        self.id = task_id
        self.state = "PENDING"
        self.result = None


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api_app, "AsyncResult", _FakeAsyncResult)
    # Deterministic: no registry record can claim the ids used below.
    monkeypatch.setattr(api_app, "get_document_registry", lambda: SimpleNamespace(list_documents=lambda: []))
    return TestClient(api_app.app)


def test_unknown_task_is_404(client: TestClient) -> None:
    resp = client.get("/tasks/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "task_not_found"


def test_queued_task_is_still_reported_as_pending(client: TestClient) -> None:
    """A task the API actually queued must keep its 200 + pending answer -- the
    ledger exists to remove false positives, not to hide real pending work."""

    task_id = api_app._remember_task("11111111-1111-1111-1111-111111111111")
    resp = client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["status"] == "pending"


def test_registry_claiming_the_task_avoids_a_false_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rebuild task is not written into the registry, but a document record may
    claim an id (uploads do); such an id must never 404."""

    claimed = "22222222-2222-2222-2222-222222222222"
    record = SimpleNamespace(task_id=claimed)
    monkeypatch.setattr(
        api_app, "get_document_registry", lambda: SimpleNamespace(list_documents=lambda: [record])
    )
    resp = client.get(f"/tasks/{claimed}")
    assert resp.status_code == 200


def test_unreadable_registry_prefers_helpfulness(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the registry cannot be read we cannot rule the id out, and the failure
    direction must be "too helpful", never a wrong 404."""

    from src.storage.document_registry import RegistryDataError

    def _boom():
        raise RegistryDataError("unreadable")

    monkeypatch.setattr(api_app, "get_document_registry", _boom)
    resp = client.get("/tasks/33333333-3333-3333-3333-333333333333")
    assert resp.status_code == 200