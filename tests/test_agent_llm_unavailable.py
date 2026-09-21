"""C10: an upstream model failure must not surface as a bare 500.

Measured before the fix: with the provider quota exhausted, a plain
``POST /agent/chat`` answered ``HTTP Error 500`` with no body the UI could key
on, while the SSE twin degrades to a structured ``error`` event.  The sync route
now answers 503 ``llm_unavailable`` with a message a user can act on, and it must
keep passing through the statuses the route itself raises (400/404/409).
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# ``src/api/__init__.py`` re-exports ``app`` (the FastAPI instance), which shadows
# the submodule name, so the module has to be pulled out by importlib.
api_app = importlib.import_module("src.api.app")
agent_routes = importlib.import_module("src.api.agent_routes")


def _client(monkeypatch: pytest.MonkeyPatch, run) -> TestClient:
    def _build_runtime():
        db = SimpleNamespace(create_session=lambda **kwargs: "ses_test", list_collections=lambda: [])
        return SimpleNamespace(run=run), db, None, None

    monkeypatch.setattr(agent_routes, "build_runtime", _build_runtime)
    return TestClient(api_app.app)


def test_an_upstream_failure_answers_503_llm_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider refused the call (quota exhausted / network) -- the client
    must get a code and a sentence, never an anonymous 500."""

    def _run(*args, **kwargs):
        raise RuntimeError("Error code: 429 - insufficient_quota")

    client = _client(monkeypatch, _run)
    resp = client.post("/agent/chat", json={"message": "你好", "mode": "deep"})

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["code"] == "llm_unavailable"
    assert detail["message"]
    assert "429" in detail["reason"]


def test_the_turn_keeps_the_status_the_route_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """The catch-all must not swallow an HTTPException raised inside the turn --
    a 409 contract error has to stay a 409."""

    def _run(*args, **kwargs):
        raise HTTPException(status_code=409, detail={"code": "knowledge_base_empty", "message": "no docs"})

    client = _client(monkeypatch, _run)
    resp = client.post("/agent/chat", json={"message": "你好", "mode": "deep"})

    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "knowledge_base_empty"


def test_a_turn_that_succeeds_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against the new handler masking the normal path."""

    result = SimpleNamespace(
        answer="好的",
        mode="deep",
        citations=[],
        non_source=[],
        warnings=[],
        failure=None,
        rounds=[],
        gate={"searched": False},
        tool_trace=[],
        dropped_citations=[],
    )

    client = _client(monkeypatch, lambda *args, **kwargs: result)
    resp = client.post("/agent/chat", json={"message": "你好", "mode": "deep"})

    assert resp.status_code == 200
    assert resp.json()["answer"] == "好的"