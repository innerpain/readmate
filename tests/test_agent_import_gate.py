"""Layering gates: agent must not reach into the retrieval stack.

These are the executable form of AD1/A7 from readmate/RAG接入Agent方案.md 10.
They read source text on purpose: the rule is about imports, and an import that
is legal today becomes a coupling bug the moment the module boundaries move.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

AGENT_FORBIDDEN = re.compile(r"chroma_store|index_builder|persistent_retriever|document_registry|retrieval\.")
ADAPTER_FORBIDDEN = re.compile(r"\bsrc\.agent\b")
NEUTRAL_FORBIDDEN = re.compile(r"\bsrc\.(agent|adapter|retrieval)\b")


def _offenders(package: str, pattern: re.Pattern[str]) -> list[str]:
    found: list[str] = []
    for path in sorted((SRC / package).rglob("*.py")):
        if pattern.search(path.read_text(encoding="utf-8")):
            found.append(str(path.relative_to(SRC)))
    return found


def test_agent_layer_never_names_the_retrieval_stack():
    assert _offenders("agent", AGENT_FORBIDDEN) == []


def test_adapter_layer_never_imports_the_agent_layer():
    assert _offenders("adapter", ADAPTER_FORBIDDEN) == []


def test_neutral_model_layer_stays_free_of_business_imports():
    assert _offenders("llm", NEUTRAL_FORBIDDEN) == []
