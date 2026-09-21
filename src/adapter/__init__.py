"""RAG Adapter: the only layer allowed to touch the retrieval stack."""

from src.adapter.contracts import (
    ADAPTER_API_VERSION,
    AdapterError,
    DocInfo,
    ReadResult,
    REQUIRED_SEARCH_FIELDS,
    SearchHit,
)
from src.adapter.mock_adapter import MockRagAdapter
from src.adapter.protocol import RagAdapter
from src.adapter.route_planner import (
    RouteDecision,
    RouteSignals,
    RetrievalRouter,
    compute_signals,
)

__all__ = [
    "ADAPTER_API_VERSION",
    "AdapterError",
    "DocInfo",
    "ReadResult",
    "REQUIRED_SEARCH_FIELDS",
    "SearchHit",
    "RagAdapter",
    "MockRagAdapter",
    "RetrievalRouter",
    "RouteDecision",
    "RouteSignals",
    "compute_signals",
]
