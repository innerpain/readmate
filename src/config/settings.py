"""Centralized, environment-backed settings for the RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.retrieval.embedding_generator import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_TOKEN_BUDGET,
    DEFAULT_QUERY_INSTRUCTION,
)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error


def _env_value(name: str, default: str = "") -> str:
    """Read the canonical key, with a lowercase .env compatibility fallback."""

    value = os.getenv(name)
    if value is None or not value.strip():
        value = os.getenv(name.lower(), default)
    return value.strip() if value else default


@dataclass(frozen=True)
class ModelSettings:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int | None = None
    normalize_embeddings: bool = True
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_max_seq_length: int = DEFAULT_EMBEDDING_MAX_SEQ_LENGTH
    embedding_token_budget: int = DEFAULT_EMBEDDING_TOKEN_BUDGET
    embedding_query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 512

    @classmethod
    def from_env(cls) -> "ModelSettings":
        dimension = os.getenv("EMBEDDING_DIMENSION", "").strip()
        return cls(
            embedding_model=_env_value("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_dimension=int(dimension) if dimension else None,
            normalize_embeddings=os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() in {"1", "true", "yes"},
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),
            embedding_max_seq_length=_env_int("EMBEDDING_MAX_SEQ_LENGTH", DEFAULT_EMBEDDING_MAX_SEQ_LENGTH),
            embedding_token_budget=_env_int("EMBEDDING_TOKEN_BUDGET", DEFAULT_EMBEDDING_TOKEN_BUDGET),
            embedding_query_instruction=_env_instruction(),
            llm_model=_env_value("MODEL"),
            llm_base_url=_env_value("BASE_URL"),
            llm_api_key=_env_value("API_KEY"),
            temperature=_env_float("TEMPERATURE", 0.0),
            max_tokens=_env_int("MAX_TOKENS", 512),
        )

    def validate(self) -> None:
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if self.embedding_dimension is not None and self.embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        if self.embedding_max_seq_length < 1:
            raise ValueError("embedding_max_seq_length must be positive")
        if self.embedding_token_budget < 1:
            raise ValueError("embedding_token_budget must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 5
    min_score: float = 0.45
    excerpt_chars: int = 300
    # Multi-channel retrieval: how wide each query searches before fusion, and
    # how many slots a single sub-query is guaranteed in the final selection.
    # D30 (2026-09-20): 40 -> 15.  Measured on the 20-question probe: pool 15
    # scored recall@5 1.000 / MRR 0.8451 at 0.79s per question, versus pool 40's
    # 0.9412 / 0.8333 at 5.10s (-84% retrieval time, metrics up).
    candidate_k: int = 15
    final_k: int = 5
    rrf_k: int = 60
    quota_per_query: int = 1
    # Post-fusion cross-encoder: reorder a wide candidate pool, then cut to final_k.
    # Dense cosine scores remain the gate values; rerank only changes selection order.
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 512
    # Weak-tail cutoff applied inside diversify: drop hits scoring below
    # top_score * ratio.  Raised out of the hardcoded 0.82 so it can be tuned
    # against the global min_score gate (A4) without a code change.
    score_floor_ratio: float = 0.82
    # D27 (2026-09-20): the four window/validation constants that used to be
    # module-level in ``persistent_retriever`` are settings now.
    max_query_chars: int = 2000
    max_query_texts: int = 8
    # A collection filter can empty the final window; widen the raw window so
    # diversify still has candidates to refill from.
    filtered_window_multiplier: int = 4
    max_filtered_window: int = 200

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        return cls(
            top_k=_env_int("RETRIEVAL_TOP_K", 5),
            min_score=_env_float("RETRIEVAL_MIN_SCORE", 0.45),
            excerpt_chars=_env_int("RETRIEVAL_EXCERPT_CHARS", 300),
            candidate_k=_env_int("RETRIEVAL_CANDIDATE_K", 15),
            final_k=_env_int("RETRIEVAL_FINAL_K", _env_int("RETRIEVAL_TOP_K", 5)),
            rrf_k=_env_int("RETRIEVAL_RRF_K", 60),
            quota_per_query=_env_int("RETRIEVAL_QUOTA_PER_QUERY", 1),
            rerank_enabled=_env_flag("RERANK_ENABLED", True),
            rerank_model=_env_value("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            rerank_max_length=_env_int("RERANK_MAX_LENGTH", 512),
            score_floor_ratio=_env_float("RETRIEVAL_SCORE_FLOOR_RATIO", 0.82),
            max_query_chars=_env_int("RETRIEVAL_MAX_QUERY_CHARS", 2000),
            max_query_texts=_env_int("RETRIEVAL_MAX_QUERY_TEXTS", 8),
            filtered_window_multiplier=_env_int("RETRIEVAL_FILTERED_WINDOW_MULTIPLIER", 4),
            max_filtered_window=_env_int("RETRIEVAL_MAX_FILTERED_WINDOW", 200),
        )

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("retrieval counts must be positive")
        if self.excerpt_chars < 1:
            raise ValueError("excerpt_chars must be positive")
        if self.candidate_k < 1 or self.final_k < 1:
            raise ValueError("retrieval windows must be positive")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if self.max_query_chars < 1 or self.max_query_texts < 1:
            raise ValueError("query limits must be positive")
        if self.filtered_window_multiplier < 1 or self.max_filtered_window < 1:
            raise ValueError("filtered window limits must be positive")
        if self.quota_per_query < 0:
            raise ValueError("quota_per_query must not be negative")
        if self.rerank_max_length < 8:
            raise ValueError("rerank_max_length must be >= 8")
        if self.rerank_enabled and not self.rerank_model.strip():
            raise ValueError("rerank_model must be non-empty when rerank is enabled")
        for name in ("min_score",):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")
        if not 0.0 <= self.score_floor_ratio <= 1.0:
            raise ValueError("score_floor_ratio must be between 0 and 1")


@dataclass(frozen=True)
class QueryPlanningSettings:
    """Query-side switches for Chinese-question / English-document retrieval."""

    enabled: bool = True
    llm_enabled: bool = True
    max_subqueries: int = 4

    @classmethod
    def from_env(cls) -> "QueryPlanningSettings":
        return cls(
            enabled=_env_flag("QUERY_PLANNING_ENABLED", True),
            llm_enabled=_env_flag("QUERY_PLANNING_LLM_ENABLED", True),
            max_subqueries=_env_int("QUERY_PLANNING_MAX_SUBQUERIES", 4),
        )

    def validate(self) -> None:
        if self.max_subqueries < 1:
            raise ValueError("max_subqueries must be positive")


@dataclass(frozen=True)
class RouteSettings:
    """D27/D37 (2026-09-20): the router's thresholds, out of the module.

    They used to be module-level constants in ``adapter/route_planner.py``, so
    tuning the router meant editing code.
    """

    min_clause_chars: int = 4
    long_question_chars: int = 40
    max_subqueries: int = 4
    cross_language_ratio: float = 0.30

    @classmethod
    def from_env(cls) -> "RouteSettings":
        return cls(
            min_clause_chars=_env_int("ROUTE_MIN_CLAUSE_CHARS", 4),
            long_question_chars=_env_int("ROUTE_LONG_QUESTION_CHARS", 40),
            max_subqueries=_env_int("ROUTE_MAX_SUBQUERIES", 4),
            cross_language_ratio=_env_float("ROUTE_CROSS_LANGUAGE_RATIO", 0.30),
        )

    def validate(self) -> None:
        if self.min_clause_chars < 1 or self.long_question_chars < 1 or self.max_subqueries < 1:
            raise ValueError("route thresholds must be positive")
        if not 0.0 <= self.cross_language_ratio <= 1.0:
            raise ValueError("cross_language_ratio must be within [0, 1]")


def get_route_settings() -> RouteSettings:
    settings = RouteSettings.from_env()
    settings.validate()
    return settings


def get_query_planning_settings() -> QueryPlanningSettings:
    settings = QueryPlanningSettings.from_env()
    settings.validate()
    return settings


def _env_instruction() -> str:
    """Empty EMBEDDING_QUERY_INSTRUCTION explicitly disables the query prefix."""

    raw = os.getenv("EMBEDDING_QUERY_INSTRUCTION")
    if raw is None:
        raw = os.getenv("embedding_query_instruction")
    return raw.strip() if raw is not None else DEFAULT_QUERY_INSTRUCTION


def get_model_settings() -> ModelSettings:
    settings = ModelSettings.from_env()
    settings.validate()
    return settings


def get_retrieval_settings() -> RetrievalSettings:
    settings = RetrievalSettings.from_env()
    settings.validate()
    return settings


@dataclass(frozen=True)
class IngestionSettings:
    """Docling parse-stage switches recorded beside ordered_document_v1."""

    # D1: how long a rebuild lock may live before another attempt may take it
    # over.  A worker killed mid-build used to leave the lock forever, and every
    # later rebuild answered ``index_build_busy`` until someone deleted the file
    # by hand.
    index_build_lock_ttl_s: int = 3600

    @classmethod
    def from_env(cls) -> "IngestionSettings":
        return cls(
            index_build_lock_ttl_s=_env_int("INDEX_BUILD_LOCK_TTL_S", 3600),
        )

    def validate(self) -> None:
        if self.index_build_lock_ttl_s < 60:
            raise ValueError("index_build_lock_ttl_s must be at least 60 seconds")


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes"}


def get_ingestion_settings() -> IngestionSettings:
    settings = IngestionSettings.from_env()
    settings.validate()
    return settings

@dataclass(frozen=True)
class AgentSettings:
    """ReadMate agent switches (same env style as the RAG settings above)."""

    db_path: str = ""
    default_mode: str = "deep"
    max_steps: int = 6
    max_gate_blocks: int = 2
    history_turns: int = 6
    # D32 (2026-09-20): conversation history is admitted by *token* budget, not by
    # turn count.  The budget is ``model context_length * this ratio``, clamped to
    # [4k, 24k] by ``src/agent/token_budget.py``.  0.30 measured on the real prompt:
    # a 32k window admits ~40 turns of history, while the worst case (three tool
    # rounds of table-heavy evidence) still lands at 79% of the window.
    history_context_ratio: float = 0.30
    tool_text_chars: int = 1200
    # D6 (2026-09-19): a table is one evidence object whose value is in its
    # rows, so it gets its own ceiling.  The prose budget silently cut long
    # tables mid-body (measured: "the table body is cut off after the
    # Ensemble rows" for the paper's Table 2).
    tool_table_chars: int = 3000
    tool_max_hits: int = 5
    tool_primary_min_ratio: float = 0.6
    tool_neighbor_min_chars: int = 120
    # D35 (2026-09-20): the adapter's read ceilings used to be module constants
    # (DEFAULT_PAGE_CHARS / MAX_ELEMENT_TEXT_CHARS), so the page size could not be
    # tuned without editing code -- and the settings field that looked like it did
    # the job (read_page_chars) was never read by anything.
    adapter_page_chars: int = 6000
    adapter_element_chars: int = 4000
    digest_chars: int = 600
    route_mode: str = "auto"          # auto | direct | planned (see 7.5)
    models_config_path: str = ""      # default config/models.toml (see 7.6)
    # 2026-09-19 用户口径：模型**自己推断**出的偏好不要每轮都试探，改为每 N 轮做一次
    # 回顾（用户明确说"记住"的那条走自动写入，与此无关）。0 = 关闭定期回顾。
    memory_review_every_turns: int = 10
    # 批 D 阶段 2 (D32): compress the window in the background once older turns fall
    # out of it.  False = no summary job at all (no LLM call, no Celery task); the
    # window then simply keeps dropping old turns, which is the pre-D32 behaviour.
    session_summary_enabled: bool = True

    @classmethod
    def from_env(cls) -> "AgentSettings":
        return cls(
            db_path=_env_value("AGENT_DB_PATH"),
            default_mode=_env_value("AGENT_DEFAULT_MODE", "deep"),
            max_steps=_env_int("AGENT_MAX_STEPS", 6),
            max_gate_blocks=_env_int("AGENT_MAX_GATE_BLOCKS", 2),
            history_turns=_env_int("AGENT_HISTORY_TURNS", 6),
            history_context_ratio=_env_float("AGENT_HISTORY_CONTEXT_RATIO", 0.30),
            tool_text_chars=_env_int("AGENT_TOOL_TEXT_CHARS", 1200),
            tool_table_chars=_env_int("AGENT_TABLE_TEXT_CHARS", 3000),
            tool_max_hits=_env_int("AGENT_TOOL_MAX_HITS", 5),
            tool_primary_min_ratio=_env_float("AGENT_TOOL_PRIMARY_MIN_RATIO", 0.6),
            tool_neighbor_min_chars=_env_int("AGENT_TOOL_NEIGHBOR_MIN_CHARS", 120),
            adapter_page_chars=_env_int("AGENT_ADAPTER_PAGE_CHARS", 6000),
            adapter_element_chars=_env_int("AGENT_ADAPTER_ELEMENT_CHARS", 4000),
            digest_chars=_env_int("AGENT_DIGEST_CHARS", 600),
            route_mode=_env_value("AGENT_ROUTE_MODE", "auto"),
            models_config_path=_env_value("AGENT_MODELS_CONFIG"),
            memory_review_every_turns=_env_int("AGENT_MEMORY_REVIEW_EVERY_TURNS", 10),
            session_summary_enabled=_env_flag("AGENT_SESSION_SUMMARY_ENABLED", True),
        )

    def validate(self) -> None:
        if self.default_mode not in {"deep", "chat"}:
            raise ValueError("agent default_mode must be deep or chat")
        if self.route_mode not in {"auto", "direct", "planned"}:
            raise ValueError("agent route_mode must be auto, direct or planned")
        if not 0.0 < float(self.history_context_ratio) <= 1.0:
            raise ValueError("history_context_ratio must be within (0, 1]")
        for name in ("max_steps", "max_gate_blocks", "history_turns", "tool_text_chars", "tool_table_chars", "tool_max_hits", "tool_neighbor_min_chars", "digest_chars", "adapter_page_chars", "adapter_element_chars"):
            if getattr(self, name) < 1:
                raise ValueError(f"agent {name} must be positive")
        if not 0.0 < self.tool_primary_min_ratio <= 1.0:
            raise ValueError("agent tool_primary_min_ratio must be in (0, 1]")
        if self.memory_review_every_turns < 0:
            raise ValueError("agent memory_review_every_turns must be >= 0 (0 disables the periodic review)")


def get_agent_settings() -> AgentSettings:
    settings = AgentSettings.from_env()
    settings.validate()
    return settings
