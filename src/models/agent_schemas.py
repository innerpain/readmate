"""HTTP schemas for the ReadMate agent routes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    collection_id: str | None = None
    # R7 (D-5b): per-message multi-collection scope.  When present it overrides
    # ``collection_id`` for this turn only -- it is never persisted to the
    # sessions row, so a chat that mixes collections still keeps a stable home
    # collection in history.
    collection_ids: list[str] | None = None
    mode: str | None = None


class AgentCitation(BaseModel):
    chunk_id: str
    document_id: str = ""
    filename: str = ""
    page: int | None = None
    # A3: last page the cited passage covers.  It has to be declared here as well
    # as on the runtime dataclass -- pydantic drops undeclared fields, which kept
    # the span from ever reaching the UI.
    page_end: int | None = None
    quote: str = ""


class AgentChatResponse(BaseModel):
    session_id: str
    mode: str
    answer: str
    citations: list[AgentCitation] = Field(default_factory=list)
    non_source: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tool_trace: list[dict] = Field(default_factory=list)
    gate: dict = Field(default_factory=dict)
    refused: bool = False
    failure: str | None = None
    # Additive instrumentation for the evaluation layers: one entry per ReAct
    # round (calls, duration, was it closable) and the chunk_ids the model cited
    # that no tool observed this turn. The `gate` dict stays exactly as it was.
    rounds: list[dict] = Field(default_factory=list)
    dropped_citations: list[str] = Field(default_factory=list)


class CollectionCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    document_ids: list[str] = Field(default_factory=list)


class CollectionDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1)


class SessionCreateRequest(BaseModel):
    collection_id: str | None = None
    mode: str | None = None


class SessionTitleRequest(BaseModel):
    """C4: rename a session.  Bounded so a pasted paragraph cannot become a title."""

    title: str = Field(min_length=1, max_length=80)


class MemoryConfirmRequest(BaseModel):
    session_id: str
    candidate_ids: list[int] | None = None


class MemoryRejectRequest(BaseModel):
    session_id: str
    candidate_ids: list[int] | None = None


class ReadRequest(BaseModel):
    chunk_id: str | None = None
    document_id: str | None = None
    page: int | None = None
    element_id: str | None = None


class DocumentMetaRequest(BaseModel):
    """FE-2: rename (alias) and/or switch a document off.

    Both fields are optional so the same route serves 改名 and 启用/取消; a ``None``
    means "leave it as it is", which is what makes a partial PATCH safe.
    """

    alias: str | None = Field(default=None, max_length=120)
    enabled: bool | None = None


class CollectionRenameRequest(BaseModel):
    """FE-2: rename one collection (分区改名)."""

    name: str = Field(min_length=1, max_length=120)


class MemoryEntryRequest(BaseModel):
    """FE-3: edit one long-term memory entry."""

    key: str | None = Field(default=None, max_length=120)
    value: str | None = Field(default=None, max_length=2000)


class ProfileRequest(BaseModel):
    """FE-3: edit the learner profile (identity / major / goal)."""

    identity: str | None = Field(default=None, max_length=200)
    major: str | None = Field(default=None, max_length=200)
    goal: str | None = Field(default=None, max_length=500)


class PreferenceRequest(BaseModel):
    """FE-3: set one preference value."""

    value: str = Field(max_length=2000)


class ProgressRequest(BaseModel):
    """FE-3: edit the learning-progress record of one collection."""

    last_focus: str | None = Field(default=None, max_length=500)
    open_questions: str | None = Field(default=None, max_length=1000)
    next_step: str | None = Field(default=None, max_length=500)
