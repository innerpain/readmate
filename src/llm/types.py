"""Shared value types for every model client in this project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMStep:
    """One assistant turn: tool calls, content, or both."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ModelProfile:
    """Everything a model needs, described as data instead of code."""

    name: str
    model: str
    base_url: str = ""
    api_key: str = ""
    provider: str = "openai-compatible"
    temperature: float = 0.0
    max_tokens: int = 4096
    token_param: str = "max_tokens"
    supports_tools: bool = True
    tool_choice: str = "auto"
    thinking: str = "none"
    thinking_key: str = "enable_thinking"
    thinking_value: object = False
    reasoning_effort: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, object] = field(default_factory=dict)
    timeout_s: float = 120.0
    max_retries: int = 1
    # D32 前置 (2026-09-20): how many tokens the model's window holds.  ReadMate
    # had no idea -- every budget was a guess -- so the context-compression work
    # needs it declared per profile (``context_length`` in models.toml, with
    # ``MODEL_CONTEXT_LENGTH`` as the env override).
    context_length: int = 32768


class ToolLLM(Protocol):
    """What the agent runtime needs from a model: one completion with tools."""

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> LLMStep: ...
