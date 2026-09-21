"""One client per model profile.  Endpoint differences are data, not code."""

from __future__ import annotations

import json

from src.llm.types import LLMStep, ModelProfile, ToolCall


class LLMClient:
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as error:  # pragma: no cover - dependency guard
                raise RuntimeError("openai package is required for a real model") from error
            kwargs: dict[str, object] = {
                "api_key": self.profile.api_key or "not-needed",
                "timeout": self.profile.timeout_s,
                "max_retries": self.profile.max_retries,
            }
            if self.profile.base_url:
                kwargs["base_url"] = self.profile.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def request_kwargs(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        """Pure function of the profile: unit-testable without any network."""

        profile = self.profile
        kwargs: dict[str, object] = {
            "model": profile.model,
            "messages": messages,
            "temperature": profile.temperature if temperature is None else temperature,
            profile.token_param: profile.max_tokens if max_tokens is None else max_tokens,
        }
        if tools and profile.supports_tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = profile.tool_choice
        extra_body = dict(profile.extra_body)
        if profile.thinking == "extra_body":
            extra_body[profile.thinking_key] = profile.thinking_value
        elif profile.thinking == "reasoning_effort":
            kwargs["reasoning_effort"] = profile.reasoning_effort or "low"
        if extra_body:
            kwargs["extra_body"] = extra_body
        if profile.extra_headers:
            kwargs["extra_headers"] = dict(profile.extra_headers)
        return kwargs

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> LLMStep:
        kwargs = self.request_kwargs(
            messages, temperature=temperature, max_tokens=max_tokens, tools=tools
        )
        response = self._get_client().chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            raw = getattr(function, "arguments", "") or "{}"
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(
                    id=str(getattr(call, "id", "")),
                    name=str(getattr(function, "name", "") or ""),
                    arguments=arguments,
                )
            )
        return LLMStep(content=message.content or "", tool_calls=calls)

    def generate(self, prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        """Single-prompt convenience for a text-only caller.

        ``QueryPlanner`` speaks this one-method protocol (it predates the neutral
        model layer).  Without it the planner silently degraded to its
        deterministic fallback, so the ``planned`` retrieval track never produced
        the English rendering it exists for -- found 2026-09-19 while verifying
        A8, where the missing rendering was the difference between finding a
        section number and reporting it absent.
        """

        step = self.complete(
            [{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens
        )
        return getattr(step, "content", "") or ""

    def stream_complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        on_delta=None,
    ) -> LLMStep:
        """``complete()`` over a streamed response (frontend plan §5).

        Text deltas stream out live via ``on_delta`` (optimistic streaming:
        if the round later turns out to carry tool_calls, the caller emits
        ``answer_reset``).  tool_calls arrive incrementally per OpenAI
        chunking rules: index-keyed buffer, ``id``/``name`` on the first
        fragment of each index, ``arguments`` concatenated.
        """

        kwargs = self.request_kwargs(
            messages, temperature=temperature, max_tokens=max_tokens, tools=tools
        )
        kwargs["stream"] = True
        stream = self._get_client().chat.completions.create(**kwargs)
        content_parts: list[str] = []
        calls_by_index: dict[int, dict] = {}
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                content_parts.append(piece)
                if on_delta is not None:
                    on_delta(piece)
            for call in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(call, "index", 0) or 0)
                slot = calls_by_index.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if getattr(call, "id", None):
                    slot["id"] = str(call.id)
                function = getattr(call, "function", None)
                if function is not None:
                    if getattr(function, "name", None):
                        slot["name"] = str(function.name)
                    if getattr(function, "arguments", None):
                        slot["arguments"] += str(function.arguments)
        calls: list[ToolCall] = []
        for index in sorted(calls_by_index):
            raw = calls_by_index[index]["arguments"] or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(
                    id=calls_by_index[index]["id"],
                    name=calls_by_index[index]["name"],
                    arguments=arguments,
                )
            )
        return LLMStep(content="".join(content_parts), tool_calls=calls)
