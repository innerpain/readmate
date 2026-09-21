"""Scripted client for tests and the offline smoke run."""

from __future__ import annotations

from src.llm.types import LLMStep


class FakeLLMClient:
    """Replays a script and records what it was asked."""

    def __init__(self, script: list[LLMStep] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[dict] = []

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> LLMStep:
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        if self.script:
            return self.script.pop(0)
        return LLMStep(content='{"answer": "no script left", "citations": [], "non_source": []}')

    def stream_complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        on_delta=None,
    ) -> LLMStep:
        """Same script, but content leaks out in 2-chars-per-chunk pieces —
        enough to exercise the optimistic-streaming path (deltas then a
        possible reset) without a network."""

        step = self.complete(
            messages, temperature=temperature, max_tokens=max_tokens, tools=tools
        )
        if on_delta is not None:
            for start in range(0, len(step.content), 2):
                on_delta(step.content[start : start + 2])
        return step

    def generate(self, prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        """Mirror ``LLMClient.generate`` so the planner path is exercised in tests."""

        step = self.complete(
            [{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens
        )
        return getattr(step, "content", "") or ""
