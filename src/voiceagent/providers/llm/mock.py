"""A scripted LLM so the whole pipeline can be tested without a network.

Each entry in ``script`` is one assistant turn:

* ``"text"``                → stream that text back
* ``[ToolCall(...), ...]``  → request those tools
* a callable               → whatever it returns (str or list of ToolCall)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..base import LLMDelta, Message, ToolCall

Scripted = str | list[ToolCall] | Callable[[list[Message]], Any]

CHUNK = 3  # characters per streamed frame, to exercise the sentence splitter


class MockLLM:
    name = "mock"

    def __init__(self, script: list[Scripted] | None = None, default: str = "好的。") -> None:
        self.script: list[Scripted] = list(script or [])
        self.default = default
        self.calls: list[list[Message]] = []

    def chat_stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMDelta]:
        self.calls.append(list(messages))
        return self._stream()

    async def _stream(self) -> AsyncIterator[LLMDelta]:
        item: Scripted = self.script.pop(0) if self.script else self.default
        if callable(item):
            item = item(self.calls[-1])
        if isinstance(item, str):
            for i in range(0, len(item), CHUNK):
                yield LLMDelta(content=item[i : i + CHUNK])
                await asyncio.sleep(0)
            yield LLMDelta(finish_reason="stop")
        else:
            yield LLMDelta(tool_calls=list(item), finish_reason="tool_calls")

    async def aclose(self) -> None:
        return None


__all__ = ["MockLLM"]
