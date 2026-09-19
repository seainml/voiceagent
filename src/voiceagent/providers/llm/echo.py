"""The zero-configuration LLM.

Nothing here is intelligent: it exists so that a fresh clone with no API key
still gives you a *complete* voice loop to verify — microphone, VAD,
end-of-turn detection, TTS, barge-in, the UI, all of it. Only the reasoning is
missing, and it says so out loud rather than failing silently.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ..base import LLMDelta, Message, ToolCall

NO_MODEL_HINT = (
    "我听到了你说：{heard}。不过我现在还没有连接大模型，"
    "请设置 DEEPSEEK_API_KEY 环境变量，或者在 voiceagent.toml 里配置 LLM。"
)


class EchoLLM:
    name = "echo"

    def __init__(self, hint: str = NO_MODEL_HINT) -> None:
        self.hint = hint

    def chat_stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMDelta]:
        return self._stream(messages)

    async def _stream(self, messages: list[Message]) -> AsyncIterator[LLMDelta]:
        heard = ""
        for message in reversed(messages):
            if message.role == "user":
                heard = message.content
                break
        text = self.hint.format(heard=heard[:120] or "（空）")
        for i in range(0, len(text), 4):
            yield LLMDelta(content=text[i : i + 4])
            await asyncio.sleep(0)
        yield LLMDelta(finish_reason="stop")

    async def aclose(self) -> None:
        return None


__all__ = ["EchoLLM", "ToolCall"]
