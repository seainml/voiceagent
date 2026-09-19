"""Any OpenAI-compatible ``/chat/completions`` endpoint with streaming + tools.

Verified against the shapes used by OpenAI, DeepSeek, Moonshot, Zhipu,
DashScope compatible-mode, SiliconFlow, Groq, Together, Ollama, vLLM and
LM Studio — they all speak the same SSE dialect, which is why this single
adapter covers effectively the whole market.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...config import LLMSettings
from ..base import LLMDelta, Message, ProviderError, ProviderUnavailable, ToolCall

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

#: Status codes worth a second attempt: rate limits and transient upstreams.
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3


class OpenAICompatLLM:
    name = "openai"

    def __init__(self, settings: LLMSettings, *, api_key: str | None = None) -> None:
        if not api_key:
            raise ProviderUnavailable(
                f"no API key found in ${settings.api_key_env}; "
                "export it or point VA_LLM__BASE_URL at a local server "
                "(Ollama/vLLM/LM Studio accept any placeholder key)"
            )
        self.settings = settings
        self.base_url = (settings.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **settings.extra_headers,
        }
        # Long read timeout: the stream can stay open while tools run.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.timeout_s, connect=15.0)
        )

    # -- provider API ----------------------------------------------------

    def chat_stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMDelta]:
        return self._stream(messages, tools, temperature, max_tokens)

    async def _stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[LLMDelta]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [m.to_openai() for m in messages],
            "stream": True,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
            **self.settings.extra_body,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            emitted = False
            try:
                async for delta in self._attempt(payload):
                    emitted = True
                    yield delta
                return
            except ProviderError as exc:
                last_error = exc
                # Only retry when nothing has been spoken yet — otherwise the
                # caller would hear the first half twice.
                if emitted or not self._retryable(exc):
                    raise
            except httpx.HTTPError as exc:
                last_error = ProviderError(f"LLM request failed: {exc}")
                if emitted:
                    raise last_error from exc
            await asyncio.sleep(0.6 * attempt)
        raise last_error or ProviderError("LLM request failed")

    async def _attempt(self, payload: dict[str, Any]) -> AsyncIterator[LLMDelta]:
        tool_acc: dict[int, dict[str, str]] = {}
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    # The status code stays in the message so the retry loop
                    # can classify it without extra plumbing.
                    raise ProviderError(
                        f"LLM HTTP {response.status_code}: {self._explain(body)}"
                    )
                async for line in response.aiter_lines():
                    delta = self._parse_line(line, tool_acc)
                    if delta is not None:
                        yield delta
        except httpx.HTTPError as exc:
            raise ProviderError(f"LLM stream broke: {exc}") from exc

    def _parse_line(
        self, line: str, tool_acc: dict[int, dict[str, str]]
    ) -> LLMDelta | None:
        line = line.strip()
        if not line or line.startswith(":"):
            return None
        if not line.startswith("data:"):
            return None
        data = line[5:].strip()
        if data == "[DONE]":
            return None
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return None

        choices = payload.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        content = delta.get("content") or ""
        for call in delta.get("tool_calls") or []:
            index = int(call.get("index", 0))
            slot = tool_acc.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if call.get("id"):
                slot["id"] = call["id"]
            function = call.get("function") or {}
            if function.get("name"):
                slot["name"] = function["name"]
            if function.get("arguments"):
                slot["arguments"] += function["arguments"]

        if finish is None and not content:
            return None  # keep-alive / role-only frame

        assembled: list[ToolCall] = []
        if finish is not None:
            assembled = [
                ToolCall(id=slot["id"] or f"call_{i}", name=slot["name"],
                         arguments=slot["arguments"] or "{}")
                for i, slot in sorted(tool_acc.items())
                if slot["name"]
            ]
            tool_acc.clear()

        return LLMDelta(
            content=content,
            tool_calls=assembled,
            finish_reason=finish,
            usage=payload.get("usage"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        text = str(exc)
        return any(f"HTTP {code}" in text for code in RETRY_STATUS)

    @staticmethod
    def _explain(body: str) -> str:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return body[:300]
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message", error))[:300]
        return str(error or payload)[:300]


__all__ = ["OpenAICompatLLM"]
