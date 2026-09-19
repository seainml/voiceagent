"""Provider contracts.

Three tiny Protocols. Anything that satisfies them can be plugged in — the
rest of the system never imports a concrete provider class directly, it goes
through :mod:`voiceagent.providers.registry`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..audio.pcm import AudioChunk, AudioClip


class ProviderError(RuntimeError):
    """Raised when a provider is misconfigured or unreachable."""


class ProviderUnavailable(ProviderError):
    """The provider cannot run here (missing package, missing key, wrong OS)."""


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Transcript:
    text: str
    language: str | None = None
    duration_s: float = 0.0
    confidence: float | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    provider: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@runtime_checkable
class ASRProvider(Protocol):
    name: str

    async def transcribe(
        self,
        clip: AudioClip,
        *,
        language: str | None = None,
        hotwords: list[str] | None = None,
    ) -> Transcript: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------


@runtime_checkable
class TTSProvider(Protocol):
    name: str

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Yield audio for ``text``.

        Implementations should stream at sentence granularity so the caller can
        start playing before the whole reply has been generated.
        """
        ...

    async def voices(self) -> list[str]: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, as the API returns it

    def parsed_args(self) -> dict[str, Any]:
        import json

        if not self.arguments:
            return {}
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


@dataclass(slots=True)
class LLMDelta:
    """One streaming increment from a chat completion."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass(slots=True)
class Message:
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.content:
            out["content"] = self.content
        elif self.role != "assistant":
            out["content"] = ""
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments or "{}"},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def chat_stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMDelta]: ...

    async def aclose(self) -> None: ...


__all__ = [
    "ASRProvider",
    "AudioChunk",
    "AudioClip",
    "LLMDelta",
    "LLMProvider",
    "Message",
    "ProviderError",
    "ProviderUnavailable",
    "TTSProvider",
    "ToolCall",
    "Transcript",
]
