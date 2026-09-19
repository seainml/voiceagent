"""OpenAI-compatible speech synthesis (OpenAI, Azure-compatible gateways,
DashScope compatible-mode, LocalAI, Kokoro-FastAPI, ...).

Prefers raw PCM output so chunks can be forwarded to the browser without any
container parsing; falls back to whatever container the endpoint supports.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from ...audio.pcm import PCM_MIME, AudioChunk
from ...config import TTSSettings
from ...text import sentence_spans
from ..base import ProviderError, ProviderUnavailable

DEFAULT_BASE_URL = "https://api.openai.com/v1"

#: Endpoints that emit raw PCM define their own rate; OpenAI uses 24 kHz.
PCM_RATES = {"pcm": 24000}
MIME_BY_FORMAT = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": PCM_MIME,
}


class OpenAITTS:
    name = "openai"

    def __init__(self, settings: TTSSettings, *, api_key: str | None = None) -> None:
        if not api_key:
            raise ProviderUnavailable(
                f"no API key found in ${settings.api_key_env}; "
                "export it or switch providers"
            )
        self.settings = settings
        self.base_url = (settings.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._fmt = (settings.response_format or "mp3").lower()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=settings.timeout_s)

    # -- provider API ----------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        return self._stream(text, voice)

    async def _stream(self, text: str, voice: str | None) -> AsyncIterator[AudioChunk]:
        # Sentences are synthesized concurrently, then yielded in order: this
        # hides per-request latency without reordering the audio.
        pending: list[tuple[int, str]] = []
        for i, sentence in enumerate(sentence_spans(text)):
            pending.append((i, sentence))
        for _i, sentence in pending:
            async for chunk in self._one(sentence, voice):
                yield chunk

    async def _one(self, sentence: str, voice: str | None) -> AsyncIterator[AudioChunk]:
        payload = {
            "model": self.settings.model,
            "input": sentence,
            "voice": voice or self.settings.voice or "alloy",
            "response_format": self._fmt,
        }
        if self.settings.speed and self.settings.speed != 1.0:
            payload["speed"] = max(0.25, min(4.0, self.settings.speed))
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/audio/speech", json=payload, headers=self._headers
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise ProviderError(
                        f"TTS HTTP {response.status_code}: {self._explain(body)}"
                    )
                mime = MIME_BY_FORMAT.get(self._fmt, "application/octet-stream")
                rate = PCM_RATES.get(self._fmt)
                async for block in response.aiter_bytes(4096):
                    if block:
                        yield AudioChunk(data=block, mime=mime, sample_rate=rate)
        except httpx.HTTPError as exc:
            raise ProviderError(f"TTS request failed: {exc}") from exc

    async def voices(self) -> list[str]:
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "sage"]

    async def aclose(self) -> None:
        await self._client.aclose()

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


__all__ = ["OpenAITTS"]
