"""OpenAI-compatible ``/audio/transcriptions`` (OpenAI, Groq, DashScope
compatible-mode, SiliconFlow, LocalAI, whisper.cpp server, ...).

Sends a WAV rather than a compressed container: it is lossless, needs no
ffmpeg, and is small enough at 16 kHz mono that the extra bytes do not matter
on a LAN or a modern uplink.
"""

from __future__ import annotations

import json

import httpx

from ...audio.pcm import TARGET_SAMPLE_RATE, AudioClip
from ...audio.vad import VADConfig, trim_silence
from ...config import ASRSettings
from ..base import ProviderError, ProviderUnavailable, Transcript

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAITranscriptionASR:
    name = "openai"

    def __init__(self, settings: ASRSettings, *, api_key: str | None = None) -> None:
        if not api_key:
            raise ProviderUnavailable(
                f"no API key found in ${settings.api_key_env}; "
                "export it or switch providers"
            )
        self.settings = settings
        self.base_url = (settings.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(timeout=settings.timeout_s)

    async def transcribe(
        self,
        clip: AudioClip,
        *,
        language: str | None = None,
        hotwords: list[str] | None = None,
    ) -> Transcript:
        if clip.n_samples == 0:
            return Transcript(text="", provider=self.name)
        audio = clip.resample(TARGET_SAMPLE_RATE)
        audio = trim_silence(audio, VADConfig(), pad_ms=120)
        if audio.n_samples == 0:
            return Transcript(text="", duration_s=clip.duration_s, provider=self.name)

        # Whisper's `prompt` is the de-facto contextual-biasing knob: a comma
        # separated vocabulary list measurably improves proper-noun recall.
        prompt = ", ".join(hotwords) if hotwords else None
        data: dict[str, str] = {"model": self.settings.model, "response_format": "json"}
        lang = language or self.settings.language
        if lang:
            data["language"] = lang.split("-")[0].lower()
        if prompt:
            data["prompt"] = prompt[:900]

        files = {"file": ("utterance.wav", audio.to_wav_bytes(), "audio/wav")}
        try:
            response = await self._client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=self._headers,
                data=data,
                files=files,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"ASR request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"ASR HTTP {response.status_code}: {self._explain(response.text)}"
            )
        payload = response.json()
        return Transcript(
            text=str(payload.get("text", "")).strip(),
            language=payload.get("language") or lang,
            duration_s=audio.duration_s,
            segments=payload.get("segments") or [],
            provider=self.name,
        )

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


__all__ = ["OpenAITranscriptionASR"]
