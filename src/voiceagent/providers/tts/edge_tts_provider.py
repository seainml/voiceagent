"""edge-tts — free Microsoft neural voices, streamed as MP3.

The best quality-per-effort option for Chinese when you have no budget and no
key: ``uv pip install -e '.[edge]'`` and you get 晓晓/云希-class voices.
Note it is an unofficial client for a public endpoint, so treat it as a
convenience provider rather than something to build a product on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ...audio.pcm import AudioChunk
from ...config import TTSSettings
from ...text import sentence_spans
from ..base import ProviderError, ProviderUnavailable

MP3_MIME = "audio/mpeg"

DEFAULT_VOICES = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "zh-cn": "zh-CN-XiaoxiaoNeural",
    "zh-tw": "zh-TW-HsiaoChenNeural",
    "zh-hk": "zh-HK-HiuMaanNeural",
    "en": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
}


class EdgeTTS:
    name = "edge"

    def __init__(self, settings: TTSSettings) -> None:
        try:
            import edge_tts  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ProviderUnavailable(
                "edge-tts is not installed; run: uv pip install -e '.[edge]'"
            ) from exc
        self._edge = edge_tts
        self.settings = settings

    def pick_voice(self, language: str | None) -> str:
        if self.settings.voice:
            return self.settings.voice
        if language:
            key = language.lower().replace("_", "-")
            if key in DEFAULT_VOICES:
                return DEFAULT_VOICES[key]
            base = DEFAULT_VOICES.get(key.split("-")[0])
            if base:
                return base
        return DEFAULT_VOICES["zh"]

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        return self._stream(text, voice or self.pick_voice(language))

    async def _stream(self, text: str, voice: str) -> AsyncIterator[AudioChunk]:
        rate = f"{int((self.settings.speed - 1.0) * 100):+d}%"
        pitch = f"{self.settings.pitch:+d}Hz"
        volume = f"{self.settings.volume:+d}%"
        for sentence in sentence_spans(text):
            communicate = self._edge.Communicate(
                sentence, voice, rate=rate, pitch=pitch, volume=volume
            )
            try:
                async for event in communicate.stream():
                    if event.get("type") == "audio" and event.get("data"):
                        yield AudioChunk(data=event["data"], mime=MP3_MIME, sample_rate=None)
            except Exception as exc:  # pragma: no cover - network dependent
                raise ProviderError(f"edge-tts failed: {exc}") from exc

    async def voices(self) -> list[str]:
        try:
            listed = await self._edge.list_voices()
        except Exception:  # pragma: no cover - network dependent
            return sorted(set(DEFAULT_VOICES.values()))
        return sorted({v["ShortName"] for v in listed})

    async def aclose(self) -> None:
        return None


__all__ = ["EdgeTTS"]
