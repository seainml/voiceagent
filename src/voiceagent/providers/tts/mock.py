"""Deterministic providers for tests and for `voiceagent doctor` dry runs."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator

import numpy as np

from ...audio.pcm import PCM_MIME, AudioChunk, AudioClip
from ...text import sentence_spans

#: 340 Hz is an A4-ish tone: audible, short, and obviously synthetic.
TONE_HZ = 440.0


class MockTTS:
    """Emits one short sine blip per sentence instead of speech."""

    name = "mock"

    def __init__(self, sample_rate: int = 16000, ms_per_sentence: int = 120) -> None:
        self.sample_rate = sample_rate
        self.ms_per_sentence = ms_per_sentence
        self.spoken: list[str] = []

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        return self._stream(text)

    async def _stream(self, text: str) -> AsyncIterator[AudioChunk]:
        for sentence in sentence_spans(text):
            self.spoken.append(sentence)
            n = int(self.sample_rate * self.ms_per_sentence / 1000)
            t = np.arange(n, dtype=np.float32) / self.sample_rate
            wave = 0.2 * np.sin(2 * math.pi * TONE_HZ * t)
            fade = min(n // 4, 200)
            if fade:
                wave[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                wave[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            yield AudioChunk(
                data=AudioClip.from_samples(wave, self.sample_rate).pcm,
                mime=PCM_MIME,
                sample_rate=self.sample_rate,
            )
            await asyncio.sleep(0)

    async def voices(self) -> list[str]:
        return ["mock-tone"]

    async def aclose(self) -> None:
        return None


__all__ = ["MockTTS"]
