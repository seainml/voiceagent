"""Adding a custom provider — the "swap any layer" promise, made concrete.

Any object satisfying one of the three Protocols in
:mod:`voiceagent.providers.base` can be plugged in. Nothing in the agent, the
session or the UI needs to change.

This example implements a deliberately crude TTS provider that speaks by
playing one short tone per syllable, so you can see the whole shape without
any dependency.

    .venv/bin/python examples/custom_provider.py
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.audio.pcm import PCM_MIME, AudioChunk, AudioClip  # noqa: E402
from voiceagent.providers.base import TTSProvider  # noqa: E402
from voiceagent.text import sentence_spans  # noqa: E402

SYLLABLE_MS = 90
BASE_HZ = 220.0


class BeepTTS:
    """Contract in three methods: synthesize(), voices(), aclose()."""

    name = "beep"

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    # NOTE: this is an *async generator* method, not `async def ... -> list`.
    # Streaming is the point: the caller can start playing before the rest is
    # generated. Yield at sentence granularity so latency stays low.
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
            syllables = max(1, sum(ch.isalnum() for ch in sentence))
            n = int(self.sample_rate * SYLLABLE_MS * syllables / 1000)
            t = np.arange(n, dtype=np.float32) / self.sample_rate
            # Rising pitch per sentence so it is audibly "speech-like".
            wave = 0.2 * np.sin(2 * math.pi * BASE_HZ * t)
            envelope = np.minimum(1.0, np.minimum(t * 40, (n / self.sample_rate - t) * 40))
            wave = (wave * envelope).astype(np.float32)
            yield AudioChunk(
                data=AudioClip.from_samples(wave, self.sample_rate).pcm,
                mime=PCM_MIME,
                sample_rate=self.sample_rate,
            )
            await asyncio.sleep(0)  # keep the event loop responsive

    async def voices(self) -> list[str]:
        return ["beep-low", "beep-high"]

    async def aclose(self) -> None:
        return None


async def main() -> None:
    tts = BeepTTS()

    # 1. It satisfies the Protocol — this is the actual contract check.
    print("isinstance(tts, TTSProvider):", isinstance(tts, TTSProvider))

    # 2. It drops straight into a session. Nothing else changes.
    from voiceagent.config import load_settings
    from voiceagent.providers.registry import ProviderBundle
    from voiceagent.session import VoiceSession
    from voiceagent.tools.builtin import build_registry

    settings = load_settings()
    registry = await build_registry(settings, include_mcp=False)
    bundle = ProviderBundle(
        asr=None,  # type: ignore[arg-type]  # not used with text input
        tts=tts,
        llm=None,  # type: ignore[arg-type]
        names={"asr": "n/a", "tts": "beep", "llm": "n/a"},
    )

    events = []

    async def on_event(event) -> None:
        events.append(event)
        if event.type.value == "audio":
            print(f"  audio chunk: {len(event.data['data'])} base64 chars")

    session = VoiceSession(
        settings, bundle, registry,
        on_event=on_event, emit_audio_payloads=True,
    )
    await session.speak("这是一段用自定义 provider 合成的话。")
    await session.aclose()

    print(f"\n{len([e for e in events if e.type.value == 'audio'])} chunks emitted")
    print("The same object would work unchanged for ASR or LLM — "
          "see the Protocols in providers/base.py.")


if __name__ == "__main__":
    asyncio.run(main())
