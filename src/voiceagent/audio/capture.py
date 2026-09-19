"""Microphone capture for the terminal entrypoint (optional ``local-audio`` extra).

Streams 20 ms frames from ``sounddevice`` into an :class:`UtteranceSegmenter`
and yields finished utterances, so ``voiceagent talk`` gets the same
turn-taking behaviour the browser has.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .pcm import TARGET_SAMPLE_RATE, AudioClip
from .vad import UtteranceSegmenter, VADConfig

FRAME_MS = 20


class MicrophoneUnavailable(RuntimeError):
    pass


async def listen(
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    vad: VADConfig | None = None,
    device: int | str | None = None,
) -> AsyncIterator[AudioClip]:
    """Yield one :class:`AudioClip` per detected utterance until cancelled."""
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on host
        raise MicrophoneUnavailable(
            "Microphone capture needs the 'local-audio' extra:\n"
            "    uv pip install -e '.[local-audio]'\n"
            "(on macOS you may also need: brew install portaudio)"
        ) from exc

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[AudioClip | None] = asyncio.Queue()
    segmenter = UtteranceSegmenter(vad or VADConfig(), sample_rate)
    frame_bytes = int(sample_rate * FRAME_MS / 1000) * 2

    def callback(indata, _frames, _time, status) -> None:  # noqa: ANN001
        if status:
            pass  # overflows/underflows are not fatal for turn-taking
        decision = segmenter.push(bytes(indata))
        if decision.finished:
            loop.call_soon_threadsafe(queue.put_nowait, segmenter.take())

    try:
        stream = sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=frame_bytes // 2,
            channels=1,
            dtype="int16",
            device=device,
            callback=callback,
        )
    except Exception as exc:  # pragma: no cover - depends on host
        raise MicrophoneUnavailable(f"could not open input device: {exc}") from exc

    with stream:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            stream.stop()


__all__ = ["listen", "MicrophoneUnavailable"]
