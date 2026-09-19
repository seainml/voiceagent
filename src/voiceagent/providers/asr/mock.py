"""Test doubles and the browser-side ASR sentinel."""

from __future__ import annotations

from ...audio.pcm import AudioClip
from ..base import ProviderError, Transcript


class MockASR:
    """Returns a queued script of transcripts — deterministic end-to-end tests."""

    name = "mock"

    def __init__(self, script: list[str] | None = None, default: str = "你好") -> None:
        self.script = list(script or [])
        self.default = default
        self.calls: list[float] = []

    async def transcribe(
        self,
        clip: AudioClip,
        *,
        language: str | None = None,
        hotwords: list[str] | None = None,
    ) -> Transcript:
        self.calls.append(clip.duration_s)
        text = self.script.pop(0) if self.script else self.default
        return Transcript(text=text, language=language or "zh", duration_s=clip.duration_s,
                          provider=self.name)

    async def aclose(self) -> None:
        return None


class BrowserASR:
    """Placeholder for recognition performed in the browser.

    The transport layer routes raw text straight into the pipeline when this
    provider is selected, so any call here is a wiring bug worth surfacing.
    """

    name = "browser"

    async def transcribe(
        self,
        clip: AudioClip,
        *,
        language: str | None = None,
        hotwords: list[str] | None = None,
    ) -> Transcript:
        raise ProviderError(
            "provider 'browser' transcribes on the client; the server must not "
            "receive audio for it"
        )

    async def aclose(self) -> None:
        return None


__all__ = ["BrowserASR", "MockASR"]
