"""macOS ``say`` — the zero-dependency default.

Nothing to install, no key, no network, and it already ships with a decent
Mandarin voice. Latency per sentence is roughly 60-120 ms including process
spawn, which is competitive with cloud TTS for short replies and unbeatable
for a first-run experience.
"""

from __future__ import annotations

import asyncio
import contextlib
import platform
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from ...audio.pcm import WAV_MIME, AudioChunk, decode_wav
from ...config import TTSSettings
from ...text import sentence_spans
from ..base import ProviderError, ProviderUnavailable

#: Voices we reach for when the caller did not name one.
DEFAULT_VOICES: dict[str, str] = {
    "zh": "Tingting",
    "zh-cn": "Tingting",
    "zh-tw": "Meijia",
    "zh-hk": "Sinji",
    "en": "Samantha",
    "en-us": "Samantha",
    "en-gb": "Daniel",
    "ja": "Kyoko",
    "ko": "Yuna",
    "fr": "Thomas",
    "de": "Anna",
    "es": "Monica",
}

#: Sample rate for the PCM we ask ``say`` to produce. 22.05 kHz is what the
#: built-in voices synthesize at natively, so no extra resampling happens.
SAY_RATE = 22050

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class MacOSSayTTS:
    name = "macos_say"

    def __init__(self, settings: TTSSettings | None = None) -> None:
        self.settings = settings or TTSSettings()
        if platform.system() != "Darwin":
            raise ProviderUnavailable("macos_say is only available on macOS")
        binary = shutil.which("say")
        if not binary:
            raise ProviderUnavailable("the `say` binary was not found on PATH")
        self._binary = binary
        self._tmpdir = tempfile.TemporaryDirectory(prefix="voiceagent-say-")
        self._counter = 0
        self._sem = asyncio.Semaphore(4)
        self._voices: list[str] | None = None

    # -- provider API ----------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        return self._stream(text, voice, language)

    async def _stream(
        self,
        text: str,
        voice: str | None,
        language: str | None,
    ) -> AsyncIterator[AudioChunk]:
        chosen = voice or self.pick_voice(language, text)
        for sentence in sentence_spans(text):
            chunk = await self._render(sentence, chosen)
            if chunk is not None:
                yield chunk

    async def voices(self) -> list[str]:
        if self._voices is not None:
            return self._voices
        proc = await asyncio.create_subprocess_exec(
            self._binary, "-v", "?",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        names: list[str] = []
        for line in stdout.decode("utf-8", "replace").splitlines():
            parts = line.split()
            if parts:
                names.append(parts[0])
        self._voices = sorted(set(names))
        return self._voices

    async def aclose(self) -> None:
        self._tmpdir.cleanup()

    # -- internals -------------------------------------------------------

    def pick_voice(self, language: str | None, text: str = "") -> str | None:
        """Choose a voice.

        The system default voice is *not* safe to fall back on: on a machine
        whose default has no Chinese data, ``say`` silently writes a zero-frame
        WAV for Chinese text. So when the text itself is Chinese we name a
        Chinese voice outright rather than trusting the system default.
        """
        if self.settings.voice:
            return self.settings.voice
        if text and _CJK.search(text):
            return DEFAULT_VOICES["zh"]
        if not language:
            return None  # let the system default decide
        key = language.lower().replace("_", "-")
        if key in DEFAULT_VOICES:
            return DEFAULT_VOICES[key]
        return DEFAULT_VOICES.get(key.split("-")[0])

    async def _render(self, sentence: str, voice: str | None) -> AudioChunk | None:
        if not sentence.strip():
            return None
        self._counter += 1
        path = Path(self._tmpdir.name) / f"say-{self._counter}.wav"
        cmd = [
            self._binary,
            "-o", str(path),
            "--file-format=WAVE",
            f"--data-format=LEI16@{SAY_RATE}",
            "-r", str(self.settings.rate),
        ]
        if voice:
            cmd += ["-v", voice]
        cmd.append(sentence)

        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ProviderError(
                f"say failed ({proc.returncode}): {stderr.decode('utf-8', 'replace')[:200]}"
            )
        data = await asyncio.to_thread(path.read_bytes)
        with contextlib.suppress(OSError):
            path.unlink()

        # An empty WAV is the worst possible failure for a voice assistant:
        # everything looks fine and nothing is heard. Catch it here.
        clip = decode_wav(data)
        if clip.n_samples == 0:
            raise ProviderError(
                f"macOS `say` produced no audio for {sentence[:40]!r} "
                f"(voice={voice or 'system default'}). The default voice often "
                "cannot speak Chinese — set `tts.voice` explicitly, e.g. "
                "\"Tingting\" for zh-CN or \"Meijia\" for zh-TW."
            )
        return AudioChunk(data=data, mime=WAV_MIME, sample_rate=SAY_RATE)


__all__ = ["DEFAULT_VOICES", "MacOSSayTTS", "SAY_RATE"]
