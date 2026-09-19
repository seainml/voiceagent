"""Playback sinks for the terminal entrypoint.

The browser has its own sink (the Web Audio API), so this module only needs to
serve ``voiceagent talk``. Two backends:

* ``sounddevice`` (optional extra ``local-audio``) — true streaming, one
  long-lived output stream, no gaps between TTS chunks.
* ``afplay``/``ffplay``/``aplay`` — fall back to shelling out per chunk. Works
  everywhere but inserts a small gap between sentences.

Both are cancellable, which is what makes barge-in work from the terminal.
"""

from __future__ import annotations

import asyncio
import contextlib
import platform
import shutil
import tempfile
from pathlib import Path

from .pcm import PCM_MIME, AudioChunk

_EXT = {"audio/wav": ".wav", "audio/mpeg": ".mp3", "audio/ogg": ".ogg", "audio/opus": ".opus"}


def _shell_player() -> list[str] | None:
    system = platform.system()
    if system == "Darwin" and shutil.which("afplay"):
        return ["afplay"]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    if shutil.which("paplay"):
        return ["paplay"]
    return None


class Speaker:
    """Async audio sink with cancellation."""

    def __init__(self, *, sample_rate: int = 24000, prefer_streaming: bool = True) -> None:
        self.sample_rate = sample_rate
        self._sd = None
        self._stream = None
        self._shell = _shell_player()
        self._proc: asyncio.subprocess.Process | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._counter = 0
        self._lock = asyncio.Lock()
        self._closed = False
        if prefer_streaming:
            self._try_open_stream()

    # -- backend setup ---------------------------------------------------

    def _try_open_stream(self) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except Exception:
            return
        self._sd = sd
        try:
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate, channels=1, dtype="int16", latency="low"
            )
            self._stream.start()
        except Exception:
            self._stream = None
            self._sd = None

    @property
    def backend(self) -> str:
        if self._stream is not None:
            return f"sounddevice@{self.sample_rate}Hz"
        if self._shell:
            return Path(self._shell[0]).name
        return "none"

    def _ensure_samplerate(self, rate: int | None) -> None:
        """Reopen the stream if the provider switched sample rates on us."""
        if rate is None or rate == self.sample_rate:
            return
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None
            self.sample_rate = rate
            self._try_open_stream()

    # -- playback --------------------------------------------------------

    async def play(self, chunk: AudioChunk) -> None:
        if self._closed or not chunk.data:
            return
        async with self._lock:
            if chunk.is_pcm and self._stream is not None:
                self._ensure_samplerate(chunk.sample_rate)
                if self._stream is not None:
                    await asyncio.to_thread(self._write_pcm, chunk.data)
                    return
            await self._play_via_shell(chunk)

    def _write_pcm(self, pcm: bytes) -> None:
        try:
            self._stream.write(pcm)  # type: ignore[union-attr]
        except Exception:
            # Device vanished (headphones unplugged, etc.) — degrade silently.
            self._stream = None

    async def _play_via_shell(self, chunk: AudioChunk) -> None:
        if not self._shell:
            return
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="voiceagent-")
        self._counter += 1
        if chunk.is_pcm:
            from .pcm import AudioClip

            clip = AudioClip(chunk.data, chunk.sample_rate or 24000)
            path = Path(self._tmpdir.name) / f"chunk-{self._counter}.wav"
            path.write_bytes(clip.to_wav_bytes())
        else:
            ext = _EXT.get(chunk.mime, ".bin")
            path = Path(self._tmpdir.name) / f"chunk-{self._counter}{ext}"
            path.write_bytes(chunk.data)

        self._proc = await asyncio.create_subprocess_exec(
            *self._shell, str(path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        proc = self._proc
        try:
            await proc.wait()
        except asyncio.CancelledError:
            await self.stop()
            raise
        finally:
            if self._proc is proc:
                self._proc = None

    async def stop(self) -> None:
        """Cancel whatever is playing — the audio half of barge-in."""
        proc = self._proc
        self._proc = None
        if proc and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

    async def aclose(self) -> None:
        self._closed = True
        await self.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


__all__ = ["Speaker", "PCM_MIME"]
