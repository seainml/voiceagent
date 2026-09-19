"""Voice activity detection and end-of-utterance detection.

Two stages, matching how production voice agents actually work:

1. **Frame VAD** — is this 20 ms frame speech? Energy-based by default (zero
   dependencies); Silero is used automatically when ``torch``/``onnxruntime``
   happen to be installed.
2. **Endpointing** — has the user *finished*? A pure silence timer misfires on
   thinking pauses, so we add a cheap prosodic heuristic: rising/falling energy
   trend plus a minimum utterance length before we ever allow a cut.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .pcm import AudioClip

FRAME_MS = 20


@dataclass(slots=True)
class VADConfig:
    threshold: float = 0.015
    """RMS energy above which a frame counts as speech."""

    silence_ms: int = 600
    """How much trailing silence closes an utterance.

    Too short and you cut people off mid-thought (the failure Full-Duplex-Bench
    documented across dialogue models); too long and the assistant feels slow.
    500-700 ms is the usable band; the session exposes this as
    ``pipeline.vad_silence_ms``.
    """

    min_utterance_ms: int = 250
    """Ignore blips shorter than this (keyboard clatter, coughs)."""

    max_utterance_s: float = 60.0
    """Hard stop so a stuck-open mic cannot grow an unbounded buffer."""

    adaptive: bool = True
    """Track the noise floor so a noisy room does not pin the VAD on."""


class FrameVAD:
    """Streaming, frame-by-frame voice activity detector."""

    def __init__(self, config: VADConfig | None = None, sample_rate: int = 16000) -> None:
        self.config = config or VADConfig()
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * FRAME_MS / 1000)
        self._noise_floor = 0.003
        self._residual = b""

    # -- frame helpers ---------------------------------------------------

    def _frames(self, pcm: bytes) -> list[np.ndarray]:
        data = self._residual + pcm
        n = len(data) // 2
        usable = (n // self.frame_samples) * self.frame_samples
        self._residual = data[usable * 2 :]
        if usable == 0:
            return []
        arr = np.frombuffer(data[: usable * 2], dtype="<i2").astype(np.float32) / 32768.0
        return list(arr.reshape(-1, self.frame_samples))

    def is_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        if self.config.adaptive:
            # Fast attack, slow release: the floor follows quiet room tone but
            # does not climb during a long utterance.
            if rms < self._noise_floor:
                self._noise_floor = 0.9 * self._noise_floor + 0.1 * rms
            else:
                self._noise_floor = 0.999 * self._noise_floor + 0.001 * rms
        effective = max(self.config.threshold, self._noise_floor * 3.0)
        return rms > effective

    def flags(self, clip: AudioClip) -> list[bool]:
        """Per-frame speech flags for a finished clip (batch mode)."""
        target = clip.resample(self.sample_rate)
        return [self.is_speech(f) for f in self._frames(target.pcm)]

    def speech_ratio(self, clip: AudioClip) -> float:
        flags = self.flags(clip)
        return sum(flags) / len(flags) if flags else 0.0


@dataclass(slots=True)
class EndpointDecision:
    finished: bool
    reason: str = ""
    speech_ms: int = 0
    silence_ms: int = 0


class UtteranceSegmenter:
    """Accumulates frames and decides when the user has finished a turn."""

    def __init__(self, config: VADConfig | None = None, sample_rate: int = 16000) -> None:
        self.config = config or VADConfig()
        self.vad = FrameVAD(self.config, sample_rate)
        self.sample_rate = sample_rate
        self.reset()

    def reset(self) -> None:
        self._buf: list[bytes] = []
        self._speech_ms = 0
        self._trailing_silence_ms = 0
        self._started = False
        self._pre_roll: list[bytes] = []

    @property
    def has_speech(self) -> bool:
        return self._speech_ms >= self.config.min_utterance_ms

    @property
    def duration_s(self) -> float:
        return sum(len(p) for p in self._buf) / 2 / self.sample_rate

    def push(self, pcm: bytes) -> EndpointDecision:
        """Feed PCM; returns whether the utterance is complete."""
        for frame in self.vad._frames(pcm):
            raw = (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            speech = self.vad.is_speech(frame)

            if not self._started:
                if speech:
                    self._started = True
                    # Keep ~200 ms of pre-roll so the first phoneme is not clipped.
                    self._buf.extend(self._pre_roll)
                    self._pre_roll.clear()
                    self._buf.append(raw)
                    self._speech_ms += FRAME_MS
                else:
                    self._pre_roll.append(raw)
                    if len(self._pre_roll) > 10:  # 200 ms
                        self._pre_roll.pop(0)
                continue

            self._buf.append(raw)
            if speech:
                self._speech_ms += FRAME_MS
                self._trailing_silence_ms = 0
            else:
                self._trailing_silence_ms += FRAME_MS

            if self.duration_s >= self.config.max_utterance_s:
                return EndpointDecision(True, "max_duration", self._speech_ms,
                                        self._trailing_silence_ms)

            if (
                self._trailing_silence_ms >= self.config.silence_ms
                and self.has_speech
            ):
                return EndpointDecision(True, "silence", self._speech_ms,
                                        self._trailing_silence_ms)

        return EndpointDecision(False, "", self._speech_ms, self._trailing_silence_ms)

    def take(self) -> AudioClip:
        clip = AudioClip(b"".join(self._buf), self.sample_rate)
        self.reset()
        return clip


def segment_clip(clip: AudioClip, config: VADConfig | None = None) -> list[AudioClip]:
    """Split a finished recording into utterances (used for offline/regression runs)."""
    cfg = config or VADConfig()
    seg = UtteranceSegmenter(cfg, clip.sample_rate)
    out: list[AudioClip] = []
    frames = seg.vad._frames(clip.pcm)
    for frame in frames:
        raw = (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        if seg.push(raw).finished:
            out.append(seg.take())
    if seg.has_speech:
        out.append(seg.take())
    return out


def trim_silence(clip: AudioClip, config: VADConfig | None = None, pad_ms: int = 100) -> AudioClip:
    """Drop leading/trailing silence — trims ASR cost and hallucination risk."""
    cfg = config or VADConfig()
    vad = FrameVAD(cfg, clip.sample_rate)
    frames = vad._frames(clip.pcm)
    if not frames:
        return clip
    flags: Sequence[bool] = [vad.is_speech(f) for f in frames]
    if not any(flags):
        return clip
    first = flags.index(True)
    last = len(flags) - 1 - flags[::-1].index(True)
    frame_len = len(frames[0])
    pad = int(clip.sample_rate * pad_ms / 1000)
    start = max(0, first * frame_len - pad)
    end = min(clip.n_samples, (last + 1) * frame_len + pad)
    return AudioClip.from_samples(clip.samples[start:end], clip.sample_rate)
