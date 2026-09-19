"""Audio primitives.

Everything inside the agent speaks one dialect: **mono, 16-bit signed
little-endian PCM**. Conversion to and from container formats, resampling and
level analysis all live here so that providers never have to care.

Deliberately implemented on ``numpy`` + the stdlib ``wave`` module only — no
ffmpeg, no soundfile, no PortAudio — so the core works on a bare machine.
"""

from __future__ import annotations

import base64
import io
import math
import wave
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

#: The rate everything downstream (VAD, ASR, most TTS) expects.
TARGET_SAMPLE_RATE = 16000

PCM_MIME = "audio/pcm"
WAV_MIME = "audio/wav"


@dataclass(slots=True)
class AudioClip:
    """A block of mono 16-bit PCM."""

    pcm: bytes
    sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = 1

    def __post_init__(self) -> None:
        if self.channels != 1:
            raise ValueError("AudioClip only supports mono audio; downmix first")

    # -- constructors ----------------------------------------------------

    @classmethod
    def silence(cls, ms: int, sample_rate: int = TARGET_SAMPLE_RATE) -> AudioClip:
        n = int(sample_rate * ms / 1000)
        return cls(b"\x00\x00" * n, sample_rate)

    @classmethod
    def from_samples(cls, samples: np.ndarray, sample_rate: int) -> AudioClip:
        samples = np.clip(samples, -1.0, 1.0)
        return cls((samples * 32767.0).astype("<i2").tobytes(), sample_rate)

    @classmethod
    def from_base64(cls, payload: str, sample_rate: int = TARGET_SAMPLE_RATE) -> AudioClip:
        return cls(base64.b64decode(payload), sample_rate)

    # -- properties ------------------------------------------------------

    @property
    def samples(self) -> np.ndarray:
        """Float32 samples in ``[-1, 1]``."""
        if not self.pcm:
            return np.zeros(0, dtype=np.float32)
        raw = np.frombuffer(self.pcm, dtype="<i2")
        return raw.astype(np.float32) / 32768.0

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate

    @property
    def n_samples(self) -> int:
        return len(self.pcm) // 2

    def is_silent(self, threshold: float = 1e-4) -> bool:
        return bool(np.max(np.abs(self.samples)) < threshold) if self.n_samples else True

    def rms(self) -> float:
        if not self.n_samples:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples))))

    # -- transforms ------------------------------------------------------

    def resample(self, target_rate: int) -> AudioClip:
        """Linear-interpolation resample.

        Whisper-family models are robust to the mild aliasing this introduces,
        and it keeps the core dependency-free. Cloud providers take the WAV we
        hand them, so no resampling happens on that path at all.
        """
        if target_rate == self.sample_rate or not self.n_samples:
            return self
        src = self.samples
        n_out = int(round(len(src) * target_rate / self.sample_rate))
        if n_out <= 0:
            return AudioClip(b"", target_rate)
        src_idx = np.linspace(0.0, len(src) - 1, num=n_out, dtype=np.float64)
        left = np.floor(src_idx).astype(np.int64)
        right = np.minimum(left + 1, len(src) - 1)
        frac = (src_idx - left).astype(np.float32)
        out = src[left] * (1.0 - frac) + src[right] * frac
        return AudioClip.from_samples(out, target_rate)

    def to_wav_bytes(self, sample_rate: int | None = None) -> bytes:
        """Encode as a RIFF/WAVE container (what every ASR HTTP API wants)."""
        rate = sample_rate or self.sample_rate
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(self.pcm)
        return buf.getvalue()

    def to_base64(self) -> str:
        return base64.b64encode(self.pcm).decode("ascii")


def decode_wav(data: bytes) -> AudioClip:
    """Decode a WAV container into an :class:`AudioClip`, downmixing if needed."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return AudioClip.from_samples(samples, rate)


def pcm_from_float_bytes(data: bytes, sample_rate: int, *, dtype: str = "<f4") -> AudioClip:
    """Convert a raw float PCM stream (some APIs return this) to int16 PCM."""
    samples = np.frombuffer(data, dtype=dtype)
    return AudioClip.from_samples(samples, sample_rate)


def concat(clips: Iterable[AudioClip]) -> AudioClip:
    clips = list(clips)
    if not clips:
        return AudioClip(b"", TARGET_SAMPLE_RATE)
    rate = clips[0].sample_rate
    parts = [c.resample(rate).pcm if c.sample_rate != rate else c.pcm for c in clips]
    return AudioClip(b"".join(parts), rate)


def wav_header_only(n_bytes: int, sample_rate: int) -> bytes:
    """A WAV header for a stream of unknown-but-announced length."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * n_bytes)
    return buf.getvalue()[:44]


def normalize(clip: AudioClip, target_peak: float = 0.9) -> AudioClip:
    """Peak-normalize, which measurably helps ASR on quiet laptop microphones."""
    samples = clip.samples
    if not len(samples):
        return clip
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-5:
        return clip
    return AudioClip.from_samples(samples * (target_peak / peak), clip.sample_rate)


def estimate_pitch_hz(clip: AudioClip) -> float:
    """Cheap autocorrelation pitch estimate — used only for voice activity heuristics."""
    samples = clip.samples
    if len(samples) < 512:
        return 0.0
    frame = samples[:4096] - float(np.mean(samples[:4096]))
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    if corr[0] <= 0:
        return 0.0
    lo = int(clip.sample_rate / 400)
    hi = min(int(clip.sample_rate / 60), len(corr) - 1)
    if hi <= lo:
        return 0.0
    peak = int(np.argmax(corr[lo:hi])) + lo
    return clip.sample_rate / peak if peak else 0.0


def dbfs(clip: AudioClip) -> float:
    rms = clip.rms()
    return 20.0 * math.log10(rms) if rms > 0 else -120.0


@dataclass(slots=True)
class AudioChunk:
    """A piece of synthesized audio on its way to a speaker.

    Providers may emit raw PCM (cheapest, no decoding needed) or an encoded
    container. Browsers decode both natively; the terminal sink decodes PCM
    directly and shells out for anything else.
    """

    data: bytes
    mime: str = PCM_MIME
    sample_rate: int | None = TARGET_SAMPLE_RATE

    @property
    def is_pcm(self) -> bool:
        return self.mime == PCM_MIME

    def as_clip(self) -> AudioClip:
        if not self.is_pcm:
            raise ValueError(f"not raw PCM: {self.mime}")
        return AudioClip(self.data, self.sample_rate or TARGET_SAMPLE_RATE)


@dataclass(slots=True)
class AudioBuffer:
    """Accumulates streaming PCM frames before a transcription call."""

    sample_rate: int = TARGET_SAMPLE_RATE
    _parts: list[bytes] = field(default_factory=list)
    _n: int = 0

    def append(self, pcm: bytes) -> None:
        if pcm:
            self._parts.append(pcm)
            self._n += len(pcm)

    @property
    def n_bytes(self) -> int:
        return self._n

    @property
    def duration_s(self) -> float:
        return self._n / 2 / self.sample_rate

    def clear(self) -> None:
        self._parts.clear()
        self._n = 0

    def clip(self) -> AudioClip:
        return AudioClip(b"".join(self._parts), self.sample_rate)

    def take(self) -> AudioClip:
        out = self.clip()
        self.clear()
        return out
