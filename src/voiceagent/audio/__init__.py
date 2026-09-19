"""Audio plumbing: one PCM dialect, resampling, VAD and playback sinks."""

from __future__ import annotations

from .pcm import (
    PCM_MIME,
    TARGET_SAMPLE_RATE,
    WAV_MIME,
    AudioBuffer,
    AudioChunk,
    AudioClip,
    concat,
    dbfs,
    decode_wav,
    normalize,
)
from .player import Speaker
from .vad import (
    EndpointDecision,
    FrameVAD,
    UtteranceSegmenter,
    VADConfig,
    segment_clip,
    trim_silence,
)

__all__ = [
    "PCM_MIME",
    "WAV_MIME",
    "TARGET_SAMPLE_RATE",
    "AudioBuffer",
    "AudioChunk",
    "AudioClip",
    "EndpointDecision",
    "FrameVAD",
    "Speaker",
    "UtteranceSegmenter",
    "VADConfig",
    "concat",
    "dbfs",
    "decode_wav",
    "normalize",
    "segment_clip",
    "trim_silence",
]
