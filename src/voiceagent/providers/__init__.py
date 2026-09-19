"""Provider layer: three Protocols, N implementations, one registry."""

from __future__ import annotations

from .base import (
    ASRProvider,
    AudioChunk,
    AudioClip,
    LLMDelta,
    LLMProvider,
    Message,
    ProviderError,
    ProviderUnavailable,
    ToolCall,
    Transcript,
    TTSProvider,
)
from .registry import (
    ProviderBundle,
    ProviderStatus,
    build_all,
    build_asr,
    build_bundle,
    build_llm,
    build_tts,
    capability_report,
    resolve_name,
)

__all__ = [
    "ASRProvider",
    "AudioChunk",
    "AudioClip",
    "LLMDelta",
    "LLMProvider",
    "Message",
    "ProviderBundle",
    "ProviderError",
    "ProviderStatus",
    "ProviderUnavailable",
    "TTSProvider",
    "ToolCall",
    "Transcript",
    "build_all",
    "build_asr",
    "build_bundle",
    "build_llm",
    "build_tts",
    "capability_report",
    "resolve_name",
]
