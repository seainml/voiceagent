"""Provider registry.

Two responsibilities:

1. **Build** a concrete provider from settings, resolving ``provider = "auto"``
   into something that actually works on this machine.
2. **Probe** what is available, so ``voiceagent doctor`` and the browser
   settings panel can show the user real options instead of a stack trace.

``auto`` resolution is deliberately local-first:

* TTS  → macOS ``say`` (always present here) → edge-tts → OpenAI → mock
* ASR  → faster-whisper (if installed) → OpenAI-compatible → browser
* LLM  → any key in the environment → Ollama/vLLM on localhost → echo
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from .base import (
    ASRProvider,
    LLMProvider,
    ProviderError,
    ProviderUnavailable,
    TTSProvider,
)

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

#: Hosts that are obviously a local server: they accept any bearer token.
LOCAL_HOSTS = {"", "127.0.0.1", "localhost", "0.0.0.0", "::1", "[::1]"}


def _is_local_endpoint(base_url: str | None) -> bool:
    from urllib.parse import urlparse

    if not base_url:
        return False
    try:
        return (urlparse(base_url).hostname or "") in LOCAL_HOSTS
    except ValueError:
        return False


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass(slots=True)
class ProviderStatus:
    kind: str
    name: str
    available: bool
    reason: str = ""
    is_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "available": self.available,
            "reason": self.reason,
            "default": self.is_default,
        }


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------


def _has_key(settings: Settings, env_name: str | None) -> bool:
    return bool(settings.secret(env_name))


def asr_status(settings: Settings) -> list[ProviderStatus]:
    local_ok = _module_available("faster_whisper")
    cloud_ok = _has_key(settings, settings.asr.api_key_env)
    return [
        ProviderStatus(
            "asr", "faster_whisper", local_ok,
            "" if local_ok else "install with: uv pip install -e '.[local-asr]'",
        ),
        ProviderStatus(
            "asr", "openai", cloud_ok,
            "" if cloud_ok else f"set ${settings.asr.api_key_env}",
        ),
        ProviderStatus("asr", "browser", True, "recognised in the browser; no server ASR"),
        ProviderStatus("asr", "mock", True, "test double"),
    ]


def tts_status(settings: Settings) -> list[ProviderStatus]:
    say_ok = platform.system() == "Darwin" and shutil.which("say") is not None
    edge_ok = _module_available("edge_tts")
    cloud_ok = _has_key(settings, settings.tts.api_key_env)
    return [
        ProviderStatus(
            "tts", "macos_say", say_ok,
            "" if say_ok else "macOS only",
        ),
        ProviderStatus(
            "tts", "edge", edge_ok,
            "" if edge_ok else "install with: uv pip install -e '.[edge]'",
        ),
        ProviderStatus(
            "tts", "openai", cloud_ok,
            "" if cloud_ok else f"set ${settings.tts.api_key_env}",
        ),
        ProviderStatus("tts", "mock", True, "test double"),
    ]


def llm_status(settings: Settings) -> list[ProviderStatus]:
    cloud_ok = _has_key(settings, settings.llm.api_key_env)
    explicit_base = bool(os.environ.get("VA_LLM__BASE_URL"))
    return [
        ProviderStatus(
            "llm", "openai", cloud_ok or explicit_base,
            "" if (cloud_ok or explicit_base)
            else f"set ${settings.llm.api_key_env} or VA_LLM__BASE_URL",
        ),
        ProviderStatus(
            "llm", "ollama", False,
            f"start a local server at {OLLAMA_BASE_URL} and set VA_LLM__BASE_URL",
        ),
        ProviderStatus("llm", "echo", True, "no model; repeats what it heard"),
        ProviderStatus("llm", "mock", True, "test double"),
    ]


def capability_report(settings: Settings) -> dict[str, Any]:
    """Everything `doctor` and the settings panel need, in one call."""
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "tools": {
            "say": shutil.which("say") or "",
            "afplay": shutil.which("afplay") or "",
            "ffplay": shutil.which("ffplay") or "",
            "claude": shutil.which("claude") or "",
            "codex": shutil.which("codex") or "",
            "dsh": shutil.which("dsh") or "",
        },
        "providers": {
            "asr": [s.to_dict() for s in asr_status(settings)],
            "tts": [s.to_dict() for s in tts_status(settings)],
            "llm": [s.to_dict() for s in llm_status(settings)],
        },
        "resolved": {
            "asr": resolve_name("asr", settings),
            "tts": resolve_name("tts", settings),
            "llm": resolve_name("llm", settings),
        },
        "config": settings.describe(),
    }


# --------------------------------------------------------------------------
# auto resolution
# --------------------------------------------------------------------------


def resolve_name(kind: str, settings: Settings) -> str:
    """Turn ``auto`` into a concrete provider name."""
    if kind == "asr":
        chosen = settings.asr.provider
        if chosen != "auto":
            return chosen
        if _module_available("faster_whisper"):
            return "faster_whisper"
        if _has_key(settings, settings.asr.api_key_env):
            return "openai"
        return "browser"

    if kind == "tts":
        chosen = settings.tts.provider
        if chosen != "auto":
            return chosen
        if platform.system() == "Darwin" and shutil.which("say"):
            return "macos_say"
        if _module_available("edge_tts"):
            return "edge"
        if _has_key(settings, settings.tts.api_key_env):
            return "openai"
        return "mock"

    if kind == "llm":
        chosen = settings.llm.provider
        if chosen != "auto":
            return chosen
        if _has_key(settings, settings.llm.api_key_env):
            return "openai"
        if os.environ.get("VA_LLM__BASE_URL"):
            return "openai"
        return "echo"

    raise ProviderError(f"unknown provider kind: {kind}")


# --------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------


def build_asr(settings: Settings, name: str | None = None) -> ASRProvider:
    name = name or resolve_name("asr", settings)
    if name == "faster_whisper":
        from .asr.faster_whisper import FasterWhisperASR

        return FasterWhisperASR(settings.asr)
    if name == "openai":
        from .asr.openai_asr import OpenAITranscriptionASR

        return OpenAITranscriptionASR(
            settings.asr, api_key=settings.secret(settings.asr.api_key_env)
        )
    if name == "browser":
        from .asr.mock import BrowserASR

        return BrowserASR()
    if name == "mock":
        from .asr.mock import MockASR

        return MockASR()
    raise ProviderError(f"unknown ASR provider: {name!r}")


def build_tts(settings: Settings, name: str | None = None) -> TTSProvider:
    name = name or resolve_name("tts", settings)
    if name == "macos_say":
        from .tts.macos_say import MacOSSayTTS

        return MacOSSayTTS(settings.tts)
    if name == "edge":
        from .tts.edge_tts_provider import EdgeTTS

        return EdgeTTS(settings.tts)
    if name == "openai":
        from .tts.openai_tts import OpenAITTS

        return OpenAITTS(settings.tts, api_key=settings.secret(settings.tts.api_key_env))
    if name == "mock":
        from .tts.mock import MockTTS

        return MockTTS()
    raise ProviderError(f"unknown TTS provider: {name!r}")


def build_llm(settings: Settings, name: str | None = None) -> LLMProvider:
    name = name or resolve_name("llm", settings)
    if name == "openai":
        from .llm.openai_llm import OpenAICompatLLM

        key = settings.secret(settings.llm.api_key_env)
        if not key and _is_local_endpoint(settings.llm.base_url):
            # Local servers (Ollama, vLLM, LM Studio) ignore the key entirely.
            key = os.environ.get("VA_LLM__API_KEY") or "not-needed"
        return OpenAICompatLLM(settings.llm, api_key=key)
    if name == "ollama":
        from .llm.openai_llm import OpenAICompatLLM

        patched = settings.llm.model_copy(update={"base_url": OLLAMA_BASE_URL})
        return OpenAICompatLLM(patched, api_key="ollama")
    if name == "echo":
        from .llm.echo import EchoLLM

        return EchoLLM()
    if name == "mock":
        from .llm.mock import MockLLM

        return MockLLM()
    raise ProviderError(f"unknown LLM provider: {name!r}")


def build_all(settings: Settings) -> tuple[ASRProvider, TTSProvider, LLMProvider]:
    return build_asr(settings), build_tts(settings), build_llm(settings)


@dataclass(slots=True)
class ProviderBundle:
    """Keeps the three providers plus their shutdown in one place."""

    asr: ASRProvider
    tts: TTSProvider
    llm: LLMProvider
    names: dict[str, str] = field(default_factory=dict)

    async def warmup(self) -> list[str]:
        """Pre-load anything expensive. Returns the providers that warmed up.

        Optional by design: providers that have nothing to preload simply do
        not define ``warmup``.
        """
        warmed: list[str] = []
        for kind, provider in (("asr", self.asr), ("tts", self.tts), ("llm", self.llm)):
            hook = getattr(provider, "warmup", None)
            if hook is None:
                continue
            try:
                await hook()
                warmed.append(kind)
            except Exception:
                # A failed warm-up must never stop the assistant from booting;
                # the real request will surface the error properly.
                pass
        return warmed

    async def aclose(self) -> None:
        for provider in (self.asr, self.tts, self.llm):
            with contextlib.suppress(Exception):
                await provider.aclose()


def build_bundle(settings: Settings) -> ProviderBundle:
    return ProviderBundle(
        asr=build_asr(settings),
        tts=build_tts(settings),
        llm=build_llm(settings),
        names={
            "asr": resolve_name("asr", settings),
            "tts": resolve_name("tts", settings),
            "llm": resolve_name("llm", settings),
        },
    )


__all__ = [
    "ProviderBundle",
    "ProviderError",
    "ProviderStatus",
    "ProviderUnavailable",
    "asr_status",
    "build_all",
    "build_asr",
    "build_bundle",
    "build_llm",
    "build_tts",
    "capability_report",
    "llm_status",
    "resolve_name",
    "tts_status",
]
