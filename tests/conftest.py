"""Shared fixtures.

Every test runs against mock providers so the suite is fast, offline and
deterministic — the real providers are exercised by `voiceagent doctor` and by
`tests/test_providers_live.py`, which skips itself when nothing is available.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import pytest_asyncio

from voiceagent.config import Settings
from voiceagent.events import Event, EventType
from voiceagent.providers.asr.mock import MockASR
from voiceagent.providers.base import ToolCall
from voiceagent.providers.llm.mock import MockLLM
from voiceagent.providers.registry import ProviderBundle
from voiceagent.providers.tts.mock import MockTTS
from voiceagent.session import VoiceSession
from voiceagent.tools.builtin import build_registry


def make_settings(**overrides: Any) -> Settings:
    """Settings with mock providers and a temporary workspace."""
    base: dict[str, Any] = {
        "asr": {"provider": "mock", "language": "zh"},
        "tts": {"provider": "mock"},
        "llm": {"provider": "mock", "model": "mock"},
        "pipeline": {
            "sentence_flush_chars": 4,
            "max_sentence_chars": 40,
            "max_spoken_chars": 200,
            "history_turns": 6,
            "asr_min_utterance_ms": 0,
        },
        "tools": {
            "enabled": [
                "ask_claude_code", "ask_codex", "ask_dsh",
                "remember", "recall", "read_file", "list_files",
                "write_file", "run_shell",
            ],
            "confirm": ["run_shell", "write_file", "ask_claude_code", "ask_codex", "ask_dsh"],
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return Settings(**base)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tools={"workspace": str(tmp_path)})


class EventCollector:
    """Records every event a session emits."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.done = asyncio.Event()

    async def __call__(self, event: Event) -> None:
        self.events.append(event)
        if event.type is EventType.ASSISTANT_DONE:
            self.done.set()

    # -- queries ---------------------------------------------------------

    def of(self, kind: EventType) -> list[Event]:
        return [e for e in self.events if e.type is kind]

    @property
    def states(self) -> list[str]:
        return [str(e.data.get("state")) for e in self.of(EventType.STATE)]

    @property
    def audio(self) -> list[bytes]:
        return [e.data.get("data", "") for e in self.of(EventType.AUDIO)]

    def text_of(self, kind: EventType) -> str:
        return "".join(str(e.data.get("text", "")) for e in self.of(kind))

    def names(self) -> list[str]:
        return [str(e.data.get("name")) for e in self.of(EventType.TOOL_CALL)]

    def metrics(self) -> dict[str, float]:
        return {str(e.data.get("name")): float(e.data.get("value", 0))
                for e in self.of(EventType.METRIC)}

    def clear(self) -> None:
        self.events.clear()
        self.done.clear()


@pytest.fixture
def collector() -> EventCollector:
    return EventCollector()


def make_bundle(
    *,
    asr: Any = None,
    tts: Any = None,
    llm: Any = None,
    script: list[Any] | None = None,
) -> ProviderBundle:
    return ProviderBundle(
        asr=asr or MockASR(),
        tts=tts or MockTTS(),
        llm=llm or MockLLM(script=script),
        names={"asr": "mock", "tts": "mock", "llm": "mock"},
    )


@pytest_asyncio.fixture
async def session_factory(settings, collector):
    """Build sessions that clean themselves up."""
    created: list[VoiceSession] = []

    async def factory(
        *,
        bundle: ProviderBundle | None = None,
        auto_confirm: bool = False,
        registry=None,
        emit_audio: bool = True,
    ) -> VoiceSession:
        registry = registry or await build_registry(settings, include_mcp=False)
        session = VoiceSession(
            settings,
            bundle or make_bundle(),
            registry,
            on_event=collector,
            workspace=settings.workspace_path(),
            auto_confirm=auto_confirm,
            emit_audio_payloads=emit_audio,
        )
        created.append(session)
        await session.start()
        return session

    yield factory

    for session in created:
        await session.aclose()


@pytest_asyncio.fixture
async def registry(settings):
    reg = await build_registry(settings, include_mcp=False)
    yield reg
    await reg.aclose()


def tool_call(name: str, arguments: str = "{}", call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


async def wait_for(predicate, timeout: float = 5.0) -> None:
    """Poll until ``predicate()`` is truthy, or fail."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met before timeout")


@pytest.fixture
def anyio_backend() -> str:  # pragma: no cover - compatibility shim
    return "asyncio"
