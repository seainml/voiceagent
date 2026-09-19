"""Events flowing between the session orchestrator and any front-end.

The same objects are used for in-process callbacks and for the JSON frames on
the WebSocket, so a front-end only has to understand one vocabulary.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    # lifecycle / state machine
    READY = "ready"
    STATE = "state"
    ERROR = "error"
    LOG = "log"
    METRIC = "metric"

    # user side
    PARTIAL_TRANSCRIPT = "partial_transcript"
    TRANSCRIPT = "transcript"

    # assistant side
    ASSISTANT_DELTA = "assistant.delta"
    ASSISTANT_DONE = "assistant.done"

    # audio out
    AUDIO = "audio"
    AUDIO_END = "audio.end"

    # transport
    PONG = "pong"

    # tools
    TOOL_CALL = "tool.call"
    TOOL_PROGRESS = "tool.progress"
    TOOL_RESULT = "tool.result"

    # human-in-the-loop
    CONFIRM_REQUEST = "confirm.request"


class PipelineState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    ACTING = "acting"
    SPEAKING = "speaking"
    AWAITING_CONFIRMATION = "awaiting_confirmation"


class Event(BaseModel):
    """A single frame. ``data`` stays open-ended on purpose."""

    type: EventType
    ts: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.type.value, "ts": self.ts, **self.data}

    # -- constructors ----------------------------------------------------

    @classmethod
    def state(cls, state: PipelineState | str, **extra: Any) -> Event:
        value = state.value if isinstance(state, PipelineState) else state
        return cls(type=EventType.STATE, data={"state": value, **extra})

    @classmethod
    def transcript(cls, text: str, *, final: bool, **extra: Any) -> Event:
        return cls(
            type=EventType.TRANSCRIPT if final else EventType.PARTIAL_TRANSCRIPT,
            data={"text": text, **extra},
        )

    @classmethod
    def assistant_delta(cls, text: str) -> Event:
        return cls(type=EventType.ASSISTANT_DELTA, data={"text": text})

    @classmethod
    def assistant_done(cls, text: str, **extra: Any) -> Event:
        return cls(type=EventType.ASSISTANT_DONE, data={"text": text, **extra})

    @classmethod
    def audio(cls, payload: bytes, mime: str, sample_rate: int | None, *, seq: int = 0) -> Event:
        import base64

        return cls(
            type=EventType.AUDIO,
            data={
                "mime": mime,
                "sample_rate": sample_rate,
                "seq": seq,
                "data": base64.b64encode(payload).decode("ascii"),
            },
        )

    @classmethod
    def audio_end(cls, *, seq: int = 0, reason: str = "complete") -> Event:
        """All audio for this turn has been sent.

        Distinct from ``assistant.done``, which only means the *text* is
        complete — the speaker usually keeps going for a while after that.
        """
        return cls(type=EventType.AUDIO_END, data={"seq": seq, "reason": reason})

    @classmethod
    def error(cls, message: str, *, where: str = "") -> Event:
        return cls(type=EventType.ERROR, data={"message": message, "where": where})

    @classmethod
    def log(cls, message: str, level: str = "info") -> Event:
        return cls(type=EventType.LOG, data={"message": message, "level": level})

    @classmethod
    def metric(cls, name: str, value: float, **extra: Any) -> Event:
        return cls(type=EventType.METRIC, data={"name": name, "value": value, **extra})
