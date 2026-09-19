"""Tool contracts.

A tool is three things: a name, a JSON Schema the model can read, and an async
callable. Everything else — confirmation, timeouts, progress reporting,
redaction of oversized output — is applied uniformly by
:class:`~voiceagent.tools.registry.ToolRegistry`, so individual tools stay
small and hard to get wrong.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..events import Event
from ..providers.base import ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings


@dataclass(slots=True)
class ToolResult:
    """What a tool hands back to the reasoning loop."""

    ok: bool
    content: str
    #: A short, speakable version. When empty the loop summarizes ``content``.
    summary: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    @classmethod
    def success(cls, content: str, *, summary: str = "", **meta: Any) -> ToolResult:
        return cls(ok=True, content=content, summary=summary, meta=meta)

    @classmethod
    def failure(cls, content: str, *, summary: str = "", **meta: Any) -> ToolResult:
        return cls(ok=False, content=content, summary=summary, meta=meta)


@dataclass
class ToolContext:
    """Everything a tool needs from its environment."""

    workspace: Path
    settings: Settings
    session_id: str = "default"
    #: Push a progress event to the front-end (long-running tools only).
    emit: Callable[[Event], Awaitable[None]] | None = None
    #: Set when the user barges in or the session is torn down.
    cancel: asyncio.Event | None = None
    #: Ask the user a yes/no question out loud. Returns False when unavailable.
    confirm: Callable[[str], Awaitable[bool]] | None = None

    async def progress(self, message: str, **extra: Any) -> None:
        if self.emit is not None:
            await self.emit(
                Event(type="tool.progress", data={"message": message, **extra})
            )

    @property
    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def resolve(self, relative: str) -> Path:
        """Resolve a path inside the workspace, refusing escapes."""
        candidate = (self.workspace / relative).expanduser()
        resolved = candidate.resolve()
        workspace = self.workspace.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise PermissionError(
                f"path escapes the workspace: {relative!r} -> {resolved}"
            )
        return resolved


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    #: Requires explicit spoken confirmation before running.
    dangerous: bool = False
    timeout_s: float = 60.0
    #: Emits progress events; the session keeps the mic live while it runs.
    streaming: bool = False
    #: Hidden from the model but callable by the UI (e.g. diagnostics).
    internal: bool = False

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }

    async def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        return await self.handler(call.parsed_args(), ctx)


def object_schema(
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Small helper so tool schemas stay readable."""
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


def string(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": description, **extra}


def integer(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "integer", "description": description, **extra}


def boolean(description: str, default: bool | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "boolean", "description": description}
    if default is not None:
        schema["default"] = default
    return schema


def array(description: str, items: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "array", "description": description, "items": items or {"type": "string"}}


__all__ = [
    "ToolContext",
    "ToolHandler",
    "ToolResult",
    "ToolSpec",
    "array",
    "boolean",
    "integer",
    "object_schema",
    "string",
]
