"""Tool registry: lookup, uniform guardrails, OpenAI schema export.

Individual tools know nothing about timeouts, truncation or error formatting —
those are applied once, here, so a new tool cannot accidentally omit them.
"""

from __future__ import annotations

import asyncio
import contextlib
import traceback
from typing import Any

from ..config import Settings
from ..providers.base import ToolCall
from .base import ToolContext, ToolResult, ToolSpec
from .mcp import MCPClient


class ToolRegistry:
    def __init__(
        self,
        specs: list[ToolSpec],
        settings: Settings,
        *,
        mcp_clients: list[MCPClient] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self._specs: dict[str, ToolSpec] = {spec.name: spec for spec in specs}
        self.settings = settings
        self.mcp_clients = list(mcp_clients or [])
        self.warnings = list(warnings or [])

    # -- introspection ---------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in self.names()]

    def dangerous(self) -> set[str]:
        """Tool names that must be confirmed before they run."""
        configured = set(self.settings.tools.confirm)
        declared = {s.name for s in self._specs.values() if s.dangerous}
        return (configured | declared) & set(self._specs)

    def openai_schema(self) -> list[dict[str, Any]]:
        return [
            spec.to_openai()
            for spec in self.specs()
            if not spec.internal
        ]

    # -- execution -------------------------------------------------------

    async def call(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Run a tool with a timeout, catching everything it throws."""
        spec = self._specs.get(call.name)
        if spec is None:
            known = ", ".join(self.names()[:25])
            return ToolResult.failure(
                f"unknown tool {call.name!r}. Available tools: {known}",
                summary="我没有这个工具。",
            )

        limit = self.settings.tools.max_output_chars
        try:
            async with asyncio.timeout(spec.timeout_s):
                result = await spec.invoke(call, ctx)
        except TimeoutError:
            return ToolResult.failure(
                f"{call.name} exceeded its {spec.timeout_s:.0f}s budget",
                summary="这个操作超时了。",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            detail = traceback.format_exc(limit=6)
            return ToolResult.failure(
                f"{call.name} raised {type(exc).__name__}: {exc}\n{detail[-1500:]}",
                summary="工具执行时出错了。",
            )

        if result is not None and len(result.content) > limit:
            result.content = result.content[:limit] + "\n… [truncated]"
            result.truncated = True
        return result or ToolResult.failure(f"{call.name} returned nothing")

    async def aclose(self) -> None:
        for client in self.mcp_clients:
            with contextlib.suppress(Exception):
                await client.aclose()
        self.mcp_clients.clear()


__all__ = ["ToolRegistry"]
