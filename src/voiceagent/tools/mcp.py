"""Minimal MCP (Model Context Protocol) client over stdio.

This is what makes the agent open-ended: anything with an MCP server — a notes
app, a calendar, a browser, a database, someone else's SaaS — becomes a voice
tool without writing Python. Only the three methods that matter for tool use
are implemented (``initialize``, ``tools/list``, ``tools/call``); sampling and
roots are intentionally out of scope.

Servers are treated as hostile-ish infrastructure: every call is bounded by a
timeout, a dead server fails its pending requests instead of hanging the turn,
and stderr is captured for diagnostics rather than dumped into the transcript.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections import deque
from pathlib import Path
from typing import Any

from ..audio.pcm import PCM_MIME  # noqa: F401  (kept for symmetry of imports)
from ..config import Settings
from .base import ToolContext, ToolResult, ToolSpec, object_schema

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "voiceagent", "version": "0.1.0"}

#: Tool-name fragments that usually mean "this changes something".
DANGEROUS_HINTS = (
    "write", "edit", "delete", "remove", "create", "move", "rename",
    "exec", "run", "shell", "command", "post", "put", "patch", "send",
    "update", "set", "upload", "install",
)


def looks_dangerous(name: str, annotations: dict[str, Any] | None = None) -> bool:
    """Best-effort risk guess so MCP tools get the confirmation treatment."""
    if annotations:
        if annotations.get("destructiveHint") is True:
            return True
        if annotations.get("readOnlyHint") is True:
            return False
    lowered = name.lower()
    return any(hint in lowered for hint in DANGEROUS_HINTS)


class MCPError(RuntimeError):
    pass


class MCPClient:
    """One MCP server subprocess."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = str(cwd) if cwd else None
        self.timeout_s = timeout_s

        self._proc: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._reader: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=30)
        self._tools: list[dict[str, Any]] = []
        self.server_info: dict[str, Any] = {}

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        child_env = {**os.environ, **self.env}
        child_env.setdefault("NO_COLOR", "1")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                cwd=self.cwd,
                env=child_env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise MCPError(
                f"MCP server {self.name!r}: command not found: {self.command}"
            ) from exc

        self._reader = asyncio.create_task(self._read_loop())
        asyncio.create_task(self._drain_stderr())
        await self._initialize()

    async def _initialize(self) -> None:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": CLIENT_INFO,
            },
        )
        self.server_info = result.get("serverInfo", {}) if isinstance(result, dict) else {}
        await self._notify("notifications/initialized", {})

    async def aclose(self) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(MCPError(f"MCP server {self.name!r} is shutting down"))
        self._pending.clear()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                async with asyncio.timeout(3):
                    await proc.wait()
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
        self._proc = None

    # -- transport -------------------------------------------------------

    async def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise MCPError(f"MCP server {self.name!r} is not running")
        data = json.dumps(payload, ensure_ascii=False) + "\n"
        proc.stdin.write(data.encode("utf-8"))
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await proc.stdin.drain()

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def request(
        self, method: str, params: dict[str, Any] | None = None, *, timeout_s: float | None = None
    ) -> Any:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
            )
            payload = await asyncio.wait_for(future, timeout_s or self.timeout_s)
        except TimeoutError as exc:
            raise MCPError(
                f"MCP server {self.name!r} did not answer {method!r} within "
                f"{timeout_s or self.timeout_s:.0f}s"
            ) from exc
        finally:
            self._pending.pop(request_id, None)

        if isinstance(payload, dict) and payload.get("error"):
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise MCPError(f"{method} failed: {message}")
        return payload.get("result") if isinstance(payload, dict) else None

    async def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._stderr_tail.append(f"[non-json stdout] {line[:200]}")
                    continue
                self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self._stderr_tail.append(f"[reader error] {exc}")
        finally:
            error = MCPError(f"MCP server {self.name!r} closed the connection")
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)
            self._pending.clear()

    def _dispatch(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            future = self._pending.get(int(message["id"]))
            if future is not None and not future.done():
                future.set_result(message)
            return
        method = message.get("method")
        if method and "id" in message:
            # Server-initiated request (sampling/roots). Politely refuse.
            asyncio.create_task(
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {"code": -32601, "message": f"{method} is not supported"},
                    }
                )
            )

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        with contextlib.suppress(asyncio.CancelledError, Exception):
            while True:
                raw = await proc.stderr.readline()
                if not raw:
                    break
                self._stderr_tail.append(raw.decode("utf-8", "replace").rstrip())

    # -- tools -----------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self.request("tools/list", {})
        tools = (result or {}).get("tools") if isinstance(result, dict) else None
        self._tools = [t for t in (tools or []) if isinstance(t, dict)]
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = await self.request(
                "tools/call", {"name": name, "arguments": arguments}
            )
        except MCPError as exc:
            return ToolResult.failure(str(exc), summary="工具调用失败了。")

        if not isinstance(result, dict):
            return ToolResult.failure("MCP returned an unexpected payload")

        parts: list[str] = []
        for block in result.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "resource":
                resource = block.get("resource") or {}
                parts.append(str(resource.get("text") or resource.get("uri") or ""))
            else:
                parts.append(json.dumps(block, ensure_ascii=False)[:1000])

        text = "\n".join(p for p in parts if p).strip() or "(no content)"
        if result.get("isError"):
            return ToolResult.failure(text, summary=f"{name} 返回了错误。")
        return ToolResult.success(text)

    @property
    def diagnostics(self) -> str:
        return "\n".join(self._stderr_tail)


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def _tool_name(server: str, tool: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in tool)
    return f"mcp__{server}__{safe}"


def _make_handler(client: MCPClient, remote_name: str):
    async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        await ctx.progress(f"调用 {client.name} / {remote_name}")
        return await client.call_tool(remote_name, args)

    return handler


async def connect_servers(settings: Settings) -> tuple[list[ToolSpec], list[MCPClient], list[str]]:
    """Start every configured MCP server and turn its tools into ToolSpecs.

    Failures are collected as warnings rather than raised: one broken MCP
    server must not stop the assistant from booting.
    """
    specs: list[ToolSpec] = []
    clients: list[MCPClient] = []
    warnings: list[str] = []

    for server_name, raw in (settings.tools.mcp or {}).items():
        if not isinstance(raw, dict) or not raw.get("command"):
            warnings.append(f"MCP server {server_name!r}: missing 'command'")
            continue
        client = MCPClient(
            server_name,
            str(raw["command"]),
            [str(a) for a in raw.get("args", [])],
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            cwd=raw.get("cwd"),
            timeout_s=float(raw.get("timeout_s") or 30.0),
        )
        try:
            await client.start()
            tools = await client.list_tools()
        except Exception as exc:
            warnings.append(f"MCP server {server_name!r} unavailable: {exc}")
            await client.aclose()
            continue

        clients.append(client)
        for tool in tools:
            remote = str(tool.get("name") or "")
            if not remote:
                continue
            schema = tool.get("inputSchema")
            if not isinstance(schema, dict):
                schema = object_schema()
            annotations = tool.get("annotations")
            specs.append(
                ToolSpec(
                    name=_tool_name(server_name, remote),
                    description=(
                        f"[MCP:{server_name}] {tool.get('description') or remote}"
                    )[:1000],
                    parameters=schema,
                    handler=_make_handler(client, remote),
                    dangerous=looks_dangerous(remote, annotations
                                              if isinstance(annotations, dict) else None),
                    timeout_s=float(raw.get("tool_timeout_s") or 120.0),
                )
            )
    return specs, clients, warnings


__all__ = ["MCPClient", "MCPError", "connect_servers", "looks_dangerous"]
