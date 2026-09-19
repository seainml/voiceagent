"""Wrap external coding agents (Claude Code, Codex, DSH) as callable tools.

Design notes, learned the hard way:

* These CLIs are **long-running and chatty**. Running them to completion in
  silence makes a voice assistant feel broken, so every adapter streams
  progress lines back to the front-end while it works.
* Their stdout formats are *not* stable API contracts. Each parser therefore
  falls back to plain-text accumulation when JSON parsing fails, so a CLI
  upgrade degrades quality rather than breaking the tool.
* They all need different flags to run non-interactively. Those live in config
  (``[tools.agents.<name>]``) rather than in code.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import AgentCLISettings, Settings
from .base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    object_schema,
    string,
)
from .exec import ExecResult, format_command, run_streaming

#: tool name -> (config key, human label, when to reach for it)
AGENT_TOOLS: dict[str, tuple[str, str, str]] = {
    "ask_claude_code": (
        "claude",
        "Claude Code",
        "best for multi-file refactors, careful reasoning and long autonomous tasks",
    ),
    "ask_codex": (
        "codex",
        "Codex CLI",
        "best for focused code changes and quick implementation work",
    ),
    "ask_dsh": (
        "dsh",
        "DeepSeek Harness",
        "general-purpose agent for research, writing, analysis and code",
    ),
}

MAX_PROGRESS_CHARS = 160


# --------------------------------------------------------------------------
# stream parsers
# --------------------------------------------------------------------------


class AgentStreamParser:
    """Base class: turn provider output into progress lines + a final answer."""

    def feed(self, stream: str, line: str) -> str | None:
        return None

    def finish(self, result: ExecResult) -> tuple[bool, str, dict[str, Any]]:
        text = result.combined().strip()
        return result.ok, text, {}


class ClaudeStreamParser(AgentStreamParser):
    """``claude --print --output-format stream-json --verbose``."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.tools: list[str] = []
        self.final: str | None = None
        self.is_error = False
        self.meta: dict[str, Any] = {}

    def feed(self, stream: str, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None
        if stream != "stdout" or not stripped.startswith("{"):
            return stripped[:MAX_PROGRESS_CHARS]
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped[:MAX_PROGRESS_CHARS]

        kind = event.get("type")
        if kind == "system":
            return f"{event.get('model', 'Claude Code')} 已就绪"
        if kind == "assistant":
            notes: list[str] = []
            content = (event.get("message") or {}).get("content") or []
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and block.get("text", "").strip():
                        text = block["text"].strip()
                        self.texts.append(text)
                        notes.append(text[:MAX_PROGRESS_CHARS])
                    elif block.get("type") == "tool_use":
                        name = str(block.get("name") or "tool")
                        self.tools.append(name)
                        notes.append(f"调用 {name}")
            return " | ".join(notes) or None
        if kind == "result":
            result_text = event.get("result")
            if isinstance(result_text, str):
                self.final = result_text
            self.is_error = bool(event.get("is_error"))
            if event.get("total_cost_usd") is not None:
                self.meta["cost_usd"] = event["total_cost_usd"]
            if event.get("duration_ms") is not None:
                self.meta["agent_duration_ms"] = event["duration_ms"]
            return None
        return None

    def finish(self, result: ExecResult) -> tuple[bool, str, dict[str, Any]]:
        text = (self.final or "").strip() or "\n".join(self.texts).strip()
        if not text:
            text = result.combined().strip()
        meta = {**self.meta}
        if self.tools:
            meta["tools_used"] = sorted(set(self.tools))
        ok = result.ok and not self.is_error and bool(text)
        return ok, text, meta


class CodexJSONParser(AgentStreamParser):
    """``codex exec --json`` — JSONL events, with a plain-text safety net."""

    ITEM_LABELS = {
        "command_execution": "执行命令",
        "file_change": "修改文件",
        "reasoning": "思考",
        "web_search": "搜索",
        "mcp_tool_call": "调用工具",
    }

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.plain: list[str] = []
        self.error: str | None = None
        self.meta: dict[str, Any] = {}
        self.saw_json = False

    def feed(self, stream: str, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None
        if not stripped.startswith("{"):
            if stream == "stdout":
                self.plain.append(stripped)
            return stripped[:MAX_PROGRESS_CHARS]
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            if stream == "stdout":
                self.plain.append(stripped)
            return stripped[:MAX_PROGRESS_CHARS]
        self.saw_json = True

        kind = str(event.get("type") or "")
        if kind in {"error", "turn.failed"}:
            self.error = str(event.get("message") or event.get("error") or kind)
            return f"出错：{self.error[:MAX_PROGRESS_CHARS]}"
        if kind in {"thread.started", "turn.started"}:
            return "Codex 已启动"
        if kind == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                self.meta["usage"] = usage
            return None

        item = event.get("item") or {}
        if not isinstance(item, dict):
            return None
        item_type = str(item.get("type") or "")
        if item_type == "agent_message":
            text = str(item.get("text") or "").strip()
            if text:
                self.texts.append(text)
                return text[:MAX_PROGRESS_CHARS]
            return None
        if item_type in self.ITEM_LABELS:
            detail = (
                item.get("command")
                or item.get("path")
                or item.get("text")
                or item.get("query")
                or ""
            )
            label = self.ITEM_LABELS[item_type]
            detail = str(detail).strip().replace("\n", " ")[:100]
            return f"{label}：{detail}" if detail else label
        return None

    def finish(self, result: ExecResult) -> tuple[bool, str, dict[str, Any]]:
        text = "\n".join(self.texts).strip()
        if not text and self.plain:
            text = "\n".join(self.plain).strip()
        if not text:
            text = result.combined().strip()
        ok = result.ok and not self.error and bool(text)
        if self.error and not text:
            text = self.error
        return ok, text, dict(self.meta)


class PlainTextParser(AgentStreamParser):
    """Fallback for CLIs that simply print their answer (e.g. ``dsh``)."""

    def __init__(self) -> None:
        self.stdout: list[str] = []

    def feed(self, stream: str, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None
        if stream == "stdout":
            self.stdout.append(stripped)
            return stripped[:MAX_PROGRESS_CHARS]
        return f"[stderr] {stripped[:MAX_PROGRESS_CHARS]}"

    def finish(self, result: ExecResult) -> tuple[bool, str, dict[str, Any]]:
        text = "\n".join(self.stdout).strip() or result.combined().strip()
        return result.ok and bool(text), text, {}


PARSERS: dict[str, Callable[[], AgentStreamParser]] = {
    "claude": ClaudeStreamParser,
    "codex": CodexJSONParser,
    "dsh": PlainTextParser,
}


# --------------------------------------------------------------------------
# handler
# --------------------------------------------------------------------------


def _build_prompt(cfg: AgentCLISettings, task: str) -> str:
    preamble = (cfg.preamble or "").strip()
    return f"{preamble}\n\n---\n\n{task}" if preamble else task


def _extra_args(cfg: AgentCLISettings, settings: Settings) -> list[str]:
    if settings.tools.allow_unsafe_agent_writes:
        return list(cfg.unsafe_args)
    return []


def _diagnose(text: str) -> str | None:
    """Turn the CLIs' common failure strings into something actionable."""
    lowered = text.lower()
    if "not logged in" in lowered or "authentication_failed" in lowered:
        return "agent-not-authenticated"
    if "please run /login" in lowered:
        return "agent-not-authenticated"
    # codex reports auth problems as a generic 401 from the Responses API.
    if "401" in lowered and any(
        marker in lowered
        for marker in ("unauthorized", "invalid_api_key", "incorrect api key", "auth error")
    ):
        return "agent-not-authenticated"
    return None


def make_agent_handler(tool_name: str) -> Callable[[dict[str, Any], ToolContext], Any]:
    config_key, label, _ = AGENT_TOOLS[tool_name]

    async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        task = str(args.get("task") or "").strip()
        if not task:
            return ToolResult.failure("no task provided")

        cfg = ctx.settings.tools.agents.get(config_key)
        if cfg is None or not cfg.enabled:
            return ToolResult.failure(f"{label} is not configured or is disabled")

        cwd = ctx.workspace
        if args.get("cwd"):
            try:
                cwd = ctx.resolve(str(args["cwd"]))
            except PermissionError as exc:
                return ToolResult.failure(str(exc))
        elif cfg.working_dir:
            cwd = Path(cfg.working_dir).expanduser()

        argv = [cfg.command, *cfg.args, *_extra_args(cfg, ctx.settings)]
        tail: list[str] = []
        if config_key == "codex" and "-o" not in argv and "--output-last-message" not in argv:
            tmp = ctx.workspace / ".voiceagent" / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            last_message = tmp / f"codex-{int(time.time() * 1000)}.txt"
            argv += ["-o", str(last_message)]
            tail.append(str(last_message))
        argv.append(_build_prompt(cfg, task))

        parser = PARSERS.get(config_key, PlainTextParser)()
        await ctx.progress(f"把任务交给 {label}：{task[:120]}")

        async def on_line(stream: str, text: str) -> None:
            note = parser.feed(stream, text)
            if note:
                await ctx.progress(note, agent=config_key)

        result = await run_streaming(
            argv,
            ctx=ctx,
            cwd=cwd,
            timeout_s=float(args.get("timeout_s") or cfg.timeout_s),
            env=cfg.env,
            on_line=on_line,
            max_chars=ctx.settings.tools.max_output_chars,
        )

        if result.cancelled:
            return ToolResult.failure(
                f"{label} 运行被取消", summary=f"已经中止了{label}的任务。", agent=config_key
            )
        if result.timed_out:
            return ToolResult.failure(
                f"{label} timed out after {result.duration_s:.0f}s",
                summary=f"{label} 运行超时了。",
                agent=config_key,
            )

        # Prefer a file-based final message when the CLI supports one: it is
        # the most stable contract codex offers across versions.
        file_final: str | None = None
        if tail:
            try:
                candidate = Path(tail[0]).read_text("utf-8", "replace").strip()
                if candidate:
                    file_final = candidate
            except OSError:
                pass

        ok, text, meta = parser.finish(result)
        if file_final:
            text = file_final
            ok = result.ok and not getattr(parser, "error", None)
        meta.update(
            {
                "agent": config_key,
                "command": format_command(argv[:-1]),
                "returncode": result.returncode,
                "duration_s": round(result.duration_s, 2),
            }
        )

        problem = _diagnose(text)
        if problem == "agent-not-authenticated":
            hints = {
                "claude": "在终端运行 `claude` 后执行 `/login`，或设置 ANTHROPIC_API_KEY。",
                "codex": "在终端运行 `codex login`，或设置有效的 OPENAI_API_KEY。",
                "dsh": "检查 dsh 的模型配置。",
            }
            return ToolResult.failure(
                f"{label} is not authenticated. {text[:400]}",
                summary=f"{label}还没有登录。{hints.get(config_key, '')}",
                **meta,
            )
        if not ok:
            return ToolResult.failure(
                f"{label} failed (exit {result.returncode}):\n{text[-3000:]}",
                summary=f"{label}执行失败了，退出码 {result.returncode}。",
                **meta,
            )
        return ToolResult.success(text[-12_000:], **meta)

    return handler


def build_agent_specs(settings: Settings) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for tool_name, (config_key, label, when) in AGENT_TOOLS.items():
        cfg = settings.tools.agents.get(config_key)
        if cfg is None or not cfg.enabled:
            continue
        specs.append(
            ToolSpec(
                name=tool_name,
                description=(
                    f"Delegate a task to {label}, a full coding agent running in the "
                    f"workspace ({when}). Use it for anything that needs many steps, "
                    "edits files, or takes longer than a few seconds — it runs "
                    "autonomously and streams progress. Give it a complete, "
                    "self-contained task description."
                ),
                parameters=object_schema(
                    {
                        "task": string(
                            "The complete task for the agent, including any context "
                            "it needs. It cannot see this conversation."
                        ),
                        "cwd": string("Optional subdirectory of the workspace to run in."),
                    },
                    required=["task"],
                ),
                handler=make_agent_handler(tool_name),
                dangerous=True,
                streaming=True,
                timeout_s=cfg.timeout_s + 30.0,
            )
        )
    return specs


__all__ = [
    "AGENT_TOOLS",
    "ClaudeStreamParser",
    "CodexJSONParser",
    "PlainTextParser",
    "build_agent_specs",
    "make_agent_handler",
]
