"""Shell execution with a denylist, a timeout and spoken confirmation.

This is the most dangerous tool in the box, so it gets the most guardrails:
it never runs without confirmation, it is confined to the workspace, and the
denylist rejects the handful of patterns that destroy a machine rather than
just annoy you. A denylist is not a sandbox — treat confirmation as the real
control and the denylist as a seatbelt.
"""

from __future__ import annotations

import re
from typing import Any

from .base import ToolContext, ToolResult, ToolSpec, integer, object_schema, string
from .exec import format_command, run_streaming

LAST_LINES = 25


def _blocked(command: str, denylist: list[str]) -> str | None:
    lowered = command.lower()
    for pattern in denylist:
        if pattern.lower() in lowered:
            return pattern
    # Extra structural checks that a substring list cannot express.
    if re.search(r"\brm\s+(-[a-z]*\s+)*-[a-z]*r[a-z]*f?\s+/(\s|$)", lowered):
        return "rm -rf /"
    return None


async def run_shell(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = str(args.get("command", "")).strip()
    if not command:
        return ToolResult.failure("no command provided")

    blocked = _blocked(command, ctx.settings.tools.shell_denylist)
    if blocked:
        return ToolResult.failure(
            f"refused: the command matches the denylist pattern {blocked!r}",
            summary="这个命令被安全策略拒绝了。",
        )

    timeout_s = float(args.get("timeout_s") or ctx.settings.tools.shell_timeout_s)
    cwd = ctx.workspace
    if args.get("cwd"):
        try:
            cwd = ctx.resolve(str(args["cwd"]))
        except PermissionError as exc:
            return ToolResult.failure(str(exc))

    await ctx.progress(f"运行命令：{command[:120]}")

    async def on_line(stream: str, text: str) -> None:
        if text.strip() and stream == "stdout":
            await ctx.progress(text.strip()[:200])

    result = await run_streaming(
        command,
        ctx=ctx,
        cwd=cwd,
        timeout_s=timeout_s,
        shell=True,
        on_line=on_line,
        max_chars=ctx.settings.tools.max_output_chars,
    )

    if result.cancelled:
        return ToolResult.failure("cancelled by the user", summary="已取消。")
    if result.timed_out:
        return ToolResult.failure(
            f"timed out after {timeout_s:.0f}s\n{_tail(result.combined())}",
            summary=f"命令超时了，超过 {timeout_s:.0f} 秒。",
        )

    body = _tail(result.combined())
    meta = {
        "command": command,
        "returncode": result.returncode,
        "duration_s": round(result.duration_s, 2),
    }
    if result.ok:
        summary = _summarize_success(body)
        return ToolResult.success(body or "(no output)", summary=summary, **meta)
    return ToolResult.failure(
        f"exit code {result.returncode}\n{body or '(no output)'}",
        summary=f"命令失败了，退出码 {result.returncode}。",
        **meta,
    )


def _tail(text: str, lines: int = LAST_LINES) -> str:
    parts = text.splitlines()
    if len(parts) <= lines:
        return text
    return "\n".join(["… (earlier output omitted)", *parts[-lines:]])


def _summarize_success(body: str) -> str:
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return "命令执行成功，没有输出。"
    return f"执行成功，输出了 {len(lines)} 行，最后一行的内容是：{lines[-1][:120]}"


SPEC = ToolSpec(
    name="run_shell",
    description=(
        "Run a shell command inside the workspace and return its output. "
        "Use for quick local operations (listing, git, build, test). "
        "Long-running or code-editing work should go to ask_claude_code / ask_codex / ask_dsh instead."
    ),
    parameters=object_schema(
        {
            "command": string("The shell command to run."),
            "cwd": string("Optional subdirectory of the workspace to run in."),
            "timeout_s": integer("Seconds before the command is killed. Default 120."),
        },
        required=["command"],
    ),
    handler=run_shell,
    dangerous=True,
    streaming=True,
    timeout_s=180.0,
)

__all__ = ["SPEC", "format_command", "run_shell"]
