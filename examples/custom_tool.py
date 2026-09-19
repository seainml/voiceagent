"""Adding a custom tool to voiceagent.

A tool is three things: a name, a JSON Schema the model can read, and an async
function. Everything else — confirmation, timeouts, output truncation, error
formatting, progress reporting — is applied uniformly by the registry, so a
tool cannot accidentally omit it.

Run this to see it registered and callable::

    .venv/bin/python examples/custom_tool.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.config import load_settings  # noqa: E402
from voiceagent.providers.base import ToolCall  # noqa: E402
from voiceagent.tools.base import (  # noqa: E402
    ToolContext,
    ToolResult,
    ToolSpec,
    integer,
    object_schema,
)
from voiceagent.tools.registry import ToolRegistry  # noqa: E402

# --------------------------------------------------------------------------
# 1. The handler. Receives validated-ish args plus a ToolContext.
# --------------------------------------------------------------------------


async def standup_summary(args: dict, ctx: ToolContext) -> ToolResult:
    """Summarize yesterday's git activity for a spoken standup."""
    days = int(args.get("days") or 1)

    # ctx.progress() surfaces a line in the UI while a slow tool runs. Without
    # it, anything taking more than a second feels broken.
    await ctx.progress(f"读取最近 {days} 天的提交…")

    code, out, err = await _git(ctx.workspace, "log", f"--since={days} days ago",
                                "--pretty=%h %an %s")
    if code != 0:
        return ToolResult.failure(f"git failed: {err}", summary="读取提交记录失败了。")

    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        return ToolResult.success(
            "(no commits)", summary=f"最近 {days} 天没有新的提交。", count=0
        )

    body = "\n".join(lines)
    # `summary` is what gets *spoken*; `content` is what the model reads.
    # Keeping them separate is what stops the assistant reading a 200-line
    # git log out loud.
    return ToolResult.success(
        body,
        summary=f"最近 {days} 天有 {len(lines)} 个提交，最后一条是：{lines[0][:80]}",
        count=len(lines),
    )


async def _git(cwd: Path, *argv: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *argv, cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


# --------------------------------------------------------------------------
# 2. The spec. The description is prompt engineering: say *when* to use it.
# --------------------------------------------------------------------------

STANDUP_SPEC = ToolSpec(
    name="standup_summary",
    description=(
        "Summarize recent git commits in the workspace, for a daily standup. "
        "Use it when the user asks what they worked on, or wants a status update."
    ),
    parameters=object_schema(
        {"days": integer("How many days back to look. Default 1.")},
    ),
    handler=standup_summary,
    dangerous=False,   # read-only, so no spoken confirmation
    timeout_s=30.0,
    streaming=True,    # it calls ctx.progress()
)


# --------------------------------------------------------------------------
# 3. Register it. In the real app this is one line in tools/builtin.py.
# --------------------------------------------------------------------------


async def main() -> None:
    settings = load_settings()
    tools = ToolRegistry([STANDUP_SPEC], settings)

    print("registered:", tools.names())
    print("needs confirmation:", tools.dangerous() or "(none)")
    print("schema the model sees:")
    print(" ", tools.openai_schema()[0]["function"]["description"][:90], "…")

    ctx = ToolContext(workspace=Path.cwd(), settings=settings)
    result = await tools.call(
        ToolCall(id="c1", name="standup_summary", arguments='{"days": 7}'), ctx
    )
    print("\nok:", result.ok)
    print("spoken summary:", result.summary)
    print("model sees:", result.content[:200])


if __name__ == "__main__":
    asyncio.run(main())
