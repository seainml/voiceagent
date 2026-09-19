"""Shared subprocess execution.

Both ``run_shell`` and the agent-CLI adapters need the same things: stream
output as it arrives, enforce a timeout, honour barge-in cancellation, cap
output size and never let a child inherit a hang. Written once here.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import ToolContext

#: Keep well under the model's context: tools are chatty.
DEFAULT_MAX_CHARS = 200_000


@dataclass(slots=True)
class ExecResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    cancelled: bool = False
    duration_s: float = 0.0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.cancelled

    def combined(self) -> str:
        out = self.stdout.strip()
        err = self.stderr.strip()
        if out and err:
            return f"{out}\n\n[stderr]\n{err}"
        return out or err


async def run_streaming(
    argv: Sequence[str] | str,
    *,
    ctx: ToolContext,
    cwd: Path | str | None = None,
    timeout_s: float = 60.0,
    shell: bool = False,
    env: dict[str, str] | None = None,
    on_line: Callable[[str, str], Awaitable[None] | None] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    stdin_data: str | None = None,
) -> ExecResult:
    """Run a command, streaming lines to ``on_line(stream, text)``.

    ``on_line`` is invoked for every line as it appears — that is how a
    five-minute coding-agent run can narrate progress instead of going silent.
    """
    started = time.monotonic()
    child_env = {**os.environ, **(env or {})}
    # Agents spawn their own tools; give them a sane, non-interactive env.
    child_env.setdefault("NO_COLOR", "1")
    child_env.setdefault("TERM", "dumb")
    child_env["PYTHONUNBUFFERED"] = "1"

    workdir = Path(cwd).expanduser() if cwd else ctx.workspace

    if shell:
        proc = await asyncio.create_subprocess_shell(
            str(argv),
            cwd=str(workdir),
            env=child_env,
            stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(workdir),
            env=child_env,
            stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    result = ExecResult(returncode=-1)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    total = 0

    async def pump(stream: asyncio.StreamReader | None, name: str, sink: list[str]) -> None:
        nonlocal total
        if stream is None:
            return
        while True:
            raw = await stream.readline()
            if not raw:
                break
            text = raw.decode("utf-8", "replace").rstrip("\n")
            if total < max_chars:
                sink.append(text)
                total += len(text)
            elif not result.truncated:
                result.truncated = True
                sink.append("… [output truncated]")
            if on_line is not None:
                outcome = on_line(name, text)
                if asyncio.iscoroutine(outcome):
                    await outcome

    cancel_task: asyncio.Task[None] | None = None

    async def watch_cancel() -> None:
        if ctx.cancel is None:
            return
        await ctx.cancel.wait()
        with contextlib.suppress(ProcessLookupError):
            proc.kill()

    try:
        if ctx.cancel is not None:
            cancel_task = asyncio.create_task(watch_cancel())

        if stdin_data and proc.stdin is not None:
            proc.stdin.write(stdin_data.encode("utf-8"))
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await proc.stdin.drain()
            proc.stdin.close()

        pumps = asyncio.gather(
            pump(proc.stdout, "stdout", stdout_parts),
            pump(proc.stderr, "stderr", stderr_parts),
        )
        try:
            async with asyncio.timeout(timeout_s):
                await proc.wait()
                await pumps
        except TimeoutError:
            result.timed_out = True
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            pumps.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pumps
    finally:
        if cancel_task is not None:
            cancel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_task
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    if ctx.cancelled:
        result.cancelled = True
    result.returncode = proc.returncode if proc.returncode is not None else -1
    result.stdout = "\n".join(stdout_parts)
    result.stderr = "\n".join(stderr_parts)
    result.duration_s = time.monotonic() - started
    return result


def format_command(argv: Sequence[str] | str) -> str:
    if isinstance(argv, str):
        return argv
    return " ".join(shlex.quote(a) for a in argv)


__all__ = ["DEFAULT_MAX_CHARS", "ExecResult", "format_command", "run_streaming"]
