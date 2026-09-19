"""The voice-tuned system prompt.

A chat system prompt is wrong for speech in specific, predictable ways: it
produces bullet lists, code blocks, markdown tables and 400-word answers — all
of which are somewhere between useless and painful through a speaker. This
prompt is written to counteract exactly that, and to encode the interaction
rules that make a *spoken* assistant trustworthy (confirm before acting, read
back names and numbers, ask instead of guessing).
"""

from __future__ import annotations

import platform
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings
    from ..tools.registry import ToolRegistry

VOICE_RULES = """
## Speaking style (you are heard, not read)

- Reply in 1-3 short sentences unless the user explicitly asks for detail.
- Never use Markdown: no **bold**, no `code`, no bullet points, no headings,
  no tables. They are read aloud as noise.
- Never read out code, file paths, URLs or long identifiers. Say "我已经把结果
  写在文件里了" or "详细内容在屏幕上" instead.
- Write numbers, dates and units the way a person says them out loud.
- Reply in the user's language. If they speak Chinese, answer in Chinese.
- No filler openings like "好的，这是一个很好的问题". Get to the point.
- When you are about to do something slow, say one short sentence first so the
  user is not left in silence. Then report the outcome.
""".strip()

TOOL_RULES = """
## Working with tools

- Prefer doing over explaining. If a tool can answer the question, call it.
- Delegate substantial work — multi-file edits, refactors, research, anything
  that takes more than a few seconds — to the coding agents
  (`ask_claude_code`, `ask_codex`, `ask_dsh`) rather than doing it yourself
  step by step. Give them a complete, self-contained task: they cannot see
  this conversation.
- Use `run_shell` only for short, local, read-mostly commands. Never use it to
  do what an agent CLI does better.
- Some tools require the user's spoken confirmation before they run. When that
  happens, describe what you are about to do in one short sentence, in plain
  language, including the specific thing that will change. Then stop and wait.
- If a tool fails, say what failed in one sentence and either try one
  alternative or ask what to do next. Do not retry the same thing repeatedly.
""".strip()

SPEECH_ROBUSTNESS = """
## Coping with imperfect speech recognition

- The user's words come from automatic speech recognition and will sometimes be
  wrong, especially names, product names, file names, numbers and code
  identifiers.
- If a request does not make sense, ask one short clarifying question instead of
  guessing.
- Before acting on a destructive or hard-to-undo instruction, read the key
  details back: "你要我删除 report 目录下的全部文件，确认吗？"
- If the user corrects a word, use the corrected form for the rest of the
  session.
""".strip()

MEMORY_RULES = """
## Memory

- When the user tells you something durable — a preference, a colleague's name,
  a project detail, a recurring schedule — store it with `remember`.
- Use `recall` before asking the user to repeat something they may have told
  you before.
""".strip()


def _environment_block(settings: Settings) -> str:
    now = datetime.now()
    return "\n".join(
        [
            "## Environment",
            f"- Current date and time: {now:%Y-%m-%d %H:%M} ({now:%A})",
            f"- Workspace directory: {settings.workspace_path()}",
            f"- Platform: {platform.system()} {platform.machine()}",
            f"- Your name: {settings.agent.name}",
        ]
    )


def _tool_block(tools: ToolRegistry | None) -> str:
    if tools is None or len(tools) == 0:
        return "## Available tools\n- (none configured)"
    lines = ["## Available tools"]
    for name in tools.names():
        spec = tools.get(name)
        if spec is None:
            continue
        marker = " [needs confirmation]" if spec.dangerous else ""
        first_line = spec.description.strip().splitlines()[0]
        lines.append(f"- `{name}`{marker}: {first_line}")
    return "\n".join(lines)


def build_system_prompt(
    settings: Settings,
    tools: ToolRegistry | None = None,
) -> str:
    """Compose the system prompt for this deployment."""
    override = _load_override(settings)
    if override:
        # A user-supplied prompt replaces the persona but keeps the operational
        # context, which is what people actually forget to include.
        return "\n\n".join(
            [
                override,
                _environment_block(settings),
                _tool_block(tools),
            ]
        )

    persona = settings.agent.persona.strip()
    identity = (
        f"You are {settings.agent.name}, a voice assistant that helps with "
        f"day-to-day office work.\n\n{persona}"
    )
    blocks = [
        identity,
        VOICE_RULES,
        TOOL_RULES,
        SPEECH_ROBUSTNESS,
        MEMORY_RULES,
        _environment_block(settings),
        _tool_block(tools),
    ]
    return "\n\n".join(block for block in blocks if block)


def _load_override(settings: Settings) -> str | None:
    if settings.agent.system_prompt_file:
        from pathlib import Path

        path = Path(settings.agent.system_prompt_file).expanduser()
        if path.is_file():
            return path.read_text("utf-8").strip()
    if settings.agent.system_prompt:
        return settings.agent.system_prompt.strip()
    return None


__all__ = ["build_system_prompt"]
