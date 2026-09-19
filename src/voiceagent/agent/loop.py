"""The reasoning loop: streaming LLM turns interleaved with tool calls.

Yields :class:`~voiceagent.events.Event` objects rather than returning a value,
so the session can start speaking the first sentence while the model is still
generating the third, and can surface tool progress without any extra plumbing.

Two behaviours worth calling out:

* **Interruption is checked between every delta.** When the user barges in, the
  loop stops immediately and leaves the conversation in a consistent state
  (the partial assistant turn is recorded as such), rather than tearing down
  mid-message and confusing the next turn.
* **Independent tool calls run concurrently** when none of them needs
  confirmation — "have Claude and Codex both look at this" should not take
  twice as long. Anything requiring confirmation runs strictly one at a time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import replace

from ..config import Settings
from ..events import Event, EventType, PipelineState
from ..providers.base import LLMProvider, Message, ToolCall
from ..tools.base import ToolContext, ToolResult
from ..tools.registry import ToolRegistry
from .prompt import build_system_prompt

#: Stop a runaway model from calling tools forever.
MAX_TURNS_MESSAGE = "已经达到工具调用次数上限，我先停下来。"


class AgentLoop:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        settings: Settings,
        *,
        history: list[Message] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.settings = settings
        self.history: list[Message] = history if history is not None else []
        self._schema_cache: list[dict] | None = None

    # -- prompt / history -------------------------------------------------

    def system_message(self) -> Message:
        return Message(role="system", content=build_system_prompt(self.settings, self.tools))

    def trimmed(self) -> list[Message]:
        """System prompt + the last N turns, in a shape every provider accepts."""
        turns = self.settings.pipeline.history_turns
        tail = self.history[-(turns * 2) :] if turns > 0 else list(self.history)
        return [self.system_message(), *_sanitize(tail)]

    def append(self, message: Message) -> None:
        self.history.append(message)

    def reset(self) -> None:
        self.history.clear()

    def truncate_last_assistant(self, keep_chars: int | None = None) -> None:
        """Record that the assistant was cut off mid-sentence by barge-in."""
        for message in reversed(self.history):
            if message.role == "assistant" and message.content:
                if keep_chars is not None:
                    message.content = message.content[:keep_chars]
                break

    # -- tool schema ------------------------------------------------------

    def tool_schema(self) -> list[dict] | None:
        if self._schema_cache is None:
            schema = self.tools.openai_schema()
            self._schema_cache = schema or None
        return self._schema_cache

    # -- main loop --------------------------------------------------------

    async def run(
        self,
        ctx: ToolContext,
        *,
        interrupt: asyncio.Event | None = None,
    ) -> AsyncIterator[Event]:
        """Run until the model produces a final answer or we run out of turns.

        Events are funnelled through a single queue so that tool progress —
        which originates deep inside a tool handler — arrives on the *same*
        ordered stream as assistant deltas. A caller therefore cannot miss an
        event by forgetting to wire up ``ctx.emit``.
        """
        interrupt = interrupt or asyncio.Event()
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        async def emit(event: Event) -> None:
            await queue.put(event)

        ctx = replace(ctx, emit=emit)

        async def body() -> None:
            try:
                async for event in self._run(ctx, interrupt):
                    await emit(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Hand the failure to the consumer rather than swallowing it in
                # a task nobody awaits.
                await queue.put(exc)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(body())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not producer.done():
                # The consumer stopped early (barge-in). Do not leak the turn.
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer

    async def _run(
        self,
        ctx: ToolContext,
        interrupt: asyncio.Event,
    ) -> AsyncIterator[Event]:
        max_iterations = max(1, self.settings.agent.max_tool_iterations)

        for iteration in range(max_iterations + 1):
            if interrupt.is_set():
                return

            yield Event.state(PipelineState.THINKING, iteration=iteration)

            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            finished = False

            async for delta in self.llm.chat_stream(self.trimmed(), tools=self.tool_schema()):
                if interrupt.is_set():
                    self._record_interrupted(text_parts, tool_calls)
                    return
                if delta.content:
                    text_parts.append(delta.content)
                    yield Event.assistant_delta(delta.content)
                if delta.tool_calls:
                    tool_calls = delta.tool_calls
                if delta.finish_reason:
                    finished = True

            text = "".join(text_parts)

            if not tool_calls:
                if text.strip():
                    self.history.append(Message(role="assistant", content=text))
                yield Event.assistant_done(text)
                return

            # The assistant asked for tools; record the request verbatim.
            self.history.append(
                Message(role="assistant", content=text, tool_calls=tool_calls)
            )

            if iteration >= max_iterations:
                yield Event.assistant_done(text or MAX_TURNS_MESSAGE)
                return

            yield Event.state(PipelineState.ACTING, tools=[c.name for c in tool_calls])

            results = await self._execute_tools(tool_calls, ctx, interrupt)
            if interrupt.is_set():
                return

            for call, result in zip(tool_calls, results, strict=True):
                self.history.append(
                    Message(
                        role="tool",
                        content=_tool_content(result),
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

            if not finished and not text:
                # Some providers end a tool turn without a finish_reason; loop.
                continue

    # -- tool execution ---------------------------------------------------

    async def _execute_tools(
        self,
        calls: list[ToolCall],
        ctx: ToolContext,
        interrupt: asyncio.Event,
    ) -> list[ToolResult]:
        dangerous = self.tools.dangerous()
        needs_confirmation = [c for c in calls if c.name in dangerous]

        for call in calls:
            yield_ = Event(
                type=EventType.TOOL_CALL,
                data={"id": call.id, "name": call.name, "arguments": call.parsed_args()},
            )
            await _emit(ctx, yield_)

        if len(calls) > 1 and not needs_confirmation:
            # Independent, safe tools: run concurrently, preserve order.
            return list(
                await asyncio.gather(
                    *(self._one(call, ctx, dangerous) for call in calls)
                )
            )

        results: list[ToolResult] = []
        for call in calls:
            if interrupt.is_set():
                results.append(ToolResult.failure("interrupted", summary="已中断。"))
                continue
            results.append(await self._one(call, ctx, dangerous))
        return results

    async def _one(
        self,
        call: ToolCall,
        ctx: ToolContext,
        dangerous: set[str],
    ) -> ToolResult:
        if call.name in dangerous:
            if ctx.confirm is None:
                result = ToolResult.failure(
                    f"{call.name} requires confirmation but no confirmer is wired up",
                    summary="这个操作需要你确认，但我现在没法问你。",
                )
                await _emit(
                    ctx,
                    Event(
                        type=EventType.TOOL_RESULT,
                        data={"id": call.id, "name": call.name, "ok": False,
                              "summary": result.summary},
                    ),
                )
                return result
            question = _confirmation_question(call)
            approved = await ctx.confirm(question)
            if not approved:
                result = ToolResult.failure(
                    "the user declined to run this tool. Do not retry it; "
                    "acknowledge briefly and ask what they would like instead.",
                    summary="好的，我取消了。",
                    declined=True,
                )
                await _emit(
                    ctx,
                    Event(
                        type=EventType.TOOL_RESULT,
                        data={"id": call.id, "name": call.name, "ok": False,
                              "summary": result.summary, "declined": True},
                    ),
                )
                return result

        result = await self.tools.call(call, ctx)
        await _emit(
            ctx,
            Event(
                type=EventType.TOOL_RESULT,
                data={
                    "id": call.id,
                    "name": call.name,
                    "ok": result.ok,
                    "summary": result.summary or _short(result.content),
                    "truncated": result.truncated,
                    **result.meta,
                },
            ),
        )
        return result

    # -- helpers ----------------------------------------------------------

    def _record_interrupted(self, text_parts: list[str], tool_calls: list[ToolCall]) -> None:
        text = "".join(text_parts)
        if text.strip() or tool_calls:
            self.history.append(
                Message(role="assistant", content=text, tool_calls=tool_calls or None)
            )


def _sanitize(messages: list[Message]) -> list[Message]:
    """Remove messages that would make the provider reject the request.

    Interrupting a turn can leave an assistant message that requested tools
    whose results never arrived, and orphan ``tool`` messages. OpenAI-compatible
    APIs answer those with a 400, so they are dropped — but the assistant's text
    is kept, because the model should know what it already said.
    """
    out: list[Message] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "tool":
            index += 1  # orphan result
            continue
        if message.role == "assistant" and message.tool_calls:
            expected = [call.id for call in message.tool_calls]
            window = messages[index + 1 : index + 1 + len(expected)]
            answered = [m.tool_call_id for m in window if m.role == "tool"]
            if answered == expected:
                out.append(message)
                out.extend(window)
                index += 1 + len(expected)
                continue
            if message.content.strip():
                out.append(Message(role="assistant", content=message.content))
            index += 1
            continue
        out.append(message)
        index += 1
    return out


def _confirmation_question(call: ToolCall) -> str:
    """Turn a tool call into a short, specific, spoken question."""
    args = call.parsed_args()
    if call.name == "run_shell":
        return f"我要执行这条命令：{args.get('command', '')}。确认执行吗？"
    if call.name == "write_file":
        target = args.get("path", "")
        lines = len(str(args.get("content", "")).splitlines())
        verb = "追加到" if args.get("append") else "写入"
        return f"我要把 {lines} 行内容{verb}文件 {target}。确认吗？"
    if call.name == "ask_claude_code":
        return f"我要让 Claude Code 处理这个任务：{args.get('task', '')[:120]}。确认吗？"
    if call.name == "ask_codex":
        return f"我要让 Codex 处理这个任务：{args.get('task', '')[:120]}。确认吗？"
    if call.name == "ask_dsh":
        return f"我要让 DSH 处理这个任务：{args.get('task', '')[:120]}。确认吗？"
    return f"我要调用工具 {call.name}，参数是 {json.dumps(args, ensure_ascii=False)[:160]}。确认吗？"


def _tool_content(result: ToolResult) -> str:
    prefix = "OK" if result.ok else "ERROR"
    return f"[{prefix}] {result.content}"


def _short(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


async def _emit(ctx: ToolContext, event: Event) -> None:
    if ctx.emit is not None:
        await ctx.emit(event)


__all__ = ["AgentLoop"]
