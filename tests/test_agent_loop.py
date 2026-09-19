"""The reasoning loop: streaming, tool calls, confirmation and interruption."""

from __future__ import annotations

import asyncio

from tests.conftest import make_bundle, tool_call
from voiceagent.agent.loop import AgentLoop
from voiceagent.events import EventType
from voiceagent.providers.base import LLMDelta, Message
from voiceagent.tools.base import ToolContext, ToolResult, ToolSpec, object_schema, string
from voiceagent.tools.registry import ToolRegistry


class SlowLLM:
    """Emits one delta per ``delay`` so interruption can be tested deterministically."""

    name = "slow"

    def __init__(self, pieces, delay=0.03):
        self.pieces = list(pieces)
        self.delay = delay
        self.finished = False

    def chat_stream(self, messages, *, tools=None, temperature=None, max_tokens=None):
        return self._stream()

    async def _stream(self):
        for piece in self.pieces:
            await asyncio.sleep(self.delay)
            yield LLMDelta(content=piece)
        self.finished = True
        yield LLMDelta(finish_reason="stop")

    async def aclose(self):
        return None


def make_ctx(settings, tmp_path, *, confirm=None, emit=None, cancel=None) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        settings=settings,
        emit=emit,
        confirm=confirm,
        cancel=cancel,
    )


async def collect(loop: AgentLoop, ctx: ToolContext) -> list:
    return [event async for event in loop.run(ctx)]


class TestPlainAnswers:
    async def test_streams_deltas_then_done(self, settings, tmp_path):
        loop = AgentLoop(make_bundle(script=["你好，我在。"]).llm, ToolRegistry([], settings), settings)
        events = await collect(loop, make_ctx(settings, tmp_path))
        assert "".join(e.data["text"] for e in events if e.type is EventType.ASSISTANT_DELTA) == "你好，我在。"
        assert [e for e in events if e.type is EventType.ASSISTANT_DONE][0].data["text"] == "你好，我在。"

    async def test_records_the_turn_in_history(self, settings, tmp_path):
        loop = AgentLoop(make_bundle(script=["好的。"]).llm, ToolRegistry([], settings), settings)
        loop.append(Message(role="user", content="早"))
        await collect(loop, make_ctx(settings, tmp_path))
        assert [m.role for m in loop.history] == ["user", "assistant"]

    async def test_system_prompt_is_first_in_what_the_model_sees(self, settings, tmp_path):
        llm = make_bundle(script=["好。"]).llm
        loop = AgentLoop(llm, ToolRegistry([], settings), settings)
        loop.append(Message(role="user", content="hi"))
        await collect(loop, make_ctx(settings, tmp_path))
        sent = llm.calls[0]
        assert sent[0].role == "system"
        assert "spoken" in sent[0].content.lower() or "heard" in sent[0].content.lower()


class TestToolCalling:
    async def test_runs_a_tool_then_answers(self, settings, tmp_path, registry):
        bundle = make_bundle(script=[
            [tool_call("remember", '{"note":"周五开周会"}')],
            "好的，我记住了。",
        ])
        loop = AgentLoop(bundle.llm, registry, settings)
        events = await collect(loop, make_ctx(settings, tmp_path))
        kinds = [e.type for e in events]
        assert EventType.TOOL_CALL in kinds
        assert EventType.TOOL_RESULT in kinds
        result = next(e for e in events if e.type is EventType.TOOL_RESULT)
        assert result.data["ok"] is True
        assert [m.role for m in loop.history] == ["assistant", "tool", "assistant"]

    async def test_tool_failure_is_fed_back_to_the_model(self, settings, tmp_path, registry):
        bundle = make_bundle(script=[
            [tool_call("read_file", '{"path":"missing.txt"}')],
            "那个文件不存在。",
        ])
        loop = AgentLoop(bundle.llm, registry, settings)
        events = await collect(loop, make_ctx(settings, tmp_path))
        result = next(e for e in events if e.type is EventType.TOOL_RESULT)
        assert result.data["ok"] is False
        assert loop.history[1].role == "tool"
        assert loop.history[1].content.startswith("[ERROR]")

    async def test_unknown_tool_does_not_crash_the_turn(self, settings, tmp_path, registry):
        bundle = make_bundle(script=[[tool_call("nope")], "换个办法。"])
        loop = AgentLoop(bundle.llm, registry, settings)
        events = await collect(loop, make_ctx(settings, tmp_path))
        assert next(e for e in events if e.type is EventType.TOOL_RESULT).data["ok"] is False
        assert next(e for e in events if e.type is EventType.ASSISTANT_DONE)

    async def test_tool_iteration_cap_is_enforced(self, settings, tmp_path, registry):
        settings.agent.max_tool_iterations = 2
        script = [[tool_call("list_files", "{}", f"c{i}")] for i in range(10)]
        loop = AgentLoop(make_bundle(script=script).llm, registry, settings)
        events = await collect(loop, make_ctx(settings, tmp_path))
        assert len([e for e in events if e.type is EventType.TOOL_CALL]) <= 3

    async def test_independent_safe_tools_run_concurrently(self, settings, tmp_path):
        started = asyncio.Event()

        async def slow(_args, _ctx):
            started.set()
            await asyncio.sleep(0.25)
            return ToolResult.success("done")

        specs = [
            ToolSpec(name=f"slow_{i}", description="d", parameters=object_schema(), handler=slow)
            for i in range(3)
        ]
        registry = ToolRegistry(specs, settings)
        loop = AgentLoop(
            make_bundle(script=[
                [tool_call(f"slow_{i}", "{}", f"c{i}") for i in range(3)],
                "都完成了。",
            ]).llm,
            registry,
            settings,
        )
        began = asyncio.get_running_loop().time()
        await collect(loop, make_ctx(settings, tmp_path))
        elapsed = asyncio.get_running_loop().time() - began
        assert elapsed < 0.6, f"expected concurrency, took {elapsed:.2f}s"

    async def test_dangerous_tools_run_one_at_a_time(self, settings, tmp_path):
        order: list[str] = []

        async def record(args, _ctx):
            order.append(args["tag"])
            await asyncio.sleep(0.05)
            return ToolResult.success("ok")

        specs = [
            ToolSpec(name="risky", description="d", parameters=object_schema(
                {"tag": string("t")}), handler=record, dangerous=True)
        ]
        registry = ToolRegistry(specs, settings)

        async def yes(_question: str) -> bool:
            return True

        loop = AgentLoop(
            make_bundle(script=[
                [tool_call("risky", '{"tag":"a"}', "c1"), tool_call("risky", '{"tag":"b"}', "c2")],
                "完成。",
            ]).llm,
            registry,
            settings,
        )
        await collect(loop, make_ctx(settings, tmp_path, confirm=yes))
        assert order == ["a", "b"]


class TestConfirmation:
    async def test_confirmed_tool_executes(self, settings, tmp_path, registry):
        questions: list[str] = []

        async def yes(question: str) -> bool:
            questions.append(question)
            return True

        loop = AgentLoop(
            make_bundle(script=[
                [tool_call("run_shell", '{"command":"echo hi"}')],
                "执行完了。",
            ]).llm,
            registry,
            settings,
        )
        events = await collect(loop, make_ctx(settings, tmp_path, confirm=yes))
        assert questions and "echo hi" in questions[0]
        assert next(e for e in events if e.type is EventType.TOOL_RESULT).data["ok"] is True

    async def test_declined_tool_does_not_run(self, settings, tmp_path, registry):
        async def no(_question: str) -> bool:
            return False

        loop = AgentLoop(
            make_bundle(script=[
                [tool_call("run_shell", '{"command":"echo should-not-run"}')],
                "好的，我取消了。",
            ]).llm,
            registry,
            settings,
        )
        events = await collect(loop, make_ctx(settings, tmp_path, confirm=no))
        result = next(e for e in events if e.type is EventType.TOOL_RESULT)
        assert result.data["ok"] is False
        assert result.data.get("declined") is True
        assert not (tmp_path / "should-not-run").exists()

    async def test_dangerous_tool_without_a_confirmer_is_refused(self, settings, tmp_path, registry):
        loop = AgentLoop(
            make_bundle(script=[
                [tool_call("run_shell", '{"command":"echo hi"}')],
                "好吧。",
            ]).llm,
            registry,
            settings,
        )
        events = await collect(loop, make_ctx(settings, tmp_path))
        result = next(e for e in events if e.type is EventType.TOOL_RESULT)
        assert result.data["ok"] is False

    async def test_safe_tools_never_ask(self, settings, tmp_path, registry):
        async def should_not_be_called(_question: str) -> bool:
            raise AssertionError("a read-only tool asked for confirmation")

        loop = AgentLoop(
            make_bundle(script=[[tool_call("list_files", "{}")], "看完了。"]).llm,
            registry,
            settings,
        )
        await collect(loop, make_ctx(settings, tmp_path, confirm=should_not_be_called))


class TestInterruption:
    async def test_interrupt_stops_the_stream(self, settings, tmp_path):
        llm = SlowLLM(["第一句话。", "第二句话。", "第三句话。", "第四句话。"], delay=0.05)
        loop = AgentLoop(llm, ToolRegistry([], settings), settings)
        interrupt = asyncio.Event()

        seen: list[str] = []

        async def consume():
            async for event in loop.run(make_ctx(settings, tmp_path), interrupt=interrupt):
                if event.type is EventType.ASSISTANT_DELTA:
                    seen.append(event.data["text"])

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.09)
        interrupt.set()
        await asyncio.wait_for(task, timeout=2)

        assert len(seen) < len(llm.pieces)
        assert not llm.finished

    async def test_partial_turn_is_recorded_after_interruption(self, settings, tmp_path):
        llm = SlowLLM(["一半", "被", "打断"], delay=0.04)
        loop = AgentLoop(llm, ToolRegistry([], settings), settings)
        interrupt = asyncio.Event()

        async def consume():
            async for _ in loop.run(make_ctx(settings, tmp_path), interrupt=interrupt):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.06)
        interrupt.set()
        await asyncio.wait_for(task, timeout=2)

        assert len(loop.history) == 1
        assert loop.history[0].role == "assistant"
        assert loop.history[0].content  # the partial text, not a torn-down turn
