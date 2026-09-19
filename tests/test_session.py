"""The session orchestrator: turns, barge-in and the confirmation handshake.

These are the tests that matter most — the interactive behaviour here is what
users notice, and it is what the browser client is written against.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import make_bundle, tool_call, wait_for
from voiceagent.audio.pcm import AudioClip
from voiceagent.events import EventType, PipelineState
from voiceagent.providers.asr.mock import MockASR
from voiceagent.providers.base import LLMDelta
from voiceagent.session import interpret_confirmation


class SlowLLM:
    name = "slow"

    def __init__(self, pieces, delay=0.05):
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


SPEECH = AudioClip.silence(400)


class TestConfirmationInterpreter:
    @pytest.mark.parametrize("text", ["确认", "确定", "好的", "可以", "yes", "OK", "执行", "嗯"])
    def test_affirmatives(self, text):
        assert interpret_confirmation(text) is True

    @pytest.mark.parametrize("text", ["取消", "不要", "不用", "停", "no", "算了", "不对"])
    def test_negatives(self, text):
        assert interpret_confirmation(text) is False

    @pytest.mark.parametrize("text", ["帮我看看明天的天气怎么样", "", "嗯这个嘛我想想"])
    def test_non_answers(self, text):
        assert interpret_confirmation(text) is None


class TestTextTurn:
    async def test_produces_transcript_answer_audio_and_metrics(self, session_factory, collector):
        session = await session_factory(bundle=make_bundle(script=["好的，我马上处理。"]))
        await session.submit_text("帮我处理一下")
        await session.wait_for_turn()

        assert collector.text_of(EventType.TRANSCRIPT) == "帮我处理一下"
        assert "马上处理" in collector.text_of(EventType.ASSISTANT_DELTA)
        assert collector.of(EventType.ASSISTANT_DONE)
        assert collector.audio, "expected TTS audio events"
        metrics = collector.metrics()
        assert "turn_total_ms" in metrics
        assert "llm_first_token_ms" in metrics
        assert metrics["tts_first_audio_ms"] >= 0

    async def test_user_message_reaches_the_model(self, session_factory):
        bundle = make_bundle(script=["收到。"])
        session = await session_factory(bundle=bundle)
        await session.submit_text("把会议改到三点")
        await session.wait_for_turn()
        sent = bundle.llm.calls[0]
        assert any(m.role == "user" and m.content == "把会议改到三点" for m in sent)

    async def test_state_returns_to_idle(self, session_factory, collector):
        session = await session_factory(bundle=make_bundle(script=["好了。"]))
        await session.submit_text("做点事")
        await session.wait_for_turn()
        await wait_for(lambda: session.state is PipelineState.IDLE)
        assert collector.states[-1] == "idle"

    async def test_empty_text_is_ignored(self, session_factory, collector):
        session = await session_factory()
        await session.submit_text("   ")
        assert not collector.of(EventType.ASSISTANT_DONE)

    async def test_history_grows_across_turns(self, session_factory):
        session = await session_factory(bundle=make_bundle(script=["一。", "二。"]))
        await session.submit_text("第一轮")
        await session.wait_for_turn()
        await session.submit_text("第二轮")
        await session.wait_for_turn()
        roles = [m.role for m in session.agent.history]
        assert roles == ["user", "assistant", "user", "assistant"]


class TestVoiceTurn:
    async def test_audio_is_transcribed_and_answered(self, session_factory, collector):
        asr = MockASR(script=["明天几点开会"])
        session = await session_factory(bundle=make_bundle(asr=asr, script=["上午十点。"]))
        await session.begin_utterance()
        await session.push_audio(SPEECH.pcm, SPEECH.sample_rate)
        await session.end_utterance()
        await session.wait_for_turn()

        assert collector.text_of(EventType.TRANSCRIPT) == "明天几点开会"
        assert "上午十点" in collector.text_of(EventType.ASSISTANT_DELTA)
        assert asr.calls, "ASR was never called"

    async def test_empty_transcript_ends_the_turn_quietly(self, session_factory, collector):
        asr = MockASR(script=["..."])
        session = await session_factory(bundle=make_bundle(asr=asr))
        await session.begin_utterance()
        await session.push_audio(SPEECH.pcm, SPEECH.sample_rate)
        await session.end_utterance()
        await asyncio.sleep(0.15)
        assert not collector.of(EventType.ASSISTANT_DONE)


class TestBargeIn:
    async def test_speaking_over_the_assistant_interrupts_it(self, session_factory, collector):
        llm = SlowLLM(["第一句。", "第二句。", "第三句。", "第四句。", "第五句。"], delay=0.05)
        session = await session_factory(bundle=make_bundle(llm=llm))
        await session.submit_text("说点长的话")
        await wait_for(lambda: collector.of(EventType.ASSISTANT_DELTA))

        # This is exactly what the browser sends the moment its VAD hears
        # speech while audio is playing.
        await session.barge_in()

        assert llm.finished is False, "the LLM stream should have been cancelled"
        assert session.state in {PipelineState.LISTENING, PipelineState.IDLE}

    async def test_incoming_audio_alone_is_not_a_barge_in(self, session_factory, collector):
        """Regression for a bug that broke every voice turn.

        The frames of the user's *own* utterance keep arriving while its turn
        is starting. Inferring barge-in from their arrival cancelled the turn
        and silently discarded the transcript, so every spoken request got an
        answer about something else.
        """
        llm = SlowLLM(["第一句。", "第二句。", "第三句。"], delay=0.05)
        session = await session_factory(bundle=make_bundle(llm=llm))
        await session.submit_text("说点长的话")
        await wait_for(lambda: collector.of(EventType.ASSISTANT_DELTA))

        await session.begin_utterance()
        await session.push_audio(SPEECH.pcm, SPEECH.sample_rate)
        await session.push_audio(SPEECH.pcm, SPEECH.sample_rate)
        await asyncio.sleep(0.05)

        assert session._turn is not None, "arriving audio must not cancel the turn"

    async def test_interrupt_does_not_leave_a_dangling_turn(self, session_factory, collector):
        llm = SlowLLM(["一句。", "两句。", "三句。", "四句。"], delay=0.05)
        session = await session_factory(bundle=make_bundle(llm=llm))
        await session.submit_text("讲个长故事")
        await wait_for(lambda: collector.of(EventType.ASSISTANT_DELTA))
        await session.interrupt(reason="test")
        assert session._turn is None
        assert session.state in {PipelineState.IDLE, PipelineState.LISTENING}

    async def test_barge_in_can_be_disabled(self, session_factory, collector, settings):
        settings.pipeline.barge_in = False
        llm = SlowLLM(["一。", "二。", "三。"], delay=0.05)
        session = await session_factory(bundle=make_bundle(llm=llm))
        await session.submit_text("讲")
        await wait_for(lambda: collector.of(EventType.ASSISTANT_DELTA))
        await session.barge_in()
        await asyncio.sleep(0.05)
        assert session._turn is not None  # still running

    async def test_barge_in_during_a_confirmation_keeps_the_question_alive(
        self, session_factory, collector
    ):
        bundle = make_bundle(script=[
            [tool_call("run_shell", '{"command":"echo keep-alive"}')],
            "已经执行完了。",
        ])
        session = await session_factory(bundle=bundle)
        await session.submit_text("执行一条命令")
        await wait_for(lambda: collector.of(EventType.CONFIRM_REQUEST))

        await session.barge_in()

        assert session._confirm_future is not None
        assert not session._confirm_future.done(), "barge-in must not answer 'no'"

        await session.submit_text("确认")
        await session.wait_for_turn()
        assert collector.of(EventType.TOOL_RESULT)[-1].data["ok"] is True


class TestConfirmationFlow:
    async def test_typed_confirmation_allows_the_tool_to_run(self, session_factory, collector):
        bundle = make_bundle(script=[
            [tool_call("run_shell", '{"command":"echo confirmed"}')],
            "命令跑完了。",
        ])
        session = await session_factory(bundle=bundle)
        await session.submit_text("帮我执行一条命令")
        await wait_for(lambda: collector.of(EventType.CONFIRM_REQUEST))

        request = collector.of(EventType.CONFIRM_REQUEST)[0]
        assert "echo confirmed" in request.data["question"]

        await session.submit_text("确认")
        await session.wait_for_turn()
        result = collector.of(EventType.TOOL_RESULT)[-1]
        assert result.data["ok"] is True

    async def test_declining_keeps_the_tool_from_running(self, session_factory, collector):
        bundle = make_bundle(script=[
            [tool_call("run_shell", '{"command":"echo nope"}')],
            "好的，已取消。",
        ])
        session = await session_factory(bundle=bundle)
        await session.submit_text("执行命令")
        await wait_for(lambda: collector.of(EventType.CONFIRM_REQUEST))
        await session.submit_text("取消")
        await session.wait_for_turn()
        result = collector.of(EventType.TOOL_RESULT)[-1]
        assert result.data["ok"] is False
        assert result.data.get("declined") is True

    async def test_spoken_confirmation_works_with_server_side_asr(
        self, session_factory, collector
    ):
        """The regression that matters: voice confirmation must not be eaten by
        the normal 'new utterance interrupts the turn' path."""
        asr = MockASR(script=["帮我执行一条命令", "确认"])
        bundle = make_bundle(asr=asr, script=[
            [tool_call("run_shell", '{"command":"echo spoken"}')],
            "命令跑完了。",
        ])
        session = await session_factory(bundle=bundle)

        await session.begin_utterance()
        await session.push_audio(SPEECH.pcm, SPEECH.sample_rate)
        await session.end_utterance()
        await wait_for(lambda: collector.of(EventType.CONFIRM_REQUEST))

        assert session._confirm_future is not None
        await session.begin_utterance()
        await session.push_audio(SPEECH.pcm, SPEECH.sample_rate)
        await session.end_utterance()
        await session.wait_for_turn()

        result = collector.of(EventType.TOOL_RESULT)[-1]
        assert result.data["ok"] is True, "the spoken '确认' did not reach the confirmer"

    async def test_ambiguous_spoken_reply_asks_again(self, session_factory, collector):
        asr = MockASR(script=["帮我执行一条命令", "嗯这个嘛我想想再说吧"])
        bundle = make_bundle(asr=asr, script=[
            [tool_call("run_shell", '{"command":"echo maybe"}')],
            "好的。",
        ])
        session = await session_factory(bundle=bundle)
        await session.begin_utterance()
        await session.push_audio(SPEECH.pcm, SPEECH.sample_rate)
        await session.end_utterance()
        await wait_for(lambda: collector.of(EventType.CONFIRM_REQUEST))

        await session.begin_utterance()
        await session.push_audio(SPEECH.pcm, SPEECH.sample_rate)
        await session.end_utterance()
        await asyncio.sleep(0.2)

        # Still waiting: the ambiguous answer must not be treated as "yes".
        assert session._confirm_future is not None
        assert not session._confirm_future.done()
        await session.interrupt(reason="cleanup")

    async def test_a_safe_tool_never_prompts(self, session_factory, collector):
        bundle = make_bundle(script=[
            [tool_call("list_files", "{}")],
            "这里是文件列表。",
        ])
        session = await session_factory(bundle=bundle)
        await session.submit_text("看看有哪些文件")
        await session.wait_for_turn()
        assert not collector.of(EventType.CONFIRM_REQUEST)
        assert collector.of(EventType.TOOL_RESULT)[-1].data["ok"] is True

    async def test_auto_confirm_skips_the_prompt(self, session_factory, collector):
        bundle = make_bundle(script=[
            [tool_call("run_shell", '{"command":"echo auto"}')],
            "跑完了。",
        ])
        session = await session_factory(bundle=bundle, auto_confirm=True)
        await session.submit_text("执行")
        await session.wait_for_turn()
        assert not collector.of(EventType.CONFIRM_REQUEST)
        assert collector.of(EventType.TOOL_RESULT)[-1].data["ok"] is True


class TestToolNarration:
    async def test_a_long_tool_run_is_not_silent(self, session_factory, collector, settings):
        # Point the agent CLI at something instant and offline; the real
        # subprocess path is covered by scripts/smoke.py.
        settings.tools.agents["dsh"].command = "echo"
        settings.tools.agents["dsh"].args = []
        bundle = make_bundle(script=[
            [tool_call("ask_dsh", '{"task":"分析一下日志"}')],
            "分析完成了。",
        ])
        session = await session_factory(bundle=bundle, auto_confirm=True)
        await session.submit_text("让 dsh 分析日志")
        await session.wait_for_turn()
        spoken = collector.text_of(EventType.ASSISTANT_DELTA)
        assert "DSH" in spoken
        assert collector.audio

    async def test_tool_summary_is_queued_for_speech(self, session_factory, collector):
        bundle = make_bundle(script=[
            [tool_call("remember", '{"note":"周四复盘"}')],
            "记住了。",
        ])
        session = await session_factory(bundle=bundle)
        await session.submit_text("记住周四复盘")
        await session.wait_for_turn()
        assert collector.audio


class TestLifecycle:
    async def test_ready_event_describes_the_deployment(self, session_factory, collector):
        await session_factory()
        ready = collector.of(EventType.READY)[0]
        assert ready.data["providers"] == {"asr": "mock", "tts": "mock", "llm": "mock"}
        assert ready.data["tools"]
        assert ready.data["agent"]

    async def test_close_is_idempotent(self, session_factory):
        session = await session_factory()
        await session.aclose()
        await session.aclose()

    async def test_errors_in_a_turn_are_reported_not_raised(self, session_factory, collector):
        class ExplodingLLM:
            name = "boom"

            def chat_stream(self, messages, **kwargs):
                return self._stream()

            async def _stream(self):
                raise RuntimeError("upstream exploded")
                yield  # pragma: no cover

            async def aclose(self):
                return None

        session = await session_factory(bundle=make_bundle(llm=ExplodingLLM()))
        await session.submit_text("会出错的一轮")
        await session.wait_for_turn()
        assert collector.of(EventType.ERROR)
