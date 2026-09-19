"""The session orchestrator — where speech, reasoning and action meet.

This is the piece that separates a demo from something you would actually use,
because it owns the two hard problems:

**Barge-in.** Interrupting is not "stop the speaker". When the user talks over
the assistant we must (1) stop audio immediately, (2) stop the LLM stream,
(3) record what was actually said so the next turn has correct history, and
(4) decide what happens to an in-flight tool. All four are handled here.

**Confirmation.** Anything destructive is announced in one short sentence and
then *waits*. The answer can arrive by voice ("确认"), by button, or from the
terminal — the session does not care which.

Everything leaves through a single ``on_event`` callback, so the WebSocket
transport and the terminal transport are interchangeable.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from .agent.loop import AgentLoop
from .audio.pcm import TARGET_SAMPLE_RATE, AudioBuffer, AudioChunk, AudioClip
from .audio.player import Speaker
from .config import Settings
from .events import Event, EventType, PipelineState
from .providers.base import Message
from .providers.registry import ProviderBundle
from .text import SentenceSplitter, truncate_for_speech
from .tools.base import ToolContext
from .tools.registry import ToolRegistry

EventSink = Callable[[Event], Awaitable[None]]

#: How long we wait for a yes/no after asking. Silence means "no" — the safe
#: default for anything that needed confirmation in the first place.
CONFIRM_TIMEOUT_S = 45.0

#: Guard against ASR emitting nothing but punctuation or noise.
_MEANINGFUL = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")

#: Single-syllable replies only match exactly — otherwise "嗯这个嘛…" would be
#: read as "yes" because it starts with 嗯.
YES_EXACT = {"嗯", "好", "是", "对", "行", "y", "ok", "go", "yes"}
YES_PREFIX = (
    "确认", "确定", "好的", "好好", "可以", "执行", "继续", "是的", "对的", "嗯嗯",
    "yeah", "yep", "okay", "sure", "confirm", "proceed", "go ahead", "do it",
)
NO_EXACT = {"不", "否", "别", "停", "no", "nope"}
NO_PREFIX = (
    "取消", "不要", "不用", "不行", "不好", "不对", "停止", "算了", "拒绝",
    "cancel", "stop", "abort",
)


def interpret_confirmation(text: str) -> bool | None:
    """Map a spoken reply to yes/no. ``None`` means "that was not an answer"."""
    cleaned = text.strip().lower().strip("。！？!?.,，、 ")
    if not cleaned or len(cleaned) > 12:
        # Long replies are new instructions, not answers to a yes/no question.
        return None
    # Negative first: "不要" must not be read as the affirmative "好".
    if cleaned in NO_EXACT or cleaned.startswith(NO_PREFIX):
        return False
    if cleaned in YES_EXACT or cleaned.startswith(YES_PREFIX):
        return True
    return None


class VoiceSession:
    """One conversation, one microphone, one speaker."""

    def __init__(
        self,
        settings: Settings,
        bundle: ProviderBundle,
        tools: ToolRegistry,
        *,
        on_event: EventSink,
        workspace: Path | None = None,
        session_id: str = "default",
        speaker: Speaker | None = None,
        auto_confirm: bool = False,
        emit_audio_payloads: bool = True,
    ) -> None:
        self.settings = settings
        self.bundle = bundle
        self.tools = tools
        self.on_event = on_event
        self.workspace = workspace or settings.workspace_path()
        self.session_id = session_id
        self.speaker = speaker
        self.auto_confirm = auto_confirm
        #: The terminal sink plays audio itself, so it does not need base64
        #: copies of every chunk travelling through the event stream.
        self.emit_audio_payloads = emit_audio_payloads

        self.agent = AgentLoop(bundle.llm, tools, settings)
        self.state = PipelineState.IDLE

        # utterance capture
        self._buffer = AudioBuffer(sample_rate=TARGET_SAMPLE_RATE)
        self._buffer_rate = TARGET_SAMPLE_RATE
        self._capturing = False

        # turn management
        self._turn: asyncio.Task[None] | None = None
        self._interrupt = asyncio.Event()
        self._confirm_future: asyncio.Future[bool] | None = None
        self._speak_queue: asyncio.Queue[str | None] | None = None
        self._closed = False

        # per-turn instrumentation
        self._marks: dict[str, float] = {}
        self._spoken_chars = 0

    # ==================================================================
    # lifecycle
    # ==================================================================

    async def start(self) -> None:
        await self._emit(
            Event(
                type=EventType.READY,
                data={
                    "session_id": self.session_id,
                    "providers": self.bundle.names,
                    "tools": self.tools.names(),
                    "warnings": self.tools.warnings,
                    "agent": self.settings.agent.name,
                    "config": self.settings.describe(),
                    # The browser runs its own VAD, so it needs these to agree
                    # with the server. One place to tune turn-taking.
                    "vad": {
                        "silence_ms": self.settings.pipeline.vad_silence_ms,
                        "threshold": self.settings.pipeline.vad_threshold,
                        "min_utterance_ms": self.settings.pipeline.asr_min_utterance_ms,
                    },
                },
            )
        )
        await self._set_state(PipelineState.IDLE)

    async def aclose(self) -> None:
        self._closed = True
        await self.interrupt(reason="session_closed")
        if self._turn is not None:
            self._turn.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn
        self._turn = None
        self.resolve_confirmation(False)

    async def wait_for_turn(self) -> None:
        """Block until the in-flight turn finishes (used by the CLI REPL)."""
        task = self._turn
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(asyncio.shield(task))

    async def speak(self, text: str) -> None:
        """Say a short line without going through the model.

        For transports that need to announce something of their own — a startup
        greeting, a notification, a warning — while keeping correct ordering
        with whatever the assistant is currently saying.
        """
        await self._say(text)

    # ==================================================================
    # audio in
    # ==================================================================

    async def begin_utterance(self) -> None:
        """The user started talking (push-to-talk down, or VAD onset)."""
        self._capturing = True
        self._buffer.clear()
        await self._set_state(PipelineState.LISTENING)

    async def push_audio(self, pcm: bytes, sample_rate: int | None = None) -> None:
        if not self._capturing:
            self._capturing = True
        if sample_rate and sample_rate != self._buffer_rate:
            pcm = AudioClip(pcm, sample_rate).resample(self._buffer_rate).pcm
        self._buffer.append(pcm)

    async def barge_in(self) -> None:
        """The user started talking over the assistant.

        Barge-in is decided by the *transport*, never inferred from the mere
        arrival of audio: the frames of the user's own in-flight utterance
        arrive while its turn is starting, and treating those as an
        interruption makes the assistant cut itself off every single turn.
        The browser sends an explicit ``interrupt`` when its VAD hears speech
        while audio is playing; the terminal runs half-duplex and does not
        listen while speaking.
        """
        if not self.settings.pipeline.barge_in:
            return
        if self._awaiting_confirmation():
            # The user is answering the question we just asked. Stop the
            # question audio so they are not talking over it, but do NOT cancel
            # the turn that is waiting for their answer.
            if self.speaker is not None:
                await self.speaker.stop()
            return
        if self._is_busy():
            await self.interrupt(reason="barge_in")

    async def end_utterance(self) -> None:
        """The user stopped talking; transcribe and respond."""
        self._capturing = False
        clip = self._buffer.take()
        if clip.duration_s * 1000 < self.settings.pipeline.asr_min_utterance_ms:
            await self._emit(Event.audio_end(reason="too_short"))
            await self._set_state(PipelineState.IDLE)
            return

        if self._awaiting_confirmation():
            # This utterance is the answer to a pending yes/no question, not a
            # new turn — interrupting here would make voice confirmation
            # impossible.
            await self._transcribe_confirmation(clip)
            return

        if self._turn is not None and not self._turn.done():
            await self.interrupt(reason="new_turn")
        self._start_turn(self._transcribe_and_respond(clip))

    async def submit_text(self, text: str) -> None:
        """Typed input, or a transcript produced in the browser."""
        text = text.strip()
        if not text:
            return
        await self._dispatch(text, source="text")

    # ==================================================================
    # interruption
    # ==================================================================

    def _is_busy(self) -> bool:
        return (self._turn is not None and not self._turn.done()) or self._confirm_future is not None

    def _awaiting_confirmation(self) -> bool:
        return self._confirm_future is not None and not self._confirm_future.done()

    async def interrupt(self, *, reason: str = "user") -> None:
        """Stop everything in flight and leave the conversation consistent."""
        self._interrupt.set()
        if self.speaker is not None:
            await self.speaker.stop()
        if self._confirm_future is not None and not self._confirm_future.done():
            self._confirm_future.set_result(False)

        task, self._turn = self._turn, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        if reason == "barge_in":
            await self._emit(Event.log("barge-in: playback and generation stopped", "debug"))
        target = PipelineState.LISTENING if self._capturing else PipelineState.IDLE
        await self._set_state(target)

    # ==================================================================
    # confirmation
    # ==================================================================

    def resolve_confirmation(self, approved: bool) -> None:
        future = self._confirm_future
        if future is not None and not future.done():
            future.set_result(approved)

    async def _confirm(self, question: str) -> bool:
        if self.auto_confirm:
            await self._emit(Event.log(f"auto-confirmed: {question}", "debug"))
            return True

        self._confirm_future = asyncio.get_running_loop().create_future()
        await self._set_state(PipelineState.AWAITING_CONFIRMATION)
        await self._emit(Event(type=EventType.CONFIRM_REQUEST, data={"question": question}))
        await self._say(question)
        try:
            approved = await asyncio.wait_for(self._confirm_future, CONFIRM_TIMEOUT_S)
        except TimeoutError:
            approved = False
            await self._say("我没有听到确认，先取消了。")
        finally:
            self._confirm_future = None
        await self._emit(
            Event.log(f"confirmation {'granted' if approved else 'declined'}", "debug")
        )
        return approved

    # ==================================================================
    # turns
    # ==================================================================

    def _start_turn(self, coro: Awaitable[None]) -> None:
        self._interrupt = asyncio.Event()
        self._marks = {"turn_start": time.monotonic()}
        self._spoken_chars = 0
        self._turn = asyncio.create_task(self._guard(coro))

    async def _guard(self, coro: Awaitable[None]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            await self._emit(Event.error(f"{type(exc).__name__}: {exc}", where="turn"))
            await self._set_state(PipelineState.IDLE)

    async def _transcribe(self, clip: AudioClip) -> str:
        """Run ASR and report the metric. Returns "" when nothing usable came back."""
        await self._set_state(PipelineState.TRANSCRIBING)
        started = time.monotonic()
        try:
            transcript = await self.bundle.asr.transcribe(
                clip,
                language=self.settings.asr.language,
                hotwords=self.settings.asr.hotwords,
            )
        except Exception as exc:
            await self._emit(Event.error(str(exc), where="asr"))
            return ""

        await self._emit(
            Event.metric(
                "asr_ms",
                (time.monotonic() - started) * 1000,
                provider=self.bundle.names.get("asr", ""),
            )
        )
        text = transcript.text.strip()
        if not _MEANINGFUL.search(text):
            await self._emit(Event.log("empty transcript; ignoring the turn", "debug"))
            return ""
        await self._emit(
            Event.transcript(
                text,
                final=True,
                language=transcript.language,
                duration_s=round(clip.duration_s, 2),
            )
        )
        return text

    async def _transcribe_confirmation(self, clip: AudioClip) -> None:
        """Interpret a spoken yes/no without disturbing the waiting turn."""
        text = await self._transcribe(clip)
        if not text:
            return
        answer = interpret_confirmation(text)
        if answer is None:
            # Ambiguous answers are the most common failure here, so give the
            # user a second chance rather than guessing or timing out silently.
            await self._emit(Event.log(f"ambiguous confirmation: {text!r}", "debug"))
            await self._say("我没有听清，请说确认，或者取消。")
            return
        await self._emit(Event.log(f"spoken confirmation -> {answer}", "debug"))
        self.resolve_confirmation(answer)

    async def _transcribe_and_respond(self, clip: AudioClip) -> None:
        text = await self._transcribe(clip)
        if not text:
            # Keep the protocol symmetric: the client is always told the turn
            # is over, even when it produced no speech.
            await self._emit(Event.audio_end(reason="empty_transcript"))
            await self._set_state(
                PipelineState.LISTENING if self._capturing else PipelineState.IDLE
            )
            return
        await self._respond(text, source="voice")

    async def _dispatch(self, text: str, *, source: str) -> None:
        """Route one piece of user text: either an answer, or a new turn."""
        if self._awaiting_confirmation():
            answer = interpret_confirmation(text)
            if answer is not None:
                self.resolve_confirmation(answer)
                return
            await self._say("我没有听清，请说确认，或者取消。")
            return
        await self._emit(Event.transcript(text, final=True, source=source))
        if self._turn is not None and not self._turn.done():
            await self.interrupt(reason="new_turn")
        self._start_turn(self._respond(text, source=source))

    async def _respond(self, text: str, *, source: str) -> None:
        """Run the agent loop and stream the answer into TTS."""
        self.agent.append(Message(role="user", content=text))

        ctx = ToolContext(
            workspace=self.workspace,
            settings=self.settings,
            session_id=self.session_id,
            emit=self._emit,
            cancel=self._interrupt,
            confirm=self._confirm,
        )

        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        splitter = SentenceSplitter(
            flush_chars=self.settings.pipeline.sentence_flush_chars,
            max_chars=self.settings.pipeline.max_sentence_chars,
        )
        self._speak_queue = sentences
        seq_holder = [0]
        speaker_task = asyncio.create_task(self._tts_worker(sentences, seq_holder))

        collected: list[str] = []
        first_token_at: float | None = None
        #: The loop records the assistant turn itself once it completes; this
        #: flag tells the cleanup path whether it still needs to.
        recorded = False

        async def enqueue(sentence: str) -> None:
            """Feed TTS, respecting the per-turn spoken-length budget."""
            if self._spoken_chars >= self.settings.pipeline.max_spoken_chars:
                return
            self._spoken_chars += len(sentence)
            await sentences.put(sentence)

        try:
            async for event in self.agent.run(ctx, interrupt=self._interrupt):
                if self._interrupt.is_set():
                    break

                if event.type is EventType.ASSISTANT_DELTA:
                    piece = str(event.data.get("text", ""))
                    if first_token_at is None and piece:
                        first_token_at = time.monotonic()
                        await self._emit(
                            Event.metric(
                                "llm_first_token_ms",
                                (first_token_at - self._marks.get("turn_start", first_token_at))
                                * 1000,
                                provider=self.bundle.names.get("llm", ""),
                            )
                        )
                    collected.append(piece)
                    for sentence in splitter.push(piece):
                        await enqueue(sentence)
                    await self._emit(event)

                elif event.type is EventType.TOOL_CALL:
                    preamble = _tool_preamble(event)
                    await enqueue(preamble)
                    # Show it too, so the transcript matches what was said.
                    await self._emit(Event.assistant_delta(preamble))
                    await self._emit(event)

                elif event.type is EventType.TOOL_RESULT:
                    await self._emit(event)
                    summary = str(event.data.get("summary") or "")
                    if event.data.get("ok") and summary:
                        await enqueue(summary)
                        await self._emit(Event.assistant_delta(summary))

                elif event.type is EventType.ASSISTANT_DONE:
                    recorded = True
                    await self._emit(event)

                elif event.type is EventType.STATE:
                    # The loop reports thinking/acting; keep SPEAKING if the
                    # speaker is already busy, otherwise mirror it.
                    if self.state is not PipelineState.SPEAKING:
                        await self._set_state(
                            PipelineState(str(event.data.get("state", "thinking")))
                        )

                else:
                    await self._emit(event)

            remainder = splitter.flush()
            if remainder and not self._interrupt.is_set():
                await enqueue(remainder)
        finally:
            self._speak_queue = None
            with contextlib.suppress(Exception):
                await sentences.put(None)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await speaker_task

            # If the user cut us off mid-sentence, keep what was already said
            # so the model does not repeat itself on the next turn.
            partial = "".join(collected).strip()
            if partial and not recorded:
                self.agent.append(Message(role="assistant", content=partial))

        # Exactly one audio.end per turn, whether or not any audio was made.
        await self._emit(
            Event.audio_end(
                seq=seq_holder[0],
                reason="interrupted" if self._interrupt.is_set() else "complete",
            )
        )
        await self._emit(
            Event.metric(
                "turn_total_ms",
                (time.monotonic() - self._marks.get("turn_start", time.monotonic())) * 1000,
                source=source,
            )
        )
        await self._set_state(
            PipelineState.LISTENING if self._capturing else PipelineState.IDLE
        )

    # ==================================================================
    # text to speech
    # ==================================================================

    async def _tts_worker(self, queue: asyncio.Queue[str | None], seq_holder: list[int]) -> None:
        """Synthesize and emit audio, prefetching the next sentence while the
        current one plays. This is what keeps inter-sentence gaps small.

        ``audio.end`` is *not* emitted here: a turn can also fail before any
        audio is produced (bad ASR, LLM error), and the client must be able to
        finalise either way. The turn always emits exactly one, in ``_respond``.
        """
        seq = 0
        pending: asyncio.Task[list[AudioChunk]] | None = None
        stop_after_playback = False

        try:
            while True:
                if pending is None:
                    sentence = await queue.get()
                    if sentence is None:
                        return
                    pending = asyncio.create_task(self._synthesize(sentence))

                chunks = await pending
                pending = None

                # Start generating the next sentence *before* we play this one.
                if not stop_after_playback:
                    with contextlib.suppress(asyncio.QueueEmpty):
                        following = queue.get_nowait()
                        if following is None:
                            stop_after_playback = True
                        else:
                            pending = asyncio.create_task(self._synthesize(following))

                if chunks:
                    await self._set_state(PipelineState.SPEAKING)
                for chunk in chunks:
                    if self._interrupt.is_set():
                        return
                    await self._emit_audio(chunk, seq)
                    seq += 1

                if stop_after_playback:
                    return
        finally:
            seq_holder[0] = seq
            if pending is not None:
                pending.cancel()

    async def _synthesize(self, text: str) -> list[AudioChunk]:
        text = truncate_for_speech(text, self.settings.pipeline.max_sentence_chars * 2)
        if not text.strip():
            return []
        chunks: list[AudioChunk] = []
        try:
            async for chunk in self.bundle.tts.synthesize(
                text,
                voice=self.settings.tts.voice,
                language=self.settings.asr.language,
            ):
                chunks.append(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(Event.error(f"TTS failed: {exc}", where="tts"))
        return chunks

    async def _say(self, text: str) -> None:
        """Speak a short line, keeping ordering with the turn's audio queue."""
        if not text.strip():
            return
        if self._speak_queue is not None and not self._interrupt.is_set():
            await self._speak_queue.put(text)
            return
        await self._set_state(PipelineState.SPEAKING)
        for seq, chunk in enumerate(await self._synthesize(text)):
            if self._interrupt.is_set():
                return
            await self._emit_audio(chunk, seq)

    async def _emit_audio(self, chunk: AudioChunk, seq: int) -> None:
        if self.speaker is not None:
            await self.speaker.play(chunk)
        if "first_audio" not in self._marks:
            self._marks["first_audio"] = time.monotonic()
            await self._emit(
                Event.metric(
                    "tts_first_audio_ms",
                    (self._marks["first_audio"] - self._marks.get("turn_start",
                                                                  self._marks["first_audio"]))
                    * 1000,
                    provider=self.bundle.names.get("tts", ""),
                )
            )
        if self.emit_audio_payloads:
            await self._emit(Event.audio(chunk.data, chunk.mime, chunk.sample_rate, seq=seq))
        else:
            await self._emit(
                Event(
                    type=EventType.AUDIO,
                    data={"mime": chunk.mime, "sample_rate": chunk.sample_rate, "seq": seq},
                )
            )

    # ==================================================================
    # plumbing
    # ==================================================================

    async def _set_state(self, state: PipelineState, **extra: object) -> None:
        self.state = state
        await self._emit(Event.state(state, **extra))

    async def _emit(self, event: Event) -> None:
        if self._closed:
            return
        with contextlib.suppress(Exception):
            await self.on_event(event)


def _tool_preamble(event: Event) -> str:
    """A short spoken line so a long tool run is never silent."""
    name = str(event.data.get("name", ""))
    return {
        "ask_claude_code": "好的，我让 Claude Code 去处理。",
        "ask_codex": "好的，我交给 Codex 处理。",
        "ask_dsh": "好的，我让 DSH 去做。",
        "run_shell": "好的，我执行一下。",
        "http_request": "好的，我去请求一下。",
        "write_file": "好的，我来写文件。",
    }.get(name, "好的，我来处理。")


__all__ = ["VoiceSession", "interpret_confirmation"]
