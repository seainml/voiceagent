#!/usr/bin/env python
"""End-to-end acceptance check, using the *real* providers.

Unlike the unit suite (which is all mocks), this exercises the machine:

  1. provider resolution and warm-up
  2. macOS ``say`` producing decodable audio
  3. the agent loop calling the real ``dsh`` CLI as a tool
  4. shell execution behind the spoken-confirmation flow
  5. a full WebSocket turn against the real FastAPI app

Run it after changing anything in the pipeline:

    .venv/bin/python scripts/smoke.py

Exits non-zero on the first failure.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceagent.audio.pcm import (  # noqa: E402
    PCM_MIME,
    WAV_MIME,
    AudioClip,
    concat,
    decode_wav,
)
from voiceagent.config import load_settings  # noqa: E402
from voiceagent.events import Event, EventType  # noqa: E402
from voiceagent.providers.base import LLMDelta, Message, ProviderUnavailable, ToolCall  # noqa: E402
from voiceagent.providers.registry import build_bundle, capability_report  # noqa: E402
from voiceagent.session import VoiceSession  # noqa: E402
from voiceagent.tools.builtin import build_registry  # noqa: E402

PASS = "\033[32m  ok  \033[0m"
FAIL = "\033[31m fail \033[0m"
SKIP = "\033[33m skip \033[0m"

results: list[tuple[str, bool]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    mark = PASS if ok else FAIL
    print(f"[{mark}] {name}" + (f"  \033[2m{detail}\033[0m" if detail else ""))


def skip(name: str, detail: str) -> None:
    print(f"[{SKIP}] {name}  \033[2m{detail}\033[0m")


class ScriptedLLM:
    """Returns a fixed sequence of turns, then repeats the last one."""

    name = "scripted"

    def __init__(self, script: list[object]) -> None:
        self.script = list(script)
        self.calls: list[list[Message]] = []

    def chat_stream(self, messages, *, tools=None, temperature=None, max_tokens=None):
        self.calls.append(list(messages))
        return self._stream()

    async def _stream(self):
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, str):
            for i in range(0, len(item), 8):
                yield LLMDelta(content=item[i : i + 8])
                await asyncio.sleep(0)
            yield LLMDelta(finish_reason="stop")
        else:
            yield LLMDelta(tool_calls=list(item), finish_reason="tool_calls")

    async def aclose(self) -> None:
        return None


def collect_events(session_events: list[Event]):
    async def sink(event: Event) -> None:
        session_events.append(event)

    return sink


# ---------------------------------------------------------------------------


async def step_providers() -> None:
    print("\n\033[1m1. providers\033[0m")
    settings = load_settings()
    report_data = capability_report(settings)
    resolved = report_data["resolved"]
    print(f"     resolved: {resolved}")
    print(f"     platform: {report_data['platform']}")
    try:
        bundle = build_bundle(settings)
    except ProviderUnavailable as exc:
        report("build providers", False, str(exc))
        return

    started = time.monotonic()
    warmed = await bundle.warmup()
    report(
        "build providers",
        True,
        f"warmup={warmed or 'nothing to warm'} in {time.monotonic() - started:.2f}s",
    )

    # --- real TTS ---------------------------------------------------------
    probe = "语音链路自检完成。"
    chunks = []
    started = time.monotonic()
    async for chunk in bundle.tts.synthesize(probe, language=settings.asr.language):
        chunks.append(chunk)
    elapsed = time.monotonic() - started
    if not chunks:
        report("tts produces audio", False, "no chunks")
    elif chunks[0].mime == PCM_MIME:
        report("tts produces audio", True, f"{len(chunks)} PCM chunks in {elapsed:.2f}s")
    elif chunks[0].mime == WAV_MIME:
        clip = decode_wav(chunks[0].data)
        report(
            "tts produces audio",
            clip.duration_s > 0.2,
            f"{clip.duration_s:.2f}s of {clip.sample_rate} Hz PCM in {elapsed:.2f}s",
        )
    else:
        report("tts produces audio", len(chunks[0].data) > 1000,
               f"{len(chunks)} chunks of {chunks[0].mime}")

    await bundle.aclose()


async def step_agent_cli() -> None:
    """Call the real dsh CLI through the tool layer."""
    print("\n\033[1m2. agent CLI tool (real subprocess)\033[0m")
    settings = load_settings()
    if not settings.tools.agents.get("dsh", None):
        skip("ask_dsh", "not configured")
        return

    from shutil import which

    binary = settings.tools.agents["dsh"].command
    if not which(binary):
        skip("ask_dsh", f"{binary} not on PATH")
        return

    bundle = build_bundle(settings)
    registry = await build_registry(settings, include_mcp=False)
    events: list[Event] = []

    llm = ScriptedLLM([
        [ToolCall(id="c1", name="ask_dsh", arguments=json.dumps(
            {"task": "只回复两个字：pong"}, ensure_ascii=False))],
        "DSH 回复了 pong，链路正常。",
    ])
    bundle.llm = llm

    session = VoiceSession(
        settings,
        bundle,
        registry,
        on_event=collect_events(events),
        workspace=settings.workspace_path(),
        auto_confirm=True,          # the confirmation flow is covered separately
        emit_audio_payloads=False,  # do not pay for base64 we will not use
    )
    await session.start()

    started = time.monotonic()
    await session.submit_text("让 dsh 回复 pong")
    await session.wait_for_turn()
    elapsed = time.monotonic() - started
    await session.aclose()
    await bundle.aclose()

    calls = [e for e in events if e.type is EventType.TOOL_CALL]
    tool_results = [e for e in events if e.type is EventType.TOOL_RESULT]
    if not calls:
        report("ask_dsh invoked", False, "the model never called the tool")
        return
    report("ask_dsh invoked", True, f"{[c.data['name'] for c in calls]}")

    result = tool_results[-1] if tool_results else None
    if result is None:
        report("ask_dsh returned", False, "no tool result")
        return
    ok = bool(result.data.get("ok"))
    report("ask_dsh returned", ok, f"ok={ok} in {elapsed:.1f}s")
    if not ok:
        detail = next(
            (e.data.get("message", "") for e in events if e.type is EventType.ERROR), ""
        )
        content = next(
            (e.data.get("summary", "") for e in tool_results), ""
        )
        print(f"       \033[2m{detail[:200]} {content[:200]}\033[0m")
        print("       \033[2m(an unauthenticated CLI is expected to fail here — "
              "the chain itself is what is being tested)\033[0m")


async def step_confirmation() -> None:
    """A dangerous tool must wait for the answer before running."""
    print("\n\033[1m3. spoken confirmation gate\033[0m")
    settings = load_settings()
    bundle = build_bundle(settings)
    registry = await build_registry(settings, include_mcp=False)
    events: list[Event] = []

    llm = ScriptedLLM([
        [ToolCall(id="c1", name="run_shell",
                  arguments=json.dumps({"command": "echo smoke-test-ok"}))],
        "命令已经执行完了。",
    ])
    bundle.llm = llm

    session = VoiceSession(
        settings, bundle, registry,
        on_event=collect_events(events),
        workspace=settings.workspace_path(),
        emit_audio_payloads=False,
    )
    await session.start()
    await session.submit_text("执行一条测试命令")

    # Wait for the question, then decline in words.
    for _ in range(200):
        if any(e.type is EventType.CONFIRM_REQUEST for e in events):
            break
        await asyncio.sleep(0.01)

    asked = any(e.type is EventType.CONFIRM_REQUEST for e in events)
    report("dangerous tool asks first", asked,
           next((e.data["question"] for e in events
                 if e.type is EventType.CONFIRM_REQUEST), "")[:60])

    await session.submit_text("取消")
    await session.wait_for_turn()
    declined = [e for e in events
                if e.type is EventType.TOOL_RESULT and e.data.get("declined")]
    report("declining blocks execution", bool(declined))
    await session.aclose()

    # Now the same thing, but approved.
    events.clear()
    bundle2 = build_bundle(settings)
    bundle2.llm = ScriptedLLM([
        [ToolCall(id="c1", name="run_shell",
                  arguments=json.dumps({"command": "echo smoke-test-ok"}))],
        "执行完成。",
    ])
    session2 = VoiceSession(
        settings, bundle2, registry,
        on_event=collect_events(events),
        workspace=settings.workspace_path(),
        emit_audio_payloads=False,
    )
    await session2.start()
    await session2.submit_text("执行一条测试命令")
    for _ in range(200):
        if any(e.type is EventType.CONFIRM_REQUEST for e in events):
            break
        await asyncio.sleep(0.01)
    await session2.submit_text("确认")
    await session2.wait_for_turn()
    ran = [e for e in events if e.type is EventType.TOOL_RESULT and e.data.get("ok")]
    report("confirming runs the command", bool(ran))
    await session2.aclose()
    await bundle.aclose()
    await bundle2.aclose()


async def step_server() -> None:
    """A real WebSocket turn against the real app."""
    print("\n\033[1m4. web server (real WebSocket round trip)\033[0m")
    try:
        from fastapi.testclient import TestClient
    except ImportError:  # pragma: no cover
        skip("websocket turn", "fastapi test client unavailable")
        return

    from voiceagent.server.app import create_app

    settings = load_settings()
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/api/health")
        report("GET /api/health", health.status_code == 200,
               health.json().get("version", ""))

        index = client.get("/")
        report("GET / serves the UI", index.status_code == 200
               and "voiceagent" in index.text)

        with client.websocket_connect("/ws") as ws:
            ready = json.loads(ws.receive_text())
            report(
                "websocket ready",
                ready["type"] == "ready",
                f"asr={ready['providers']['asr']} tts={ready['providers']['tts']} "
                f"llm={ready['providers']['llm']} tools={len(ready['tools'])}",
            )

            ws.send_text(json.dumps({"type": "ping"}))
            got_pong = False
            for _ in range(50):
                if json.loads(ws.receive_text())["type"] == "pong":
                    got_pong = True
                    break
            report("websocket ping/pong", got_pong)

            if ready["providers"]["asr"] == "browser":
                skip(
                    "audio turn",
                    "server ASR is 'browser' — the UI does recognition client-side, "
                    "or install .[local-asr] / set an API key",
                )
                return

            # Close the loop for real: synthesize an utterance, send it back as
            # if the user had spoken it, and wait for the answer. TTS -> ASR ->
            # LLM -> TTS, with no mocks anywhere.
            probe = "帮我记一下，明天上午十点开会。"
            probe_bundle = build_bundle(settings)
            clips: list[AudioClip] = []
            async for chunk in probe_bundle.tts.synthesize(probe):
                if chunk.is_pcm:
                    clips.append(chunk.as_clip())
                else:
                    clips.append(decode_wav(chunk.data))
            await probe_bundle.aclose()
            if not clips:
                report("synthesize probe utterance", False, "TTS produced no audio")
                return
            utterance = concat(clips).resample(16000)
            if utterance.duration_s < 0.2:
                report("synthesize probe utterance", False, "audio too short")
                return
            report("synthesize probe utterance", True, f"{utterance.duration_s:.1f}s")

            ws.send_text(json.dumps({"type": "audio.start"}))
            size = 32000  # 1 s of 16 kHz PCM16 per frame
            for offset in range(0, len(utterance.pcm), size):
                ws.send_text(json.dumps({
                    "type": "audio.chunk",
                    "pcm": base64.b64encode(utterance.pcm[offset : offset + size]).decode("ascii"),
                    "sample_rate": 16000,
                }))
            ws.send_text(json.dumps({"type": "audio.end"}))

            transcript = ""
            saw_audio = False
            saw_error = None
            started = time.monotonic()
            for _ in range(2000):
                event = json.loads(ws.receive_text())
                if event["type"] == "transcript":
                    transcript = event.get("text", "")
                if event["type"] == "audio":
                    saw_audio = True
                if event["type"] == "error":
                    saw_error = event["message"]
                if event["type"] == "audio.end":
                    break
            elapsed = time.monotonic() - started

            report("ASR understood the utterance", bool(transcript), f"“{transcript}”")
            report(
                "full voice round trip",
                saw_audio and not saw_error,
                f"audio back in {elapsed:.1f}s" if saw_audio else (saw_error or "no audio"),
            )


async def main() -> int:
    print("\033[1mvoiceagent smoke test\033[0m")
    await step_providers()
    await step_agent_cli()
    await step_confirmation()
    await step_server()

    failed = [name for name, ok in results if not ok]
    print()
    if failed:
        print(f"\033[31m{len(failed)} check(s) failed:\033[0m " + ", ".join(failed))
        return 1
    print(f"\033[32mall {len(results)} checks passed\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
