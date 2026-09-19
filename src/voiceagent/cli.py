"""``voiceagent`` command line.

    voiceagent doctor            what is wired up on this machine?
    voiceagent serve             browser UI on http://127.0.0.1:8765
    voiceagent chat              typed conversation with full tool access
    voiceagent talk              microphone in, speaker out
    voiceagent tools             list the tools the model will see
    voiceagent say "..."         TTS smoke test
    voiceagent transcribe a.wav  ASR smoke test
    voiceagent init              write a starter voiceagent.toml
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import DEFAULT_CONFIG_FILE, load_settings
from .events import Event, EventType, PipelineState
from .providers.base import ProviderUnavailable
from .providers.registry import build_bundle, capability_report
from .session import VoiceSession
from .tools.builtin import build_registry

app = typer.Typer(
    add_completion=False,
    help="Voice-first office assistant with pluggable speech and tools.",
    no_args_is_help=True,
)
console = Console()

STATE_STYLE = {
    PipelineState.IDLE: "dim",
    PipelineState.LISTENING: "green",
    PipelineState.TRANSCRIBING: "cyan",
    PipelineState.THINKING: "blue",
    PipelineState.ACTING: "magenta",
    PipelineState.SPEAKING: "yellow",
    PipelineState.AWAITING_CONFIRMATION: "yellow",
}


def _version(value: bool) -> None:  # pragma: no cover - typer callback
    if value:
        console.print(f"voiceagent {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True),
) -> None:
    """Voice-first office assistant."""


# ---------------------------------------------------------------------------
# doctor / introspection
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Show which providers and tools are usable right now."""
    settings = load_settings()
    report = capability_report(settings)

    console.print(Panel.fit(
        f"[bold]voiceagent[/bold] {__version__}   [dim]{report['platform']}[/dim]",
        border_style="blue",
    ))

    resolved = report["resolved"]
    table = Table(title="Providers", show_lines=False, title_justify="left")
    table.add_column("kind", style="bold")
    table.add_column("provider")
    table.add_column("available")
    table.add_column("note", style="dim")
    for kind in ("asr", "tts", "llm"):
        for index, entry in enumerate(report["providers"][kind]):
            mark = "[green]yes[/green]" if entry["available"] else "[red]no[/red]"
            chosen = " [cyan]<- using[/cyan]" if entry["name"] == resolved.get(kind) else ""
            note = entry["reason"] or ("default" if entry["default"] else "")
            table.add_row(
                kind if index == 0 else "",
                f"{entry['name']}{chosen}",
                mark,
                # Reasons contain things like '.[local-asr]' which Rich would
                # otherwise swallow as markup.
                escape(note),
            )
    console.print(table)

    binaries = Table(title="External binaries", title_justify="left")
    binaries.add_column("name")
    binaries.add_column("path", style="dim")
    for name, path in report["tools"].items():
        binaries.add_row(name, path or "[red]not found[/red]")
    console.print(binaries)

    if resolved.get("llm") == "echo":
        console.print(
            Panel(
                "No LLM is configured, so replies will just repeat what you said.\n"
                "Set a key to unlock real answers:\n\n"
                "    export DEEPSEEK_API_KEY=sk-...        # or any OpenAI-compatible endpoint\n"
                "    export VA_LLM__BASE_URL=http://127.0.0.1:11434/v1   # Ollama / vLLM / LM Studio",
                title="heads up",
                border_style="yellow",
            )
        )


@app.command()
def tools() -> None:
    """List the tools the model is given."""

    async def run() -> None:
        settings = load_settings()
        registry = await build_registry(settings)
        dangerous = registry.dangerous()
        table = Table(title=f"{len(registry)} tools", title_justify="left")
        table.add_column("name", style="cyan")
        table.add_column("confirm")
        table.add_column("description", style="dim")
        for spec in registry.specs():
            table.add_row(
                spec.name,
                "[yellow]yes[/yellow]" if spec.name in dangerous else "no",
                spec.description.strip().splitlines()[0][:90],
            )
        console.print(table)
        for warning in registry.warnings:
            console.print(f"[yellow]warning:[/yellow] {warning}")
        await registry.aclose()

    asyncio.run(run())


@app.command()
def config() -> None:
    """Print the resolved configuration (secrets are never included)."""
    settings = load_settings()
    console.print_json(json.dumps(settings.describe(), ensure_ascii=False, indent=2, default=str))


@app.command()
def init(force: bool = typer.Option(False, "--force", help="Overwrite an existing file.")) -> None:
    """Write a starter voiceagent.toml."""
    target = Path(DEFAULT_CONFIG_FILE)
    if target.exists() and not force:
        console.print(f"[yellow]{target} already exists[/yellow] — use --force to overwrite")
        raise typer.Exit(code=1)
    source = Path(__file__).resolve().parents[2] / "voiceagent.example.toml"
    if source.is_file():
        target.write_text(source.read_text("utf-8"), encoding="utf-8")
        console.print(f"[green]wrote[/green] {target}")
    else:  # pragma: no cover - packaging fallback
        target.write_text("# see docs/configuration.md\n", encoding="utf-8")
        console.print(f"[green]wrote[/green] {target} (minimal)")


# ---------------------------------------------------------------------------
# smoke tests
# ---------------------------------------------------------------------------


@app.command()
def say(
    text: str = typer.Argument(..., help="Text to synthesize."),
    play: bool = typer.Option(True, "--play/--no-play", help="Also play it."),
    out: Path | None = typer.Option(None, "--out", help="Write the audio to this file."),
) -> None:
    """Synthesize speech — a quick check that TTS works."""

    async def run() -> None:
        from .audio.pcm import PCM_MIME, AudioClip, concat
        from .audio.player import Speaker

        settings = load_settings()
        bundle = build_bundle(settings)
        console.print(f"[dim]tts provider:[/dim] {bundle.names['tts']}")

        speaker = Speaker() if play else None
        chunks = []
        started = time.monotonic()
        try:
            async for chunk in bundle.tts.synthesize(text, language=settings.asr.language):
                chunks.append(chunk)
                if speaker is not None:
                    await speaker.play(chunk)
        finally:
            if speaker is not None:
                await speaker.aclose()

        total = sum(len(c.data) for c in chunks)
        first = chunks[0] if chunks else None
        console.print(
            f"[green]{len(chunks)} chunks / {total} bytes[/green] "
            f"in {time.monotonic() - started:.2f}s"
            + (f" [dim](first: {first.mime})[/dim]" if first else "")
        )

        if out is not None:
            if first and first.mime == PCM_MIME:
                clip = concat([c.as_clip() for c in chunks])
                out.write_bytes(clip.to_wav_bytes())
            else:
                out.write_bytes(b"".join(c.data for c in chunks))
            console.print(f"[green]wrote[/green] {out}")
        _ = AudioClip
        await bundle.aclose()

    asyncio.run(run())


@app.command()
def transcribe(
    path: Path = typer.Argument(..., exists=True, readable=True),
    language: str | None = typer.Option(None, "--language"),
) -> None:
    """Transcribe a WAV file — a quick check that ASR works."""

    async def run() -> None:
        from .audio.pcm import decode_wav

        settings = load_settings()
        bundle = build_bundle(settings)
        console.print(f"[dim]asr provider:[/dim] {bundle.names['asr']}")
        clip = decode_wav(path.read_bytes())
        started = time.monotonic()
        result = await bundle.asr.transcribe(clip, language=language or settings.asr.language,
                                             hotwords=settings.asr.hotwords)
        elapsed = time.monotonic() - started
        console.print(Panel(result.text or "[dim](empty)[/dim]",
                            title=f"{clip.duration_s:.1f}s audio in {elapsed:.2f}s"))
        await bundle.aclose()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# interactive
# ---------------------------------------------------------------------------


def _render_event(event: Event, state: dict[str, Any]) -> None:
    """Pretty-print one session event for the terminal."""
    kind = event.type

    if kind is EventType.READY:
        names = event.data.get("providers", {})
        console.print(
            f"[dim]asr={names.get('asr')} tts={names.get('tts')} llm={names.get('llm')} "
            f"tools={len(event.data.get('tools', []))}[/dim]"
        )
        for warning in event.data.get("warnings", []):
            console.print(f"[yellow]warning:[/yellow] {warning}")
        if names.get("llm") == "echo":
            console.print("[yellow]no LLM configured — replies will be echoes[/yellow]")

    elif kind is EventType.STATE:
        state["state"] = str(event.data.get("state", ""))

    elif kind is EventType.TRANSCRIPT:
        console.print(f"\n[bold blue]你[/bold blue] {event.data.get('text', '')}")

    elif kind is EventType.PARTIAL_TRANSCRIPT:
        console.print(f"[dim]{event.data.get('text', '')}[/dim]", end="\r")

    elif kind is EventType.ASSISTANT_DELTA:
        if not state.get("streaming"):
            console.print("[bold green]助手[/bold green] ", end="")
            state["streaming"] = True
        console.print(Text(str(event.data.get("text", ""))), end="")

    elif kind is EventType.ASSISTANT_DONE:
        if state.get("streaming"):
            console.print()
            state["streaming"] = False
        elif event.data.get("text"):
            console.print(f"[bold green]助手[/bold green] {event.data['text']}")

    elif kind is EventType.TOOL_CALL:
        args = json.dumps(event.data.get("arguments", {}), ensure_ascii=False)
        console.print(f"  [magenta]→ {event.data.get('name')}[/magenta] [dim]{args[:160]}[/dim]")

    elif kind is EventType.TOOL_PROGRESS:
        console.print(f"    [dim]{event.data.get('message', '')}[/dim]")

    elif kind is EventType.TOOL_RESULT:
        ok = event.data.get("ok")
        colour = "green" if ok else "red"
        summary = event.data.get("summary") or ""
        console.print(f"  [{colour}]{'✓' if ok else '✗'} {summary}[/{colour}]")

    elif kind is EventType.CONFIRM_REQUEST:
        console.print(f"\n[yellow]需要确认：[/yellow]{event.data.get('question', '')}")
        console.print("[dim]直接说“确认”或“取消”，也可以在这里输入 y/n[/dim]")

    elif kind is EventType.ERROR:
        where = event.data.get("where") or ""
        console.print(f"[red]error{'[' + where + ']' if where else ''}: "
                      f"{event.data.get('message')}[/red]")

    elif kind is EventType.LOG and event.data.get("level") != "debug":
        console.print(f"[dim]{event.data.get('message')}[/dim]")

    elif kind is EventType.METRIC:
        state.setdefault("metrics", {})[str(event.data.get("name"))] = event.data.get("value")


async def _console_confirmer(session: VoiceSession, state: dict[str, Any]) -> None:
    """Fallback: answer pending confirmations from stdin."""
    while not state.get("closed"):
        if state.get("awaiting_confirm"):
            try:
                line = await asyncio.to_thread(input, "确认? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                return
            from .session import interpret_confirmation

            answer = interpret_confirmation(line)
            session.resolve_confirmation(bool(answer))
            state["awaiting_confirm"] = False
        await asyncio.sleep(0.1)


async def _make_session(
    *,
    session_id: str,
    speak_locally: bool,
    emit_audio: bool,
    auto_confirm: bool,
) -> tuple[VoiceSession, dict[str, Any], Any]:
    settings = load_settings()
    bundle = build_bundle(settings)
    registry = await build_registry(settings)

    state: dict[str, Any] = {}

    async def on_event(event: Event) -> None:
        if event.type is EventType.CONFIRM_REQUEST:
            state["awaiting_confirm"] = True
        _render_event(event, state)

    from .audio.player import Speaker

    speaker = Speaker() if speak_locally else None
    session = VoiceSession(
        settings,
        bundle,
        registry,
        on_event=on_event,
        session_id=session_id,
        speaker=speaker,
        auto_confirm=auto_confirm,
        emit_audio_payloads=emit_audio,
    )
    state["session"] = session
    return session, state, bundle


@app.command()
def chat(
    auto_confirm: bool = typer.Option(
        False, "--yes", help="Skip the confirmation prompt (use with care)."
    ),
) -> None:
    """Typed conversation with the full tool set."""
    asyncio.run(_chat(session_id="cli-chat", auto_confirm=auto_confirm))


async def _chat(*, session_id: str, auto_confirm: bool) -> None:
    session, state, bundle = await _make_session(
        session_id=session_id, speak_locally=False, emit_audio=False, auto_confirm=auto_confirm
    )
    confirmer = asyncio.create_task(_console_confirmer(session, state))
    try:
        await session.start()
        console.print("[dim]输入内容后回车；:quit 退出。Ctrl-C 打断当前回答。[/dim]")
        while True:
            try:
                line = await asyncio.to_thread(input, "\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            stripped = line.strip()
            if stripped in {":q", ":quit", "exit", "quit"}:
                break
            if not stripped:
                continue
            await session.submit_text(stripped)
            await session.wait_for_turn()
    finally:
        state["closed"] = True
        confirmer.cancel()
        await session.aclose()
        await bundle.aclose()
        if session.speaker is not None:
            await session.speaker.aclose()
        console.print("\n[dim]bye[/dim]")


@app.command()
def talk(
    auto_confirm: bool = typer.Option(
        False, "--yes", help="Skip the confirmation prompt (use with care)."
    ),
) -> None:
    """Voice loop: microphone in, speaker out."""
    asyncio.run(_talk(session_id="cli-talk", auto_confirm=auto_confirm))


async def _talk(*, session_id: str, auto_confirm: bool) -> None:
    from .audio.capture import MicrophoneUnavailable, listen
    from .audio.vad import VADConfig

    settings = load_settings()
    session, state, bundle = await _make_session(
        session_id=session_id, speak_locally=True, emit_audio=False, auto_confirm=auto_confirm
    )
    confirmer = asyncio.create_task(_console_confirmer(session, state))
    try:
        await session.start()
        backend = session.speaker.backend if session.speaker else "none"
        console.print(f"[dim]麦克风已打开，播放后端 {backend}。Ctrl-C 退出。[/dim]")
        try:
            vad = VADConfig(
                silence_ms=settings.pipeline.vad_silence_ms,
                threshold=settings.pipeline.vad_threshold,
                min_utterance_ms=settings.pipeline.asr_min_utterance_ms,
                max_utterance_s=settings.pipeline.max_utterance_s,
            )
            async for clip in listen(vad=vad, sample_rate=16000):
                # Half-duplex: without acoustic echo cancellation the terminal
                # microphone hears the assistant's own voice, so anything
                # captured while it is speaking is almost certainly not the
                # user. Barge-in from the terminal is the Ctrl-C path instead.
                if session.state is PipelineState.SPEAKING:
                    continue
                await session.begin_utterance()
                await session.push_audio(clip.pcm, clip.sample_rate)
                await session.end_utterance()
                await session.wait_for_turn()
        except MicrophoneUnavailable as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        pass
    finally:
        state["closed"] = True
        confirmer.cancel()
        await session.aclose()
        await bundle.aclose()
        if session.speaker is not None:
            await session.speaker.aclose()
        console.print("\n[dim]bye[/dim]")


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind address. Default from config."),
    port: int | None = typer.Option(None, help="Port. Default from config."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Run the browser UI and WebSocket server."""
    import uvicorn

    settings = load_settings()
    from .server.app import create_app

    bind_host = host or settings.server.host
    bind_port = port or settings.server.port
    application = create_app(settings)

    console.print(Panel.fit(
        f"[bold]voiceagent[/bold] listening on [cyan]http://{bind_host}:{bind_port}[/cyan]\n"
        f"[dim]asr={settings.asr.provider} tts={settings.tts.provider} "
        f"llm={settings.llm.provider}[/dim]",
        border_style="blue",
    ))
    if bind_host not in {"127.0.0.1", "localhost"} and not settings.server.auth_token:
        console.print("[yellow]warning:[/yellow] binding to a public interface with no "
                      "server.auth_token set")
    uvicorn.run(application, host=bind_host, port=bind_port, reload=reload, log_level="info")


def _provider_error(exc: ProviderUnavailable) -> None:
    console.print(f"[red]{exc}[/red]")
    console.print("[dim]run `voiceagent doctor` to see what is available[/dim]")


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except ProviderUnavailable as exc:
        _provider_error(exc)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
