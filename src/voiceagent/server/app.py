"""FastAPI application: static UI, capability report, and the voice WebSocket.

The WebSocket protocol is deliberately small — a front-end only has to
understand the events from :mod:`voiceagent.events` plus four client verbs. See
``docs/architecture.md`` for the full table.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..audio.pcm import decode_wav
from ..config import Settings, load_settings
from ..events import Event, EventType
from ..providers.base import ProviderUnavailable
from ..providers.registry import ProviderBundle, build_bundle, capability_report
from ..session import VoiceSession
from ..tools.builtin import build_registry

STATIC_DIR = Path(__file__).parent / "static"


class AppState:
    """Shared, lazily-built process state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._bundle: ProviderBundle | None = None
        self._registry = None
        self._lock = asyncio.Lock()
        self._warmup_task: asyncio.Task[None] | None = None
        self.started_at = time.time()

    async def resources(self):
        async with self._lock:
            if self._bundle is None:
                self._bundle = build_bundle(self.settings)
            if self._registry is None:
                self._registry = await build_registry(self.settings)
            if self._warmup_task is None:
                # Loading a local ASR model takes seconds; do it while the user
                # is still reading the page rather than on their first sentence.
                self._warmup_task = asyncio.create_task(self._bundle.warmup())
            return self._bundle, self._registry

    async def aclose(self) -> None:
        if self._warmup_task is not None:
            self._warmup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._warmup_task
            self._warmup_task = None
        if self._registry is not None:
            await self._registry.aclose()
            self._registry = None
        if self._bundle is not None:
            await self._bundle.aclose()
            self._bundle = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    state = AppState(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await state.aclose()

    app = FastAPI(title="voiceagent", version=__version__, lifespan=lifespan)
    app.state.voiceagent = state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- REST ----------------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "uptime_s": round(time.time() - state.started_at, 1),
            "session_ready": state._bundle is not None,
        }

    @app.get("/api/capabilities")
    async def capabilities() -> dict[str, Any]:
        report = capability_report(settings)
        try:
            bundle, registry = await state.resources()
            report["resolved"] = bundle.names
            report["tools"] = registry.names()
            report["tool_warnings"] = registry.warnings
        except ProviderUnavailable as exc:
            report["error"] = str(exc)
        return report

    @app.get("/api/tools")
    async def tools() -> dict[str, Any]:
        try:
            _bundle, registry = await state.resources()
        except ProviderUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "tools": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "dangerous": spec.name in registry.dangerous(),
                }
                for spec in registry.specs()
            ],
            "warnings": registry.warnings,
        }

    # -- UI ------------------------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # -- WebSocket -----------------------------------------------------

    @app.websocket("/ws")
    async def ws_endpoint(
        websocket: WebSocket,
        token: str | None = Query(default=None),
    ) -> None:
        if settings.server.auth_token and token != settings.server.auth_token:
            await websocket.close(code=4401, reason="invalid token")
            return
        await websocket.accept()
        await _serve_session(websocket, state, settings)

    return app


async def _serve_session(websocket: WebSocket, state: AppState, settings: Settings) -> None:
    send_lock = asyncio.Lock()

    async def send(event: Event) -> None:
        # Every task in a session can emit; serialise the writes.
        async with send_lock:
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.send_text(json.dumps(event.to_wire(), ensure_ascii=False))

    session: VoiceSession | None = None
    try:
        try:
            bundle, registry = await state.resources()
        except ProviderUnavailable as exc:
            await send(Event.error(str(exc), where="startup"))
            await websocket.close(code=1011, reason="provider unavailable")
            return

        session = VoiceSession(
            settings,
            bundle,
            registry,
            on_event=send,
            session_id=f"ws-{int(time.time() * 1000)}",
        )
        await session.start()

        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await send(Event.error("malformed frame", where="ws"))
                continue
            await _handle_frame(session, message, send)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        with contextlib.suppress(Exception):
            await send(Event.error(f"{type(exc).__name__}: {exc}", where="ws"))
    finally:
        if session is not None:
            await session.aclose()
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close()


async def _handle_frame(session: VoiceSession, message: dict[str, Any], send) -> None:
    """Client verbs.

    ``audio.start`` / ``audio.chunk``* / ``audio.end``, then ``text``,
    ``barge_in``, ``interrupt``, ``confirm``, ``metric`` or ``ping``.

    The utterance is only closed by ``audio.end``: a chunk is a *fragment* of
    what the user is saying, and treating each one as a finished turn shears
    every sentence into 40 ms pieces.
    """
    kind = str(message.get("type") or "")

    if kind == "audio.start":
        await session.begin_utterance()

    elif kind == "audio.chunk":
        try:
            pcm = base64.b64decode(message.get("pcm") or "")
        except (ValueError, TypeError):
            await send(Event.error("audio.chunk is not valid base64", where="ws"))
            return
        sample_rate = int(message.get("sample_rate") or 16000)
        if message.get("mime") == "audio/wav":
            # A WAV-flavoured chunk carries its own container.
            pcm = decode_wav(pcm).pcm
        await session.push_audio(pcm, sample_rate)

    elif kind == "audio.end":
        await session.end_utterance()

    elif kind in {"text", "transcript"}:
        await session.submit_text(str(message.get("text") or ""))

    elif kind == "interrupt":
        await session.interrupt(reason="client")

    elif kind == "barge_in":
        # Sent by the browser the moment its VAD hears speech over playback.
        await session.barge_in()

    elif kind == "confirm":
        session.resolve_confirmation(bool(message.get("approved")))

    elif kind == "metric":
        # Client-side measurements (real time-to-first-audio cannot be seen
        # from the server: only the browser knows when playback began).
        try:
            value = float(message.get("value"))
        except (TypeError, ValueError):
            return
        await send(
            Event.metric(
                str(message.get("name") or "client_metric"),
                value,
                source="client",
            )
        )

    elif kind == "ping":
        await send(Event(type=EventType.PONG, data={"t": time.time()}))

    else:
        await send(Event.error(f"unknown frame type {kind!r}", where="ws"))


def default_app() -> FastAPI:  # pragma: no cover - convenience for `uvicorn`
    """Factory for ``uvicorn voiceagent.server.app:default_app --factory``."""
    return create_app()


__all__ = ["AppState", "create_app", "default_app"]
