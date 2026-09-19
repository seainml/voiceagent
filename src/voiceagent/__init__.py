"""voiceagent — a voice-first office assistant with pluggable speech and tools.

The package is intentionally layered so that any single piece can be replaced:

    audio/      PCM plumbing, VAD, resampling, playback
    providers/  ASR / TTS / LLM adapters behind small Protocols
    tools/      things the agent can *do* (shell, HTTP, files, agent CLIs, MCP)
    agent/      the reasoning loop that turns transcripts into actions
    session.py  the orchestrator wiring ASR -> agent -> TTS with barge-in
    server/     FastAPI + WebSocket transport and the browser UI
    cli.py      the terminal entrypoint
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
