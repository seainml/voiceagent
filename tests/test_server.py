"""HTTP + WebSocket transport."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.conftest import make_settings
from voiceagent.server.app import create_app


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tools={"workspace": str(tmp_path)})
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def authed_client(tmp_path):
    settings = make_settings(
        server={"auth_token": "s3cret"}, tools={"workspace": str(tmp_path)}
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


class TestRest:
    def test_health(self, client):
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        assert payload["version"]

    def test_capabilities_reports_resolved_providers(self, client):
        payload = client.get("/api/capabilities").json()
        assert payload["resolved"] == {"asr": "mock", "tts": "mock", "llm": "mock"}
        assert payload["providers"]["tts"]

    def test_tools_endpoint_lists_confirmation_flags(self, client):
        payload = client.get("/api/tools").json()
        names = {tool["name"]: tool for tool in payload["tools"]}
        assert names["run_shell"]["dangerous"] is True
        assert names["read_file"]["dangerous"] is False

    def test_index_serves_the_ui(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "voiceagent" in response.text

    def test_static_assets_are_mounted(self, client):
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/static/pcm-worklet.js").status_code == 200


class TestWebSocket:
    @staticmethod
    def drain_until(ws, predicate, limit: int = 600) -> list[dict]:
        """Read frames until ``predicate(event)`` is true.

        The session emits `state` frames between the ones a test cares about,
        so tests must not assume the very next frame is the interesting one.
        """
        seen: list[dict] = []
        for _ in range(limit):
            event = json.loads(ws.receive_text())
            seen.append(event)
            if predicate(event):
                break
        return seen

    def test_ready_then_a_text_turn(self, client):
        with client.websocket_connect("/ws") as ws:
            ready = json.loads(ws.receive_text())
            assert ready["type"] == "ready"
            assert ready["providers"]["llm"] == "mock"
            assert ready["vad"]["silence_ms"] > 0

            ws.send_text(json.dumps({"type": "text", "text": "你好"}))
            # Drain to `audio.end` — the point at which every TTS chunk has
            # been sent. `assistant.done` only means the *text* is complete,
            # and legitimately arrives while the speaker is still going.
            seen = self.drain_until(ws, lambda e: e["type"] == "audio.end")

            kinds = [e["type"] for e in seen]
            assert "transcript" in kinds
            assert "assistant.delta" in kinds
            assert "assistant.done" in kinds
            assert "audio" in kinds
            assert "state" in kinds

    def test_ping_pong(self, client):
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())  # ready
            ws.send_text(json.dumps({"type": "ping"}))
            seen = self.drain_until(ws, lambda e: e["type"] == "pong")
            assert seen[-1]["type"] == "pong"

    def test_unknown_frame_type_is_reported(self, client):
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "nonsense"}))
            seen = self.drain_until(ws, lambda e: e["type"] == "error")
            assert "nonsense" in seen[-1]["message"]

    def test_malformed_json_is_reported(self, client):
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())
            ws.send_text("{not json")
            seen = self.drain_until(ws, lambda e: e["type"] == "error")
            assert seen

    def test_bad_base64_audio_is_reported(self, client):
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "audio.start"}))
            ws.send_text(json.dumps({"type": "audio.chunk", "pcm": "!!!not base64!!!"}))
            seen = self.drain_until(ws, lambda e: e["type"] == "error")
            assert seen

    def test_audio_round_trip_reaches_the_transcriber(self, client):
        import base64

        from voiceagent.audio.pcm import AudioClip

        pcm = base64.b64encode(AudioClip.silence(500).pcm).decode("ascii")
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "audio.start"}))
            ws.send_text(json.dumps({"type": "audio.chunk", "pcm": pcm, "sample_rate": 16000}))
            ws.send_text(json.dumps({"type": "audio.end"}))

            seen = self.drain_until(ws, lambda e: e["type"] == "assistant.done")
            # MockASR returns its default text, so the loop must complete.
            assert any(e["type"] == "transcript" for e in seen)
    def test_an_utterance_spans_multiple_audio_chunks(self, client):
        """Regression: `end_utterance` used to fire on every chunk.

        That sheared each sentence into 40 ms pieces, so every turn was either
        "too short" or transcribed a fragment of a word.
        """
        import base64

        from voiceagent.audio.pcm import AudioClip

        pcm = base64.b64encode(AudioClip.silence(400).pcm).decode("ascii")
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())  # ready
            ws.send_text(json.dumps({"type": "audio.start"}))
            for _ in range(4):
                ws.send_text(json.dumps({
                    "type": "audio.chunk", "pcm": pcm, "sample_rate": 16000,
                }))
            ws.send_text(json.dumps({"type": "audio.end"}))

            seen = self.drain_until(ws, lambda e: e["type"] == "audio.end")

        end = seen[-1]
        assert end["reason"] != "too_short", "the chunks were not assembled"
        assert len([e for e in seen if e["type"] == "transcript"]) == 1
        assert [e for e in seen if e["type"] == "error"] == []

    def test_interrupt_is_accepted(self, client):
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "interrupt"}))
            seen = self.drain_until(ws, lambda e: e["type"] == "state")
            assert seen[-1]["state"] in {"idle", "listening"}


class TestAuth:
    def test_missing_token_closes_the_socket(self, authed_client):
        with (
            pytest.raises(WebSocketDisconnect),
            authed_client.websocket_connect("/ws") as ws,
        ):
            ws.receive_text()

    def test_valid_token_is_accepted(self, authed_client):
        with authed_client.websocket_connect("/ws?token=s3cret") as ws:
            assert json.loads(ws.receive_text())["type"] == "ready"

    def test_rest_endpoints_stay_open(self, authed_client):
        assert authed_client.get("/api/health").status_code == 200
