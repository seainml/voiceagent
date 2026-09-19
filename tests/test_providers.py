"""Provider contracts, auto-resolution and the built-in implementations."""

from __future__ import annotations

import platform

import pytest

from tests.conftest import make_settings, tool_call
from voiceagent.audio.pcm import PCM_MIME, AudioClip
from voiceagent.providers.base import (
    ASRProvider,
    LLMProvider,
    Message,
    ProviderUnavailable,
    ToolCall,
    TTSProvider,
)
from voiceagent.providers.llm.echo import EchoLLM
from voiceagent.providers.llm.mock import MockLLM
from voiceagent.providers.registry import (
    build_asr,
    build_bundle,
    build_llm,
    build_tts,
    capability_report,
    resolve_name,
)


class TestProtocols:
    def test_mock_providers_satisfy_their_protocols(self):
        from voiceagent.providers.asr.mock import MockASR
        from voiceagent.providers.tts.mock import MockTTS

        assert isinstance(MockASR(), ASRProvider)
        assert isinstance(MockTTS(), TTSProvider)
        assert isinstance(MockLLM(), LLMProvider)


class TestAutoResolution:
    """`resolve_name` only does work when the provider is left on "auto"."""

    def test_asr_falls_back_to_browser_when_nothing_is_installed(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            "voiceagent.providers.registry._module_available", lambda name: False
        )
        assert resolve_name("asr", make_settings(asr={"provider": "auto"})) == "browser"

    def test_asr_prefers_local_whisper(self, monkeypatch):
        monkeypatch.setattr(
            "voiceagent.providers.registry._module_available",
            lambda name: name == "faster_whisper",
        )
        assert resolve_name("asr", make_settings(asr={"provider": "auto"})) == "faster_whisper"

    def test_asr_uses_the_cloud_when_a_key_exists(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(
            "voiceagent.providers.registry._module_available", lambda name: False
        )
        assert resolve_name("asr", make_settings(asr={"provider": "auto"})) == "openai"

    def test_llm_falls_back_to_echo_without_a_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("VA_LLM__BASE_URL", raising=False)
        assert resolve_name("llm", make_settings(llm={"provider": "auto"})) == "echo"

    def test_llm_uses_openai_compat_when_a_key_exists(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        assert resolve_name("llm", make_settings(llm={"provider": "auto"})) == "openai"

    def test_llm_uses_openai_compat_for_an_explicit_local_endpoint(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("VA_LLM__BASE_URL", "http://127.0.0.1:11434/v1")
        assert resolve_name("llm", make_settings(llm={"provider": "auto"})) == "openai"

    def test_explicit_provider_wins_over_auto(self):
        settings = make_settings(llm={"provider": "mock"})
        assert resolve_name("llm", settings) == "mock"


class TestFactories:
    def test_builds_a_bundle_with_mock_providers(self):
        bundle = build_bundle(make_settings())
        assert bundle.names == {"asr": "mock", "tts": "mock", "llm": "mock"}

    def test_openai_asr_without_a_key_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderUnavailable):
            build_asr(make_settings(asr={"provider": "openai"}))

    def test_openai_tts_without_a_key_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderUnavailable):
            build_tts(make_settings(tts={"provider": "openai"}))

    def test_openai_llm_without_a_key_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("VA_LLM__API_KEY", raising=False)
        with pytest.raises(ProviderUnavailable):
            build_llm(make_settings(llm={"provider": "openai"}))

    def test_a_local_endpoint_needs_no_real_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        provider = build_llm(
            make_settings(llm={"provider": "openai", "base_url": "http://127.0.0.1:11434/v1"})
        )
        assert provider.name == "openai"

    def test_unknown_provider_name_is_rejected(self):
        from voiceagent.providers.base import ProviderError

        with pytest.raises(ProviderError):
            build_tts(make_settings(tts={"provider": "nope"}))

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_macos_say_is_the_default_tts_on_darwin(self):
        assert resolve_name("tts", make_settings(tts={"provider": "auto"})) == "macos_say"


class TestCapabilityReport:
    def test_report_has_the_expected_shape(self):
        report = capability_report(make_settings())
        assert set(report) >= {"platform", "tools", "providers", "resolved", "config"}
        for kind in ("asr", "tts", "llm"):
            assert report["providers"][kind]
            assert report["resolved"][kind]
            for entry in report["providers"][kind]:
                assert set(entry) == {"kind", "name", "available", "reason", "default"}


class TestMockProviders:
    async def test_mock_asr_returns_the_script_in_order(self):
        from voiceagent.providers.asr.mock import MockASR

        asr = MockASR(script=["第一条", "第二条"])
        clip = AudioClip.silence(500)
        assert (await asr.transcribe(clip)).text == "第一条"
        assert (await asr.transcribe(clip)).text == "第二条"
        assert await asr.transcribe(clip) is not None

    async def test_browser_asr_refuses_audio(self):
        from voiceagent.providers.asr.mock import BrowserASR
        from voiceagent.providers.base import ProviderError

        with pytest.raises(ProviderError):
            await BrowserASR().transcribe(AudioClip.silence(100))

    async def test_mock_tts_yields_pcm_per_sentence(self):
        from voiceagent.providers.tts.mock import MockTTS

        tts = MockTTS()
        chunks = [c async for c in tts.synthesize("第一句。第二句。")]
        assert len(chunks) == 2
        assert all(c.mime == PCM_MIME for c in chunks)
        assert all(len(c.data) > 0 for c in chunks)

    async def test_mock_llm_streams_and_can_request_tools(self):
        llm = MockLLM(script=["你好。", [tool_call("remember", '{"note":"x"}')]])
        first = "".join([
            d.content async for d in llm.chat_stream([Message(role="user", content="hi")])
        ])
        assert first == "你好。"
        deltas = [d async for d in llm.chat_stream([Message(role="user", content="hi")])]
        assert deltas[-1].tool_calls[0].name == "remember"
        assert deltas[-1].finish_reason == "tool_calls"

    async def test_echo_llm_repeats_what_it_heard(self):
        llm = EchoLLM()
        text = "".join([
            d.content async for d in llm.chat_stream([Message(role="user", content="开会")])
        ])
        assert "开会" in text
        assert "没有连接大模型" in text


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
class TestMacOSSayLive:
    """The real `say` binary, because its failure mode is silent emptiness."""

    async def test_chinese_text_produces_real_audio_without_a_configured_voice(self):
        from voiceagent.audio.pcm import decode_wav
        from voiceagent.providers.tts.macos_say import MacOSSayTTS

        tts = MacOSSayTTS(make_settings(tts={"provider": "macos_say", "voice": None}).tts)
        try:
            chunks = [c async for c in tts.synthesize("帮我记一下，明天上午十点开会。")]
        finally:
            await tts.aclose()
        assert chunks, "no audio produced"
        clip = decode_wav(chunks[0].data)
        assert clip.duration_s > 0.5, (
            "the system default voice silently produced an empty WAV for Chinese"
        )

    def test_chinese_text_selects_a_chinese_voice(self):
        from voiceagent.providers.tts.macos_say import MacOSSayTTS

        tts = MacOSSayTTS(make_settings(tts={"provider": "macos_say", "voice": None}).tts)
        assert tts.pick_voice(None, "明天开会") == "Tingting"
        assert tts.pick_voice("zh") == "Tingting"
        assert tts.pick_voice("en") == "Samantha"
        # Non-Chinese text with no language hint: trust the system default.
        assert tts.pick_voice(None, "hello there") is None

    def test_an_explicit_voice_always_wins(self):
        from voiceagent.providers.tts.macos_say import MacOSSayTTS

        tts = MacOSSayTTS(make_settings(tts={"provider": "macos_say", "voice": "Meijia"}).tts)
        assert tts.pick_voice("en", "hello") == "Meijia"


class TestToolCallParsing:
    def test_parses_valid_json_arguments(self):
        call = ToolCall(id="1", name="f", arguments='{"a": 1}')
        assert call.parsed_args() == {"a": 1}

    def test_malformed_json_degrades_to_empty(self):
        assert ToolCall(id="1", name="f", arguments="{not json").parsed_args() == {}

    def test_non_object_json_degrades_to_empty(self):
        assert ToolCall(id="1", name="f", arguments="[1,2]").parsed_args() == {}

    def test_empty_arguments_are_fine(self):
        assert ToolCall(id="1", name="f", arguments="").parsed_args() == {}


class TestMessageSerialization:
    def test_plain_message(self):
        assert Message(role="user", content="hi").to_openai() == {
            "role": "user",
            "content": "hi",
        }

    def test_assistant_tool_call_message(self):
        payload = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="run_shell", arguments='{"command":"ls"}')],
        ).to_openai()
        assert payload["tool_calls"][0]["function"]["name"] == "run_shell"
        assert "content" not in payload

    def test_tool_result_message(self):
        payload = Message(role="tool", content="ok", tool_call_id="c1", name="run_shell").to_openai()
        assert payload["tool_call_id"] == "c1"
        assert payload["name"] == "run_shell"
