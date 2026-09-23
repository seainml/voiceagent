"""Configuration: precedence, secret handling and redaction.

`describe()` is what `voiceagent config` prints and what the browser receives in
the `ready` event, so its contents are a contract: scripts/service.sh reads the
port out of it, and it must never leak a token.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_settings
from voiceagent.config import Settings, config_file_path, load_settings


class TestPrecedence:
    def test_defaults_apply_with_no_configuration(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VA_ASR__PROVIDER", raising=False)
        monkeypatch.delenv("VA_CONFIG", raising=False)
        settings = Settings(_env_file=None)
        assert settings.asr.provider == "auto"
        assert settings.server.port == 8765

    def test_toml_overrides_defaults(self, monkeypatch, tmp_path):
        config = tmp_path / "voiceagent.toml"
        config.write_text(
            '[server]\nport = 9111\n\n[asr]\nprovider = "openai"\nlanguage = "en"\n',
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VA_ASR__PROVIDER", raising=False)
        settings = Settings(_env_file=None)
        assert settings.server.port == 9111
        assert settings.asr.provider == "openai"
        assert settings.asr.language == "en"

    def test_environment_beats_toml(self, monkeypatch, tmp_path):
        (tmp_path / "voiceagent.toml").write_text(
            '[server]\nport = 9111\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VA_SERVER__PORT", "9222")
        settings = Settings(_env_file=None)
        assert settings.server.port == 9222

    def test_explicit_kwargs_beat_environment(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VA_SERVER__PORT", "9222")
        assert Settings(_env_file=None, server={"port": 9333}).server.port == 9333

    def test_nested_env_uses_a_double_underscore(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VA_PIPELINE__VAD_SILENCE_MS", "950")
        monkeypatch.setenv("VA_TTS__VOICE", "Meijia")
        settings = Settings(_env_file=None)
        assert settings.pipeline.vad_silence_ms == 950
        assert settings.tts.voice == "Meijia"

    def test_dotenv_is_read_but_env_still_wins(self, monkeypatch, tmp_path):
        (tmp_path / ".env").write_text(
            "VA_SERVER__PORT=9444\nVA_TTS__VOICE=FromDotenv\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VA_SERVER__PORT", raising=False)
        monkeypatch.delenv("VA_TTS__VOICE", raising=False)
        settings = Settings()
        assert settings.server.port == 9444
        assert settings.tts.voice == "FromDotenv"

        monkeypatch.setenv("VA_TTS__VOICE", "FromEnv")
        assert Settings().tts.voice == "FromEnv"

    def test_unknown_toml_keys_are_ignored(self, monkeypatch, tmp_path):
        (tmp_path / "voiceagent.toml").write_text(
            "[nonsense]\nfoo = 1\n\n[asr]\nalso_unknown = true\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        assert Settings(_env_file=None).asr.provider == "auto"

    def test_va_config_points_at_a_different_file(self, monkeypatch, tmp_path):
        elsewhere = tmp_path / "custom.toml"
        elsewhere.write_text('[server]\nport = 9555\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VA_CONFIG", str(elsewhere))
        assert config_file_path() == elsewhere
        assert Settings(_env_file=None).server.port == 9555


class TestSecrets:
    def test_secret_reads_the_named_variable(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_KEY", "sk-abc")
        assert make_settings().secret("MY_TEST_KEY") == "sk-abc"

    def test_secret_is_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("MY_TEST_KEY", raising=False)
        assert make_settings().secret("MY_TEST_KEY") is None

    def test_secret_is_none_for_an_empty_name(self):
        assert make_settings().secret(None) is None
        assert make_settings().secret("") is None

    def test_secret_is_stripped(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_KEY", "  sk-abc\n")
        assert make_settings().secret("MY_TEST_KEY") == "sk-abc"


class TestDescribe:
    """`voiceagent config` output is a contract; scripts parse it."""

    def test_reports_the_resolved_providers(self):
        payload = make_settings().describe()
        assert payload["asr"]["provider"] == "mock"
        assert payload["tts"]["provider"] == "mock"
        assert payload["llm"]["provider"] == "mock"
        assert payload["agent"]["name"]

    def test_reports_host_and_port_so_scripts_can_find_the_server(self):
        settings = make_settings(server={"host": "0.0.0.0", "port": 9111})
        assert settings.describe()["server"] == {
            "host": "0.0.0.0",
            "port": 9111,
            "auth_required": False,
        }

    def test_auth_token_is_never_included_only_its_presence(self, monkeypatch):
        monkeypatch.setenv("VA_SERVER__AUTH_TOKEN", "super-secret-token")
        settings = Settings(_env_file=None)
        described = settings.describe()
        assert described["server"]["auth_required"] is True
        assert "super-secret-token" not in str(described)

    def test_api_keys_are_never_included(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-not-appear")
        settings = make_settings(llm={"provider": "openai", "model": "m"})
        assert "sk-should-not-appear" not in str(settings.describe())


class TestPaths:
    def test_workspace_is_resolved_against_the_cwd(self, tmp_path):
        settings = make_settings(tools={"workspace": str(tmp_path)})
        assert settings.workspace_path() == tmp_path.resolve()

    def test_relative_workspace_is_expanded(self):
        settings = make_settings(tools={"workspace": "."})
        assert settings.workspace_path() == Path.cwd().resolve()


class TestLoading:
    def test_load_settings_caches(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VA_CONFIG", raising=False)
        first = load_settings(refresh=True)
        assert load_settings() is first

    def test_overrides_bypass_the_cache(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        load_settings(refresh=True)
        assert load_settings(server={"port": 9666}).server.port == 9666


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ("1000", 1000),
        ("250", 250),
    ],
)
def test_integer_env_values_are_coerced(monkeypatch, tmp_path, env, expected):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VA_PIPELINE__VAD_SILENCE_MS", env)
    assert Settings(_env_file=None).pipeline.vad_silence_ms == expected
