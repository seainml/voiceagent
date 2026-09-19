"""Configuration.

Precedence (highest first):

    explicit kwargs  >  environment (``VA_*``)  >  ``.env``  >  ``voiceagent.toml``  >  defaults

Nested keys use a double underscore in the environment, e.g.::

    VA_ASR__PROVIDER=openai
    VA_TTS__VOICE=zh-CN-XiaoxiaoNeural
    VA_LLM__MODEL=deepseek-chat

Secrets are never stored in the TOML file; only the *name* of the environment
variable that holds them (``api_key_env``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

try:  # pydantic-settings >= 2.2
    from pydantic_settings import TomlConfigSettingsSource
except ImportError:  # pragma: no cover - older pydantic-settings
    TomlConfigSettingsSource = None  # type: ignore[assignment, misc]

DEFAULT_CONFIG_FILE = "voiceagent.toml"

#: Prepended to every wrapped-agent task. These CLIs answer in Markdown, which
#: sounds terrible through a speaker, so we ask for a spoken summary up front.
SPOKEN_SUMMARY_PREAMBLE = (
    "You are being invoked by a voice assistant, so your final answer will be "
    "read aloud. Do the task autonomously and completely. When you are done, "
    "end with a spoken summary of at most three short sentences in the user's "
    "language: no Markdown, no code blocks, no bullet lists, no file paths "
    "unless essential."
)


def config_file_path() -> Path:
    """Resolve the TOML config path (``VA_CONFIG`` overrides the default name)."""
    return Path(os.environ.get("VA_CONFIG") or DEFAULT_CONFIG_FILE).expanduser()


class ASRSettings(BaseModel):
    """Speech recognition.

    ``provider`` accepts ``auto``, ``faster_whisper`` (local, offline),
    ``openai`` (any OpenAI-compatible ``/audio/transcriptions`` endpoint),
    ``browser`` (recognition happens in the browser, the server just receives
    text) or ``mock`` (tests).
    """

    provider: str = "auto"
    model: str = "base"  # faster-whisper size, or the remote model id
    language: str | None = "zh"
    hotwords: list[str] = Field(default_factory=list)
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    vad_filter: bool = True
    # Local model download directory (kept inside the workspace by default).
    download_root: str | None = ".voiceagent/models"
    timeout_s: float = 60.0


class TTSSettings(BaseModel):
    """Speech synthesis.

    ``provider`` accepts ``auto``, ``macos_say`` (zero-dependency, macOS),
    ``openai`` (OpenAI-compatible ``/audio/speech``), ``edge`` (edge-tts) or
    ``mock``.
    """

    provider: str = "auto"
    voice: str | None = None  # None => pick a sensible default per language
    model: str = "tts-1"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    response_format: str = "mp3"  # mp3 | wav | pcm | opus
    rate: int = 195  # words per minute, used by macos_say
    pitch: int = 0  # used by edge-tts
    volume: int = 0  # used by edge-tts
    timeout_s: float = 60.0
    # Playback speed multiplier applied to the generated audio (1.0 = unchanged).
    speed: float = 1.0


class LLMSettings(BaseModel):
    """The reasoning model. Any OpenAI-compatible chat endpoint works."""

    provider: str = "auto"  # auto | openai | mock
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/v1"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.5
    max_tokens: int = 1200
    timeout_s: float = 120.0
    # Extra headers / body fields for gateways that need them.
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class AgentCLISettings(BaseModel):
    """One external coding agent exposed to the voice layer as a tool."""

    enabled: bool = True
    command: str
    args: list[str] = Field(default_factory=list)
    timeout_s: float = 900.0
    env: dict[str, str] = Field(default_factory=dict)
    working_dir: str | None = None
    description: str = ""
    #: Prepended to the user's task. Defaults ask for a short *spoken* summary,
    #: because these agents otherwise return Markdown that is awful read aloud.
    preamble: str = ""
    #: Appended only when ``tools.allow_unsafe_agent_writes`` is enabled —
    #: e.g. ``--dangerously-skip-permissions`` so the agent can edit files.
    unsafe_args: list[str] = Field(default_factory=list)


class ToolsSettings(BaseModel):
    workspace: str = "."
    enabled: list[str] = Field(
        default_factory=lambda: [
            "ask_claude_code",
            "ask_codex",
            "ask_dsh",
            "run_shell",
            "http_request",
            "read_file",
            "write_file",
            "list_files",
            "remember",
            "recall",
            "mcp",
        ]
    )
    #: Tools that must be confirmed out loud before they run.
    confirm: list[str] = Field(
        default_factory=lambda: [
            "run_shell",
            "write_file",
            "ask_claude_code",
            "ask_codex",
            "ask_dsh",
        ]
    )
    max_output_chars: int = 4000
    shell_timeout_s: float = 120.0
    shell_denylist: list[str] = Field(
        default_factory=lambda: [
            "rm -rf /",
            "mkfs",
            "dd if=",
            ":(){:|:&};:",
            "shutdown",
            "reboot",
            "> /dev/",
        ]
    )
    http_allow_hosts: list[str] = Field(default_factory=list)  # empty => any host
    memory_file: str = ".voiceagent/memory.jsonl"
    #: Opt-in: let the wrapped agent CLIs bypass their own permission prompts.
    #: Off by default because it is exactly as dangerous as it sounds.
    allow_unsafe_agent_writes: bool = False
    agents: dict[str, AgentCLISettings] = Field(
        default_factory=lambda: {
            "claude": AgentCLISettings(
                command="claude",
                args=["--print", "--output-format", "stream-json", "--verbose"],
                unsafe_args=["--dangerously-skip-permissions"],
                preamble=SPOKEN_SUMMARY_PREAMBLE,
                description="Claude Code, Anthropic's terminal coding agent.",
            ),
            "codex": AgentCLISettings(
                command="codex",
                args=["exec", "--color", "never", "--skip-git-repo-check", "--json"],
                unsafe_args=["--dangerously-bypass-approvals-and-sandbox"],
                preamble=SPOKEN_SUMMARY_PREAMBLE,
                description="OpenAI Codex CLI, running non-interactively.",
            ),
            "dsh": AgentCLISettings(
                command="dsh",
                args=["--profile", "headless"],
                preamble=SPOKEN_SUMMARY_PREAMBLE,
                description="DeepSeek Harness, answering one task and exiting.",
            ),
        }
    )
    #: MCP servers to expose as tools, keyed by server name::
    #:
    #:     [tools.mcp.filesystem]
    #:     command = "npx"
    #:     args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    mcp: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PipelineSettings(BaseModel):
    barge_in: bool = True
    #: Flush a partial sentence to TTS once it reaches this many characters and
    #: ends on a natural break — keeps time-to-first-audio low.
    sentence_flush_chars: int = 28
    max_sentence_chars: int = 180
    tts_concurrency: int = 2
    asr_min_utterance_ms: int = 250
    #: Trailing silence that closes a turn. There is a real tension here:
    #: Full-Duplex-Bench showed dialogue models systematically interrupt people
    #: during *thinking* pauses, which argues for waiting longer; latency work
    #: argues for cutting sooner. 600 ms is the compromise, and it is the single
    #: most worthwhile knob to tune per user and per microphone.
    vad_silence_ms: int = 600
    vad_threshold: float = 0.015
    max_utterance_s: float = 60.0
    #: Hard ceiling on how much of one reply is spoken aloud. Anything longer
    #: stays on screen — nobody wants a 900-word answer through a speaker.
    max_spoken_chars: int = 700
    #: Keep this many turns of history in the prompt.
    history_turns: int = 12


class AgentSettings(BaseModel):
    name: str = "Aria"
    persona: str = (
        "You are a calm, efficient office assistant. You are heard, not read, "
        "so you speak in short, natural sentences."
    )
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    max_tool_iterations: int = 6


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    #: When set, clients must pass ``?token=`` on the WebSocket URL.
    auth_token: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(
        env_prefix="VA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    asr: ASRSettings = Field(default_factory=ASRSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Put the TOML file *below* the environment so env vars always win."""
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if TomlConfigSettingsSource is not None:
            path = config_file_path()
            if path.is_file():
                sources.append(TomlConfigSettingsSource(settings_cls, toml_file=path))
        sources.append(file_secret_settings)
        return tuple(sources)

    # -- helpers ---------------------------------------------------------

    def secret(self, env_name: str | None) -> str | None:
        """Read a secret from the environment (never from the config file)."""
        if not env_name:
            return None
        value = os.environ.get(env_name)
        if value:
            return value.strip()
        # Allow the common alias used by many gateways.
        return None

    def workspace_path(self) -> Path:
        return Path(self.tools.workspace).expanduser().resolve()

    def describe(self) -> dict[str, Any]:
        """A redacted snapshot, safe to log or to send to the browser."""
        return {
            "asr": {"provider": self.asr.provider, "model": self.asr.model,
                    "language": self.asr.language},
            "tts": {"provider": self.tts.provider, "voice": self.tts.voice},
            "llm": {"provider": self.llm.provider, "model": self.llm.model,
                    "base_url": self.llm.base_url},
            "tools": {"enabled": list(self.tools.enabled), "confirm": list(self.tools.confirm)},
            "agent": {"name": self.agent.name},
        }


_CACHED: Settings | None = None


def load_settings(refresh: bool = False, **overrides: Any) -> Settings:
    """Load settings, caching the result for the process lifetime."""
    global _CACHED
    if _CACHED is None or refresh or overrides:
        settings = Settings(**overrides)
        if overrides:
            return settings
        _CACHED = settings
    return _CACHED
