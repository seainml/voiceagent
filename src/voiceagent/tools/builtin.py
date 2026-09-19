"""Assemble the default tool set from configuration.

Everything the shipped assistant can do is registered here. Adding a capability
means adding a :class:`~voiceagent.tools.base.ToolSpec` to ``BUILTIN`` (or an
MCP server to the config) — the loop, the UI and the confirmation flow need no
changes.
"""

from __future__ import annotations

from ..config import Settings
from .base import ToolSpec
from .registry import ToolRegistry

#: Built-in tools by name. Agent-CLI and MCP tools are added dynamically.
BUILTIN_SPEC_NAMES = (
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
)


def _static_specs() -> dict[str, ToolSpec]:
    from . import files, http, memory, shell

    specs = [
        shell.SPEC,
        http.SPEC,
        files.READ_SPEC,
        files.LIST_SPEC,
        files.WRITE_SPEC,
        memory.REMEMBER_SPEC,
        memory.RECALL_SPEC,
    ]
    return {spec.name: spec for spec in specs}


def _enabled(name: str, settings: Settings) -> bool:
    enabled = settings.tools.enabled
    if not enabled:
        return True
    if name in enabled:
        return True
    # `mcp` acts as a wildcard for every mcp__* tool.
    return name.startswith("mcp__") and "mcp" in enabled


async def build_registry(
    settings: Settings, *, include_mcp: bool = True
) -> ToolRegistry:
    """Build the registry, degrading gracefully when things are missing."""
    from .agent_cli import build_agent_specs

    warnings: list[str] = []
    specs: list[ToolSpec] = list(_static_specs().values())

    try:
        specs.extend(build_agent_specs(settings))
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"agent CLI tools unavailable: {exc}")

    mcp_clients = []
    if include_mcp and settings.tools.mcp:
        from .mcp import connect_servers

        mcp_specs, mcp_clients, mcp_warnings = await connect_servers(settings)
        specs.extend(mcp_specs)
        warnings.extend(mcp_warnings)

    selected = [spec for spec in specs if _enabled(spec.name, settings)]
    if not selected:
        warnings.append("no tools are enabled; check tools.enabled in your config")

    return ToolRegistry(selected, settings, mcp_clients=mcp_clients, warnings=warnings)


__all__ = ["BUILTIN_SPEC_NAMES", "build_registry"]
