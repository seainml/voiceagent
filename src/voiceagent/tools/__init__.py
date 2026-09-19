"""The capability layer: what the assistant can actually *do*."""

from __future__ import annotations

from .base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    array,
    boolean,
    integer,
    object_schema,
    string,
)
from .builtin import BUILTIN_SPEC_NAMES, build_registry
from .exec import ExecResult, format_command, run_streaming
from .registry import ToolRegistry

__all__ = [
    "BUILTIN_SPEC_NAMES",
    "ExecResult",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "array",
    "boolean",
    "build_registry",
    "format_command",
    "integer",
    "object_schema",
    "run_streaming",
    "string",
]
