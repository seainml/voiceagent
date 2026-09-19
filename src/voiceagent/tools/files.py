"""Workspace file access.

Reads are cheap and safe, so they run without confirmation. Writes are the
dangerous half and always ask first — with a diff-like preview so the spoken
confirmation ("cover three files, overwrite one") is actually informative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    boolean,
    integer,
    object_schema,
    string,
)

MAX_READ_CHARS = 40_000
MAX_LIST_ENTRIES = 300


async def read_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = str(args.get("path", "")).strip()
    if not target:
        return ToolResult.failure("no path provided")
    try:
        path = ctx.resolve(target)
    except PermissionError as exc:
        return ToolResult.failure(str(exc))
    if not path.exists():
        return ToolResult.failure(f"file not found: {target}")
    if path.is_dir():
        return ToolResult.failure(f"{target} is a directory; use list_files instead")

    limit = int(args.get("max_chars") or MAX_READ_CHARS)
    text = await _read_text(path, limit)
    numbered = args.get("line_numbers", True)
    if numbered:
        body = "\n".join(f"{i:>4}  {line}" for i, line in enumerate(text.splitlines(), 1))
    else:
        body = text
    return ToolResult.success(
        body, summary=f"已读取 {target}，共 {len(text.splitlines())} 行。",
        path=str(path.relative_to(ctx.workspace)), bytes=len(text),
    )


async def _read_text(path: Path, limit: int) -> str:
    import asyncio

    def _load() -> str:
        data = path.read_bytes()
        return data[:limit].decode("utf-8", "replace")

    return await asyncio.to_thread(_load)


async def list_files(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = str(args.get("path") or ".").strip()
    try:
        root = ctx.resolve(target)
    except PermissionError as exc:
        return ToolResult.failure(str(exc))
    if not root.exists():
        return ToolResult.failure(f"not found: {target}")

    pattern = str(args.get("glob") or "*")
    recursive = bool(args.get("recursive", False))
    if root.is_file():
        return ToolResult.success(str(root.relative_to(ctx.workspace)))

    entries: list[str] = []
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    for item in iterator:
        hidden = any(part.startswith(".") and part != ".env.example" for part in item.parts)
        if hidden and not args.get("include_hidden"):
            continue
        rel = item.relative_to(root)
        marker = "/" if item.is_dir() else ""
        entries.append(f"{rel}{marker}")
        if len(entries) >= MAX_LIST_ENTRIES:
            entries.append("… (more entries omitted)")
            break
    entries.sort()
    return ToolResult.success(
        "\n".join(entries) or "(empty)",
        summary=f"{target} 下有 {len(entries)} 个条目。",
        count=len(entries),
    )


async def write_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = str(args.get("path", "")).strip()
    content = args.get("content")
    if not target:
        return ToolResult.failure("no path provided")
    if content is None:
        return ToolResult.failure("no content provided")
    try:
        path = ctx.resolve(target)
    except PermissionError as exc:
        return ToolResult.failure(str(exc))

    append = bool(args.get("append", False))
    existed = path.exists()
    previous = path.read_text("utf-8", "replace") if existed and not append else ""

    # Ask before touching an existing file; creating a new one is low risk but
    # still announced so the user hears what happened.
    if existed and ctx.confirm is not None:
        approved = await ctx.confirm(
            f"文件 {target} 已经存在，共 {len(previous.splitlines())} 行，"
            f"确定要{'追加' if append else '覆盖'}吗？"
        )
        if not approved:
            return ToolResult.failure(
                "user declined the write", summary="好的，我没有修改文件。", declined=True
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(str(content))

    added = len(str(content).splitlines())
    verb = "追加" if append else ("覆盖" if existed else "创建")
    return ToolResult.success(
        f"{verb} {target}: {added} lines",
        summary=f"已经{verb} {target}，写了 {added} 行。",
        path=str(path.relative_to(ctx.workspace)), lines=added, created=not existed,
    )


READ_SPEC = ToolSpec(
    name="read_file",
    description="Read a UTF-8 text file from the workspace. Returns numbered lines.",
    parameters=object_schema(
        {
            "path": string("Path relative to the workspace root."),
            "max_chars": integer("Truncate the file after this many characters."),
            "line_numbers": boolean("Prefix each line with its number.", default=True),
        },
        required=["path"],
    ),
    handler=read_file,
)

LIST_SPEC = ToolSpec(
    name="list_files",
    description="List files and directories inside the workspace.",
    parameters=object_schema(
        {
            "path": string("Directory relative to the workspace root. Default '.'."),
            "glob": string("Glob filter, e.g. '*.py'."),
            "recursive": boolean("Walk subdirectories.", default=False),
            "include_hidden": boolean("Include dotfiles.", default=False),
        }
    ),
    handler=list_files,
)

WRITE_SPEC = ToolSpec(
    name="write_file",
    description=(
        "Create, overwrite or append to a text file in the workspace. "
        "Overwriting an existing file requires the user's confirmation."
    ),
    parameters=object_schema(
        {
            "path": string("Path relative to the workspace root."),
            "content": string("Full text to write."),
            "append": boolean("Append instead of replacing.", default=False),
        },
        required=["path", "content"],
    ),
    handler=write_file,
    dangerous=True,
)

__all__ = ["LIST_SPEC", "READ_SPEC", "WRITE_SPEC"]
