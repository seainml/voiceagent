"""Long-term memory.

A voice assistant that forgets your colleague's name between sentences is
tiring. This is deliberately the simplest thing that works: an append-only
JSONL file, retrieved by token overlap. No embeddings, no database — it stays
useful up to a few thousand entries, which is far more than personal office
notes ever reach.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    integer,
    object_schema,
    string,
)

MAX_ENTRIES = 20_000
_TOKEN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text)}


def _memory_path(ctx: ToolContext) -> Path:
    path = Path(ctx.settings.tools.memory_file)
    if not path.is_absolute():
        path = ctx.workspace / path
    return path


def load_entries(ctx: ToolContext) -> list[dict[str, Any]]:
    path = _memory_path(ctx)
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            entries.append(item)
    return entries[-MAX_ENTRIES:]


async def remember(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    note = str(args.get("note") or "").strip()
    if not note:
        return ToolResult.failure("nothing to remember")
    tags = args.get("tags") or []
    entry = {
        "ts": time.time(),
        "note": note,
        "tags": [str(t) for t in tags] if isinstance(tags, list) else [],
        "session": ctx.session_id,
    }

    path = _memory_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return ToolResult.success(
        f"remembered: {note}",
        summary="好的，我记住了。",
        total=len(load_entries(ctx)),
    )


async def recall(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(args.get("query") or "").strip()
    limit = int(args.get("limit") or 5)
    entries = load_entries(ctx)
    if not entries:
        return ToolResult.success(
            "(no memories stored yet)",
            summary="我还没有记住任何相关内容。",
            matches=0,
        )

    if not query:
        chosen = entries[-limit:]
        score_note = "most recent"
    else:
        wanted = _tokens(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in entries:
            haystack = _tokens(str(entry.get("note", "")) + " " + " ".join(
                str(t) for t in entry.get("tags", [])
            ))
            if not haystack:
                continue
            overlap = len(wanted & haystack)
            if overlap:
                # Favour overlap, then recency.
                scored.append((overlap + entry.get("ts", 0) / 1e12, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        chosen = [entry for _score, entry in scored[:limit]]
        score_note = f"matched {len(scored)} entries"

    if not chosen:
        return ToolResult.success(
            "(nothing matched)", summary=f"我没有找到和{query}相关的记录。", matches=0
        )

    lines = [
        f"- {time.strftime('%Y-%m-%d', time.localtime(e.get('ts', 0)))}: {e.get('note', '')}"
        for e in chosen
    ]
    body = f"# memories ({score_note})\n" + "\n".join(lines)
    return ToolResult.success(
        body,
        summary=f"找到 {len(chosen)} 条相关记录，最近的一条是：{chosen[-1].get('note', '')[:80]}",
        matches=len(chosen),
    )


REMEMBER_SPEC = ToolSpec(
    name="remember",
    description=(
        "Store a durable note that survives across sessions: a preference, a "
        "name, a project detail, a recurring task. Use it proactively when the "
        "user tells you something worth keeping."
    ),
    parameters=object_schema(
        {
            "note": string("The fact to remember, written as a standalone sentence."),
            "tags": {"type": "array", "items": {"type": "string"},
                     "description": "Optional tags for retrieval."},
        },
        required=["note"],
    ),
    handler=remember,
)

RECALL_SPEC = ToolSpec(
    name="recall",
    description="Search durable notes stored earlier. Omit the query to list the most recent.",
    parameters=object_schema(
        {
            "query": string("What to look for. Empty returns the most recent notes."),
            "limit": integer("Maximum entries to return. Default 5."),
        }
    ),
    handler=recall,
)

__all__ = ["RECALL_SPEC", "REMEMBER_SPEC", "load_entries"]
