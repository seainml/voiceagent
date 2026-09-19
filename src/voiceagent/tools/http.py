"""Outbound HTTP — the escape hatch for "just call my webhook".

Covers the long tail of daily workflows that do not deserve a bespoke tool:
filing a ticket, posting to a chat channel, hitting an internal API, triggering
a CI job. Restricted to http(s) and optionally to an allowlist of hosts.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import httpx

from .base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    integer,
    object_schema,
    string,
)

MAX_BODY_CHARS = 8000
ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}


async def http_request(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    url = str(args.get("url", "")).strip()
    if not url:
        return ToolResult.failure("no url provided")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ToolResult.failure(f"only http/https is allowed, got {parsed.scheme!r}")

    allow = ctx.settings.tools.http_allow_hosts
    if allow and parsed.hostname not in allow:
        return ToolResult.failure(
            f"host {parsed.hostname!r} is not in tools.http_allow_hosts"
        )

    method = str(args.get("method") or "GET").upper()
    if method not in ALLOWED_METHODS:
        return ToolResult.failure(f"unsupported method {method}")

    headers = {str(k): str(v) for k, v in (args.get("headers") or {}).items()}
    body = args.get("body")
    json_body = args.get("json")
    timeout_s = float(args.get("timeout_s") or 30.0)

    payload: dict[str, Any] = {"headers": headers}
    if json_body is not None:
        payload["json"] = json_body
    elif body is not None:
        payload["content"] = str(body).encode("utf-8")

    await ctx.progress(f"{method} {parsed.netloc}{parsed.path}")
    try:
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            response = await client.request(method, url, **payload)
    except httpx.HTTPError as exc:
        return ToolResult.failure(f"request failed: {exc}", summary="请求失败了。")

    text = response.text[:MAX_BODY_CHARS]
    pretty = text
    if "application/json" in response.headers.get("content-type", ""):
        try:
            pretty = json.dumps(response.json(), ensure_ascii=False, indent=2)[:MAX_BODY_CHARS]
        except json.JSONDecodeError:
            pretty = text

    ok = response.status_code < 400
    body_report = f"HTTP {response.status_code}\n{pretty}"
    meta = {"status": response.status_code, "url": url, "method": method}
    if ok:
        return ToolResult.success(
            body_report, summary=f"请求成功，状态码 {response.status_code}。", **meta
        )
    return ToolResult.failure(
        body_report, summary=f"请求返回了错误状态码 {response.status_code}。", **meta
    )


SPEC = ToolSpec(
    name="http_request",
    description=(
        "Make an HTTP request to any http(s) URL. Use for webhooks, internal "
        "APIs, posting messages to chat platforms, or fetching a JSON endpoint."
    ),
    parameters=object_schema(
        {
            "url": string("Absolute http(s) URL."),
            "method": string("HTTP method.", enum=sorted(ALLOWED_METHODS)),
            "headers": {"type": "object", "description": "Request headers."},
            "json": {"description": "JSON body (object or array)."},
            "body": string("Raw text body, used when `json` is absent."),
            "timeout_s": integer("Timeout in seconds. Default 30."),
        },
        required=["url"],
    ),
    handler=http_request,
)

__all__ = ["SPEC"]
