"""The tool layer: guardrails, built-ins, agent-CLI parsers and MCP naming."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import make_settings, tool_call
from voiceagent.tools.agent_cli import (
    ClaudeStreamParser,
    CodexJSONParser,
    PlainTextParser,
    build_agent_specs,
)
from voiceagent.tools.base import ToolContext
from voiceagent.tools.exec import ExecResult
from voiceagent.tools.files import LIST_SPEC, READ_SPEC, WRITE_SPEC
from voiceagent.tools.mcp import looks_dangerous
from voiceagent.tools.memory import RECALL_SPEC, REMEMBER_SPEC, load_entries
from voiceagent.tools.shell import run_shell


@pytest.fixture
def ctx(tmp_path):
    settings = make_settings(tools={"workspace": str(tmp_path)})
    return ToolContext(workspace=tmp_path, settings=settings, session_id="test")


class TestToolRegistry:
    async def test_unknown_tool_returns_a_helpful_failure(self, registry):
        result = await registry.call(
            tool_call("no_such_tool"), ToolContext(workspace=Path("."), settings=make_settings())
        )
        assert not result.ok
        assert "unknown tool" in result.content
        assert "run_shell" in result.content

    async def test_exception_is_caught_and_reported(self, registry, ctx):
        result = await registry.call(tool_call("read_file", '{"path":"nope.txt"}'), ctx)
        assert not result.ok
        assert "not found" in result.content

    async def test_large_output_is_truncated(self, registry, ctx, tmp_path):
        big = tmp_path / "big.txt"
        big.write_text("x" * 50_000, encoding="utf-8")
        registry.settings.tools.max_output_chars = 1000
        result = await registry.call(tool_call("read_file", '{"path":"big.txt"}'), ctx)
        assert result.truncated
        assert len(result.content) < 2000

    async def test_dangerous_set_is_config_plus_declared(self, registry):
        dangerous = registry.dangerous()
        assert "run_shell" in dangerous          # from settings.tools.confirm
        assert "ask_claude_code" in dangerous    # from settings.tools.confirm
        assert "read_file" not in dangerous

    async def test_openai_schema_has_the_right_envelope(self, registry):
        schema = registry.openai_schema()
        first = schema[0]
        assert first["type"] == "function"
        assert set(first["function"]) == {"name", "description", "parameters"}
        assert first["function"]["parameters"]["type"] == "object"


class TestShell:
    async def test_denylist_blocks_a_dangerous_command(self, ctx):
        result = await run_shell({"command": "rm -rf /"}, ctx)
        assert not result.ok
        assert "denylist" in result.content

    async def test_denylist_catches_a_variation(self, ctx):
        assert not (await run_shell({"command": "sudo reboot"}, ctx)).ok

    async def test_runs_a_harmless_command(self, ctx):
        result = await run_shell({"command": "echo hello"}, ctx)
        assert result.ok
        assert "hello" in result.content

    async def test_reports_a_nonzero_exit(self, ctx):
        result = await run_shell({"command": "exit 3"}, ctx)
        assert not result.ok
        assert "3" in result.content

    async def test_times_out(self, ctx):
        result = await run_shell({"command": "sleep 5", "timeout_s": 1}, ctx)
        assert not result.ok
        assert "timed out" in result.content.lower() or "超时" in result.summary


class TestFiles:
    async def test_write_then_read_round_trip(self, ctx):
        written = await WRITE_SPEC.handler({"path": "notes/a.txt", "content": "第一行\n第二行"}, ctx)
        assert written.ok
        read = await READ_SPEC.handler({"path": "notes/a.txt"}, ctx)
        assert read.ok
        assert "第一行" in read.content
        assert "1  " in read.content  # numbered

    async def test_append_mode_keeps_previous_content(self, ctx, tmp_path):
        await WRITE_SPEC.handler({"path": "a.txt", "content": "one\n"}, ctx)
        await WRITE_SPEC.handler({"path": "a.txt", "content": "two\n", "append": True}, ctx)
        assert (tmp_path / "a.txt").read_text() == "one\ntwo\n"

    async def test_path_traversal_is_refused(self, ctx):
        result = await READ_SPEC.handler({"path": "../../etc/passwd"}, ctx)
        assert not result.ok
        assert "workspace" in result.content

    async def test_reading_a_directory_is_refused(self, ctx):
        result = await READ_SPEC.handler({"path": "."}, ctx)
        assert not result.ok

    async def test_listing_is_sorted_and_marks_directories(self, ctx, tmp_path):
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("a")
        result = await LIST_SPEC.handler({}, ctx)
        assert result.ok
        lines = result.content.splitlines()
        assert lines[0] == "a.txt"
        assert "sub/" in lines

    async def test_overwriting_asks_and_can_be_declined(self, ctx, tmp_path):
        target = tmp_path / "existing.txt"
        target.write_text("original")

        async def say_no(_question: str) -> bool:
            return False

        ctx.confirm = say_no
        result = await WRITE_SPEC.handler({"path": "existing.txt", "content": "new"}, ctx)
        assert not result.ok
        assert result.meta.get("declined")
        assert target.read_text() == "original"

    async def test_overwriting_proceeds_when_confirmed(self, ctx, tmp_path):
        target = tmp_path / "existing.txt"
        target.write_text("original")

        asked: list[str] = []

        async def say_yes(question: str) -> bool:
            asked.append(question)
            return True

        ctx.confirm = say_yes
        result = await WRITE_SPEC.handler({"path": "existing.txt", "content": "new"}, ctx)
        assert result.ok
        assert target.read_text() == "new"
        assert asked and "existing.txt" in asked[0]


class TestMemory:
    async def test_remember_then_recall(self, ctx):
        assert (await REMEMBER_SPEC.handler({"note": "张三负责后端服务"}, ctx)).ok
        assert (await REMEMBER_SPEC.handler({"note": "周五下午开周会"}, ctx)).ok
        result = await RECALL_SPEC.handler({"query": "张三"}, ctx)
        assert result.ok
        assert "张三" in result.content

    async def test_recall_without_a_query_returns_recent_notes(self, ctx):
        for note in ("第一条", "第二条", "第三条"):
            await REMEMBER_SPEC.handler({"note": note}, ctx)
        result = await RECALL_SPEC.handler({"limit": 2}, ctx)
        assert "第三条" in result.content
        assert "第一条" not in result.content

    async def test_recall_on_empty_memory_is_not_an_error(self, ctx):
        result = await RECALL_SPEC.handler({"query": "任何"}, ctx)
        assert result.ok
        assert result.meta.get("matches") == 0

    async def test_notes_survive_a_reload(self, ctx):
        await REMEMBER_SPEC.handler({"note": "持久化测试", "tags": ["demo"]}, ctx)
        entries = load_entries(ctx)
        assert entries[-1]["note"] == "持久化测试"
        assert entries[-1]["tags"] == ["demo"]

    async def test_malformed_lines_are_skipped(self, ctx, tmp_path):
        path = tmp_path / ".voiceagent" / "memory.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"note":"ok","ts":1}\nnot json\n', encoding="utf-8")
        entries = load_entries(ctx)
        assert len(entries) == 1


class TestClaudeParser:
    def test_extracts_the_result_event(self):
        parser = ClaudeStreamParser()
        lines = [
            json.dumps({"type": "system", "model": "claude-opus-4"}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "好的，我来处理。"}]}}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Bash"}]}}),
            json.dumps({"type": "result", "subtype": "success", "is_error": False,
                        "result": "改好了。", "total_cost_usd": 0.02,
                        "duration_ms": 1234}),
        ]
        notes = [n for n in (parser.feed("stdout", line) for line in lines) if n]
        ok, text, meta = parser.finish(ExecResult(returncode=0))
        assert ok
        assert text == "改好了。"
        assert meta["cost_usd"] == 0.02
        assert meta["tools_used"] == ["Bash"]
        assert any("Bash" in n for n in notes)

    def test_is_error_result_is_a_failure(self):
        parser = ClaudeStreamParser()
        parser.feed("stdout", json.dumps({
            "type": "result", "is_error": True, "result": "Not logged in · Please run /login"}))
        ok, text, _ = parser.finish(ExecResult(returncode=1))
        assert not ok
        assert "Not logged in" in text

    def test_falls_back_to_accumulated_text(self):
        parser = ClaudeStreamParser()
        parser.feed("stdout", json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "只有这段"}]}}))
        ok, text, _ = parser.finish(ExecResult(returncode=0))
        assert ok and text == "只有这段"

    def test_non_json_lines_become_progress(self):
        parser = ClaudeStreamParser()
        assert parser.feed("stderr", "warming up") == "warming up"


class TestCodexParser:
    def test_collects_agent_messages(self):
        parser = CodexJSONParser()
        parser.feed("stdout", json.dumps({"type": "thread.started", "thread_id": "t"}))
        parser.feed("stdout", json.dumps({"type": "item.completed", "item": {
            "type": "command_execution", "command": "pytest -q"}}))
        parser.feed("stdout", json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": "测试通过了。"}}))
        parser.feed("stdout", json.dumps({"type": "turn.completed",
                                          "usage": {"output_tokens": 12}}))
        ok, text, meta = parser.finish(ExecResult(returncode=0))
        assert ok
        assert text == "测试通过了。"
        assert meta["usage"]["output_tokens"] == 12

    def test_error_event_is_surfaced(self):
        parser = CodexJSONParser()
        parser.feed("stdout", json.dumps({"type": "turn.failed", "message": "boom"}))
        ok, text, _ = parser.finish(ExecResult(returncode=1))
        assert not ok
        assert "boom" in text

    def test_plain_text_survives_a_format_change(self):
        parser = CodexJSONParser()
        parser.feed("stdout", "the CLI printed plain text")
        ok, text, _ = parser.finish(ExecResult(returncode=0))
        assert ok and "plain text" in text


class TestPlainTextParser:
    def test_joins_stdout_lines(self):
        parser = PlainTextParser()
        parser.feed("stdout", "pong")
        ok, text, _ = parser.finish(ExecResult(returncode=0))
        assert ok and text == "pong"

    def test_nonzero_exit_is_a_failure(self):
        parser = PlainTextParser()
        parser.feed("stdout", "oops")
        ok, _, _ = parser.finish(ExecResult(returncode=1))
        assert not ok


class TestAgentSpecs:
    def test_three_agent_tools_are_built(self):
        names = {spec.name for spec in build_agent_specs(make_settings())}
        assert names == {"ask_claude_code", "ask_codex", "ask_dsh"}

    def test_disabled_agents_are_dropped(self):
        settings = make_settings()
        settings.tools.agents["codex"].enabled = False
        names = {spec.name for spec in build_agent_specs(settings)}
        assert "ask_codex" not in names

    def test_agent_tools_require_confirmation(self):
        assert all(spec.dangerous for spec in build_agent_specs(make_settings()))


class TestMcpHeuristics:
    @pytest.mark.parametrize("name", ["write_file", "delete_issue", "exec_command", "send_message"])
    def test_write_like_names_look_dangerous(self, name):
        assert looks_dangerous(name)

    @pytest.mark.parametrize("name", ["read_file", "list_issues", "get_weather", "search"])
    def test_read_like_names_look_safe(self, name):
        assert not looks_dangerous(name)

    def test_explicit_annotations_win(self):
        assert looks_dangerous("read_file", {"destructiveHint": True})
        assert not looks_dangerous("write_file", {"readOnlyHint": True})
