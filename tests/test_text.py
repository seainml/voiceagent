"""Speech-oriented text handling."""

from __future__ import annotations

from voiceagent.text import (
    SentenceSplitter,
    sanitize_for_speech,
    sentence_spans,
    truncate_for_speech,
)


class TestSentenceSplitter:
    def test_splits_on_hard_stops(self):
        splitter = SentenceSplitter(flush_chars=100, max_chars=200)
        out = splitter.push("你好。今天天气不错！要不要出去走走？")
        assert out == ["你好。", "今天天气不错！", "要不要出去走走？"]
        assert splitter.flush() is None

    def test_streaming_across_deltas(self):
        splitter = SentenceSplitter(flush_chars=6, max_chars=60)
        emitted = []
        for delta in ["今天", "的会议", "推迟到", "下午三点", "。", "记得", "通知大家", "。"]:
            emitted.extend(splitter.push(delta))
        assert emitted == ["今天的会议推迟到下午三点。", "记得通知大家。"]
        assert splitter.flush() is None

    def test_flush_returns_unterminated_tail(self):
        splitter = SentenceSplitter(flush_chars=100)
        assert splitter.push("这句话没有标点") == []
        assert splitter.flush() == "这句话没有标点"

    def test_english_sentences(self):
        parts = sentence_spans("First one. Second one! Third?")
        assert parts == ["First one.", "Second one!", "Third?"]

    def test_a_trailing_period_only_counts_at_the_end(self):
        splitter = SentenceSplitter(flush_chars=1000)
        assert splitter.push("It works.") == []  # could still become "works.py"
        assert splitter.flush() == "It works."

    def test_keeps_closing_quotes_with_the_sentence(self):
        parts = sentence_spans("他说：「好。」然后走了。")
        assert parts[0].endswith("」")
        assert parts[-1] == "然后走了。"

    def test_hard_cut_of_a_runaway_sentence(self):
        splitter = SentenceSplitter(flush_chars=10_000, max_chars=30)
        out = splitter.push("啊" * 100)
        # One push drains every complete segment it can, then holds the tail.
        assert [len(part) for part in out] == [30, 30, 30]
        assert splitter.flush() == "啊" * 10

    def test_does_not_split_inside_a_filename_or_a_decimal(self):
        splitter = SentenceSplitter(flush_chars=1, max_chars=1000)
        out = splitter.push("打开 voiceagent.toml 把阈值改成 3.14 就行了。")
        assert out == ["打开 voiceagent.toml 把阈值改成 3.14 就行了。"]

    def test_punctuation_only_is_never_emitted(self):
        splitter = SentenceSplitter(flush_chars=1, max_chars=10)
        assert splitter.push("。。。") == []
        assert splitter.flush() is None

    def test_soft_stop_flush_keeps_time_to_first_audio_low(self):
        splitter = SentenceSplitter(flush_chars=8, max_chars=200)
        out = splitter.push("首先我们需要确认一下，")
        assert out == ["首先我们需要确认一下，"]


class TestSanitizeForSpeech:
    def test_strips_markdown(self):
        text = "## 标题\n\n**重点**：`code` 和 [链接](http://x.com)"
        cleaned = sanitize_for_speech(text)
        assert "*" not in cleaned
        assert "`" not in cleaned
        assert "#" not in cleaned
        assert "链接" in cleaned
        assert "重点" in cleaned

    def test_replaces_code_fences_with_a_spoken_note(self):
        cleaned = sanitize_for_speech("这是代码：\n```python\nprint(1)\n```\n完成。")
        assert "print(1)" not in cleaned
        assert "代码块" in cleaned

    def test_collapses_bare_urls(self):
        assert "http" not in sanitize_for_speech("见 https://example.com/a/b?c=1 这里")

    def test_removes_emoji(self):
        assert "🎉" not in sanitize_for_speech("完成了🎉")

    def test_strips_bullets_and_headings(self):
        cleaned = sanitize_for_speech("- 一\n- 二\n# 三")
        assert cleaned.splitlines() == ["一", "二", "三"]


class TestTruncate:
    def test_short_text_untouched(self):
        assert truncate_for_speech("短句。", 100) == "短句。"

    def test_cuts_at_a_sentence_boundary(self):
        text = "第一句。第二句。第三句。" + "第四句很长。" * 20
        out = truncate_for_speech(text, 30)
        assert len(out) < len(text)
        assert out.startswith("第一句。")
        assert out.endswith("更多细节在屏幕上。")

    def test_suffix_is_dropped_when_it_would_not_shorten_anything(self):
        # 12 chars of text against a 12-char budget: a suffix would make the
        # "summary" longer than what it replaces.
        text = "第一句。第二句。第三句。第四句。"
        out = truncate_for_speech(text, 12)
        assert len(out) < len(text)
        assert "更多细节" not in out

    def test_cuts_mid_clause_when_there_is_no_boundary(self):
        out = truncate_for_speech("啊" * 500, 50)
        assert len(out) < 500
        assert out.endswith("更多细节在屏幕上。")
