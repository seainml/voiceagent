"""Text handling tuned for *speech* rather than for reading.

Two jobs live here:

* :class:`SentenceSplitter` — incremental sentence segmentation so TTS can
  start on sentence one while the LLM is still writing sentence four. This is
  the single biggest latency win in the whole pipeline.
* :func:`sanitize_for_speech` — LLMs emit Markdown, URLs, emoji and code
  fences. None of that should ever reach a speaker.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

#: Characters that reliably end a sentence in Chinese and English.
#: ASCII "." is handled separately — see ``_is_period_terminator``.
HARD_STOPS = "。！？!?…\n\r"
#: Characters we will cut on only when a segment has grown long enough.
SOFT_STOPS = "，,、；;：: "
#: Punctuation that may legitimately follow a terminator ("他说：「好。」").
_TRAILING = "”’」』）)】》\"'"

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BARE_URL = re.compile(r"https?://\S+|www\.\S+")
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s{0,3}([-*+]|\d+[.)])\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_HR = re.compile(r"^\s{0,3}([-*_])\s*(\1\s*){2,}$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\u2190-\u21ff\u2b00-\u2bff]"
)
_WS = re.compile(r"[ \t\u00a0]+")


def sanitize_for_speech(text: str, *, keep_code_summary: bool = True) -> str:
    """Strip anything a listener cannot use."""
    if not text:
        return ""
    out = text
    out = _FENCED_CODE.sub(
        "\n代码块，请看屏幕。\n" if keep_code_summary else "\n", out
    )
    out = _MD_IMAGE.sub("", out)
    out = _MD_LINK.sub(r"\1", out)
    out = _BARE_URL.sub("链接", out)
    out = _TABLE_ROW.sub("", out)
    out = _HR.sub("", out)
    out = _HEADING.sub("", out)
    out = _BLOCKQUOTE.sub("", out)
    out = _BULLET.sub("", out)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _BOLD_ITALIC.sub(r"\2", out)
    out = _EMOJI.sub("", out)
    out = _WS.sub(" ", out)
    out = re.sub(r"\n{2,}", "\n", out)
    return out.strip()


def truncate_for_speech(text: str, max_chars: int, suffix: str = "，更多细节在屏幕上。") -> str:
    """Shorten an over-long reply at a sentence boundary, preserving meaning.

    The suffix is dropped when adding it would make the result no shorter than
    the original — a "summary" that is longer than what it replaces is worse
    than useless.
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = max(window.rfind(ch) for ch in HARD_STOPS + SOFT_STOPS)
    if cut < max_chars * 0.4:
        cut = max_chars - 1
    prefix = window[: cut + 1].rstrip()
    if not prefix:
        return window
    if len(prefix) + len(suffix) >= len(text):
        return prefix
    return prefix + suffix


def sentence_spans(text: str) -> list[str]:
    """Split a finished string into speakable sentences."""
    splitter = SentenceSplitter(flush_chars=1, max_chars=10_000, min_chars=1)
    parts = splitter.push(text)
    tail = splitter.flush()
    if tail:
        parts.append(tail)
    return parts


@dataclass(slots=True)
class SentenceSplitter:
    """Incremental, streaming sentence segmenter.

    ``flush_chars`` controls the latency/quality trade-off: a partial sentence
    is emitted early only once it is at least this long *and* ends on a soft
    stop, which sounds natural instead of chopped.
    """

    flush_chars: int = 28
    max_chars: int = 180
    min_chars: int = 2
    _buf: str = field(default="", init=False)

    def push(self, delta: str) -> list[str]:
        """Feed an LLM delta; return zero or more complete segments to speak."""
        if not delta:
            return []
        self._buf += delta
        return self._drain(final=False)

    def flush(self) -> str | None:
        """Return whatever is left over — call this when the stream ends.

        :meth:`push` already drains every complete sentence, so all that can
        remain is an unterminated tail (or trailing punctuation to discard).
        """
        remainder = self._buf.strip()
        self._buf = ""
        return remainder if self._speakable(remainder) else None

    # -- internals -------------------------------------------------------

    def _drain(self, *, final: bool) -> list[str]:
        out: list[str] = []
        while True:
            segment = self._next_segment(final=final)
            if segment is None:
                break
            cleaned = segment.strip()
            if self._speakable(cleaned):
                out.append(cleaned)
            elif cleaned and final:
                pass  # pure punctuation: drop silently
        return out

    def _next_segment(self, *, final: bool) -> str | None:
        buf = self._buf
        if not buf:
            return None

        # 1) a hard terminator, past the minimum length
        for i, ch in enumerate(buf):
            if ch in HARD_STOPS:
                end = self._sentence_end(buf, i)
                if self._speakable(buf[:end]) or final:
                    return self._take(end)
            elif ch == "." and self._is_period_terminator(buf, i, final):
                end = self._sentence_end(buf, i)
                if self._speakable(buf[:end]):
                    return self._take(end)
        # 2) an over-long segment: cut on the last soft stop
        if len(buf) >= self.max_chars:
            window = buf[: self.max_chars]
            cut = max(window.rfind(ch) for ch in SOFT_STOPS)
            if cut < self.max_chars * 0.5:
                cut = self.max_chars - 1
            return self._take(cut + 1)
        # 3) enough text at a natural break that speaking early is worth it.
        #    The break almost always sits *past* flush_chars, so search the
        #    whole buffer rather than only the first flush_chars characters.
        if len(buf) >= self.flush_chars:
            cut = max(buf.rfind(ch) for ch in SOFT_STOPS)
            if cut >= self.min_chars and (
                cut + 1 >= self.flush_chars or len(buf) >= self.flush_chars * 2
            ):
                return self._take(cut + 1)
        return None

    @staticmethod
    def _sentence_end(buf: str, index: int) -> int:
        """Extend past any closing quotes/brackets that belong to the sentence."""
        end = index + 1
        while end < len(buf) and buf[end] in _TRAILING:
            end += 1
        return end

    @staticmethod
    def _is_period_terminator(buf: str, index: int, final: bool) -> bool:
        """Decide whether an ASCII ``.`` ends a sentence.

        Bare periods are everywhere in things people read out loud —
        ``config.toml``, ``3.14``, ``example.com``, ``v1.2`` — so a period only
        counts when it is followed by whitespace. A trailing period is
        ambiguous while streaming, so it is only accepted once the stream ends.
        """
        following = buf[index + 1] if index + 1 < len(buf) else ""
        if following:
            return following.isspace() or following in _TRAILING
        return final

    def _take(self, n: int) -> str:
        segment, self._buf = self._buf[:n], self._buf[n:]
        return segment

    @staticmethod
    def _speakable(text: str) -> bool:
        """True when there is at least one letter/digit/CJK character."""
        stripped = text.strip()
        if len(stripped) < 1:
            return False
        return any(
            ch.isalnum() or unicodedata.category(ch).startswith("L") for ch in stripped
        )


__all__ = [
    "SentenceSplitter",
    "sanitize_for_speech",
    "sentence_spans",
    "truncate_for_speech",
]
