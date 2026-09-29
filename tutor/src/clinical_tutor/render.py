"""Convert the light Markdown used in lessons into Telegram-safe HTML.

Supported: ``**bold**``, ``_italic_``, ``` `code` ```, fenced code blocks, ``[text](url)`` links,
and ``- `` bullets (rendered as ``•``). Everything else is escaped.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

TELEGRAM_LIMIT = 4096
SAFE_LIMIT = 3900
TELEGRAM_TAGS = {"b", "i", "u", "s", "code", "pre", "a", "blockquote", "tg-spoiler"}


class _TagChecker(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.problems: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in TELEGRAM_TAGS:
            self.problems.append(f"unsupported tag <{tag}>")
        self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"mismatched </{tag}>")
            return
        self.stack.pop()


def check_html(rendered: str) -> list[str]:
    """Problems Telegram would reject: unsupported tags, unbalanced tags, or over-long text."""
    checker = _TagChecker()
    checker.feed(rendered)
    checker.close()
    problems = checker.problems + [f"unclosed <{tag}>" for tag in checker.stack]
    visible = len(html.unescape(re.sub(r"<[^>]+>", "", rendered)))
    if visible > TELEGRAM_LIMIT:
        problems.append(f"{visible} visible characters (Telegram limit {TELEGRAM_LIMIT})")
    return problems

_FENCE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC = re.compile(r"(?<![\w*])_(?!\s)((?:(?!\n\n).)+?)(?<!\s)_(?![\w*])", re.DOTALL)
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_BULLET = re.compile(r"^(\s*)[-*] ", re.MULTILINE)
# A single newline inside a paragraph is a soft wrap (as in Markdown), unless the next line
# starts a list item. Telegram would otherwise show hard breaks mid-sentence on a phone.
_SOFT_WRAP = re.compile(r"(?<=[^\s\x00])[ \t]*\n(?![ \t]*(?:\n|[-*] |\d+[.)] |\x00))[ \t]*")


def escape(text: str) -> str:
    return html.escape(text, quote=False)


def code_block(code: str, language: str | None = None) -> str:
    body = escape(code.rstrip("\n"))
    if language:
        return f'<pre><code class="language-{escape(language)}">{body}</code></pre>'
    return f"<pre>{body}</pre>"


def md(text: str) -> str:
    """Render lesson Markdown to Telegram HTML."""
    placeholders: list[str] = []

    def stash(fragment: str) -> str:
        placeholders.append(fragment)
        return f"\x00{len(placeholders) - 1}\x00"

    text = _FENCE.sub(lambda m: stash(code_block(m.group(2), m.group(1) or None)), text)
    text = _SOFT_WRAP.sub(" ", text)
    text = _INLINE_CODE.sub(lambda m: stash(f"<code>{escape(m.group(1))}</code>"), text)
    text = _LINK.sub(
        lambda m: stash(f'<a href="{escape(m.group(2))}">{escape(m.group(1))}</a>'), text
    )
    text = escape(text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    text = _BULLET.sub(r"\1• ", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], text)
    return text.strip()


def split_message(rendered: str, limit: int = SAFE_LIMIT) -> list[str]:
    """Split rendered HTML on paragraph boundaries so each chunk fits one Telegram message.

    Paragraphs are never split internally unless a single paragraph exceeds the limit, in which
    case it is split on lines. Code blocks are kept whole where possible.
    """
    if len(rendered) <= limit:
        return [rendered]
    chunks: list[str] = []
    current = ""
    for paragraph in rendered.split("\n\n"):
        pieces = [paragraph] if len(paragraph) <= limit else _split_lines(paragraph, limit)
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_lines(paragraph: str, limit: int) -> list[str]:
    out: list[str] = []
    current = ""
    for line in paragraph.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                out.append(current)
            current = line[:limit]
    if current:
        out.append(current)
    return out


def progress_bar(done: int, total: int, width: int = 10) -> str:
    if total <= 0:
        return "░" * width
    filled = round(width * done / total)
    return "▓" * filled + "░" * (width - filled)
