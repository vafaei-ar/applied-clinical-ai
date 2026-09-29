from __future__ import annotations

from pathlib import Path
from typing import Any

from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage

from clinical_tutor.bot import send
from clinical_tutor.steps import Button, Out


class StubBot:
    def __init__(self, reject_html: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.reject_html = reject_html

    async def send_message(self, chat_id, text, parse_mode="default", reply_markup=None):
        if self.reject_html and parse_mode == "default":
            raise TelegramBadRequest(
                method=SendMessage(chat_id=chat_id, text=text), message="Bad Request: can't parse entities"
            )
        self.sent.append({"kind": "text", "text": text, "markup": reply_markup})

    async def send_photo(self, chat_id, photo, caption=None, reply_markup=None):
        self.sent.append({"kind": "photo", "caption": caption, "markup": reply_markup})


async def test_long_text_is_split_and_buttons_go_last():
    bot = StubBot()
    text = "\n\n".join("y" * 1500 for _ in range(4))
    await send(bot, 1, [Out(text, [[Button("Continue ▶", "n:0:1")]])])
    assert len(bot.sent) == 2
    assert bot.sent[0]["markup"] is None and bot.sent[1]["markup"] is not None


async def test_html_rejection_falls_back_to_plain_text():
    bot = StubBot(reject_html=True)
    await send(bot, 1, [Out("<b>bold</b> &amp; more")])
    assert bot.sent == [{"kind": "text", "text": "bold & more", "markup": None}]


async def test_image_is_sent_as_photo(tmp_path: Path):
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG")
    bot = StubBot()
    await send(bot, 1, [Out("caption", image=image)])
    assert bot.sent[0]["kind"] == "photo" and bot.sent[0]["caption"] == "caption"
