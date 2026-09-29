"""Telegram transport: routes updates to the engine and sends its ``Out`` messages."""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    TelegramObject,
)

from .config import Settings
from .content import load_course
from .engine import Engine
from .llm import Tutor
from .render import split_message
from .steps import Out
from .store import Store

log = logging.getLogger(__name__)

GUIDE_PATH = Path(__file__).resolve().parents[2] / "CONTENT_GUIDE.md"
COMMANDS = [
    BotCommand(command="continue", description="Pick up where you left off"),
    BotCommand(command="map", description="Course map and lessons"),
    BotCommand(command="review", description="Spaced review of missed items"),
    BotCommand(command="interview", description="Mock interview question"),
    BotCommand(command="quiz", description="Fresh questions on this lesson"),
    BotCommand(command="later", description="Laptop to-do list"),
    BotCommand(command="progress", description="Your stats"),
    BotCommand(command="help", description="How this works"),
]


class AllowlistMiddleware(BaseMiddleware):
    """Silently ignore anyone who isn't on the allowlist (the bot spends your API credits)."""

    def __init__(self, allowed: set[int]):
        self.allowed = allowed

    async def __call__(self, handler, event: TelegramObject, data: dict[str, Any]) -> Any:
        user = data.get("event_from_user")
        if user is None or user.id not in self.allowed:
            if user is not None:
                log.info("ignoring update from non-allowlisted user %s", user.id)
            return None
        return await handler(event, data)


def _markup(out: Out) -> InlineKeyboardMarkup | None:
    if not out.buttons:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=b.text, callback_data=b.data) for b in row]
            for row in out.buttons
        ]
    )


def _plain(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text))


async def send(bot: Bot, chat_id: int, outs: list[Out], source: Message | None = None) -> None:
    """Send engine output. ``edit`` messages replace ``source`` (the message whose button was tapped)."""
    for out in outs:
        markup = _markup(out)
        if out.edit and source is not None:
            try:
                if source.photo:
                    await source.edit_caption(caption=out.text[:1024], reply_markup=markup)
                else:
                    await source.edit_text(out.text, reply_markup=markup)
                continue
            except TelegramBadRequest as exc:
                if "message is not modified" in str(exc):
                    continue
                log.info("edit failed (%s); sending a new message", exc)
        if out.image is not None:
            try:
                await bot.send_photo(
                    chat_id, FSInputFile(out.image), caption=out.text[:1024], reply_markup=markup
                )
                continue
            except (TelegramBadRequest, FileNotFoundError) as exc:
                log.warning("photo %s failed: %s", out.image, exc)
        chunks = split_message(out.text)
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            try:
                await bot.send_message(chat_id, chunk, reply_markup=markup if last else None)
            except TelegramBadRequest as exc:
                if "can't parse entities" not in str(exc):
                    raise
                log.warning("HTML rejected, sending plain text: %s", exc)
                await bot.send_message(
                    chat_id,
                    _plain(chunk)[:4096],
                    parse_mode=None,
                    reply_markup=markup if last else None,
                )


def build_router(engine: Engine, bot: Bot) -> Router:
    router = Router()

    def notifier(chat_id: int):
        async def notify(text: str) -> None:
            await bot.send_message(chat_id, text)
            await bot.send_chat_action(chat_id, ChatAction.TYPING)

        return notify

    async def typing(chat_id: int) -> None:
        with contextlib.suppress(Exception):
            await bot.send_chat_action(chat_id, ChatAction.TYPING)

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        user = message.from_user
        assert user is not None
        await send(bot, message.chat.id, await engine.start(user.id, user.first_name))

    @router.message(Command("continue", "next"))
    async def on_continue(message: Message) -> None:
        await typing(message.chat.id)
        outs = await engine.resume(message.from_user.id, notifier(message.chat.id))
        await send(bot, message.chat.id, outs)

    @router.message(Command("help"))
    async def on_help(message: Message) -> None:
        await send(bot, message.chat.id, await engine.help(message.from_user.id))

    @router.message(Command("map"))
    async def on_map(message: Message) -> None:
        await send(bot, message.chat.id, await engine.course_map(message.from_user.id))

    @router.message(Command("review"))
    async def on_review(message: Message) -> None:
        await send(bot, message.chat.id, await engine.review(message.from_user.id))

    @router.message(Command("interview"))
    async def on_interview(message: Message) -> None:
        await typing(message.chat.id)
        await send(bot, message.chat.id, await engine.interview(message.from_user.id))

    @router.message(Command("quiz"))
    async def on_quiz(message: Message) -> None:
        await typing(message.chat.id)
        await send(bot, message.chat.id, await engine.more_questions(message.from_user.id))

    @router.message(Command("later"))
    async def on_later(message: Message) -> None:
        await send(bot, message.chat.id, await engine.later(message.from_user.id))

    @router.message(Command("progress"))
    async def on_progress(message: Message) -> None:
        await send(bot, message.chat.id, await engine.progress(message.from_user.id))

    @router.message(F.text)
    async def on_text(message: Message) -> None:
        await typing(message.chat.id)
        outs = await engine.text(
            message.from_user.id, message.text or "", notifier(message.chat.id)
        )
        await send(bot, message.chat.id, outs)

    @router.message()
    async def on_other(message: Message) -> None:
        await message.answer(
            "I read text messages only. Type your answer or question, or /continue."
        )

    @router.callback_query()
    async def on_button(query: CallbackQuery) -> None:
        data = query.data or ""
        uid = query.from_user.id
        message = query.message if isinstance(query.message, Message) else None
        chat_id = message.chat.id if message else uid
        # Acknowledge immediately so the button stops spinning, even if the tutor takes a while.
        with contextlib.suppress(TelegramBadRequest):
            await query.answer()
        kind, _, rest = data.partition(":")
        args = [int(x) for x in rest.split(":") if x.lstrip("-").isdigit()] if rest else []

        if kind in {"d", "mq", "iv", "ivg", "n", "L", "go"}:
            await typing(chat_id)
        if kind == "n" and len(args) == 2:
            outs = await engine.advance(uid, args[0], args[1], notifier(chat_id))
        elif kind == "go":
            outs = await engine.resume(uid, notifier(chat_id))
        elif kind == "a" and len(args) == 2:
            outs = await engine.answer_quiz(uid, args[0], args[1])
        elif kind == "d" and len(args) == 2:
            outs = await engine.deeper(uid, args[0], args[1])
        elif kind == "mq":
            outs = await engine.more_questions(uid)
        elif kind == "rv":
            outs = await engine.reveal(uid)
        elif kind == "sk":
            outs = await engine.skip(uid)
        elif kind == "sr" and len(args) == 1:
            outs = await engine.self_rate(uid, args[0])
        elif kind == "rev":
            outs = await engine.review(uid)
        elif kind == "iv":
            outs = await engine.interview(uid)
        elif kind == "ivg":
            outs = await engine.interview(uid, fresh=True)
        elif kind == "map":
            outs = await engine.course_map(uid)
        elif kind == "M" and len(args) == 1:
            outs = await engine.module_view(uid, args[0])
        elif kind == "L" and len(args) == 1:
            outs = await engine.jump(uid, args[0], notifier(chat_id))
        elif kind == "ld" and len(args) == 1:
            outs = await engine.finish_later(uid, args[0])
        else:
            outs = []

        # Buttons on a lesson message are spent once used; remove them to keep the chat tidy.
        if (
            message is not None
            and kind in {"n", "rv", "sk", "sr", "go"}
            and not any(o.edit for o in outs)
        ):
            with contextlib.suppress(TelegramBadRequest):
                await message.edit_reply_markup(reply_markup=None)
        await send(bot, chat_id, outs, source=message)

    return router


async def nudge_loop(engine: Engine, bot: Bot, hour: int) -> None:
    while True:
        await asyncio.sleep(600)
        if datetime.now().hour != hour:
            continue
        try:
            for user_id, outs in await engine.nudges():
                with contextlib.suppress(Exception):
                    await send(bot, user_id, outs)
        except Exception:  # noqa: BLE001 - never let the nudge loop die
            log.exception("nudge loop failed")


async def main(course_dir: Path) -> None:
    settings = Settings()
    allowed = settings.allowed_ids
    course = load_course(course_dir)
    store = await Store(settings.data_dir / "tutor.sqlite3").open()
    tutor = (
        Tutor(
            settings.anthropic_api_key,
            settings.tutor_model,
            settings.tutor_effort,
            settings.tutor_language,
        )
        if settings.anthropic_api_key
        else None
    )
    guide = GUIDE_PATH.read_text(encoding="utf-8") if GUIDE_PATH.is_file() else ""
    engine = Engine(course, store, tutor, settings.data_dir, guide)

    bot = Bot(settings.telegram_bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dispatcher = Dispatcher()
    dispatcher.update.outer_middleware(AllowlistMiddleware(allowed))
    dispatcher.include_router(build_router(engine, bot))
    await bot.set_my_commands(COMMANDS)

    log.info(
        "Course loaded: %d lessons. AI tutor: %s. Allowed users: %s",
        len(course.order),
        f"on ({settings.tutor_model})" if tutor else "off",
        sorted(allowed),
    )
    tasks = []
    if 0 <= settings.nudge_hour <= 23:
        tasks.append(asyncio.create_task(nudge_loop(engine, bot, settings.nudge_hour)))
    try:
        await dispatcher.start_polling(bot)
    finally:
        for task in tasks:
            task.cancel()
        await store.close()
        await bot.session.close()


def run(course_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(main(course_dir))
