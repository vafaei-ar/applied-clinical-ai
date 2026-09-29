"""Outgoing message types and how each lesson step renders to Telegram HTML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .content import (
    CodeStep,
    ImageStep,
    LaptopStep,
    QuizStep,
    RecapStep,
    Step,
    TextStep,
    ThinkStep,
)
from .render import code_block, escape, md

LETTERS = "ABCDEF"


@dataclass(frozen=True)
class Button:
    text: str
    data: str


@dataclass
class Out:
    """One message for the bot to send (or, with ``edit``, to replace the tapped message)."""

    text: str
    buttons: list[list[Button]] = field(default_factory=list)
    image: Path | None = None
    edit: bool = False


def quiz_text(step: QuizStep, header: str = "🧩 <b>Quick check</b>") -> str:
    parts = [header, md(step.question)]
    if step.code:
        parts.append(code_block(step.code, step.language))
    parts.append("\n".join(f"<b>{LETTERS[i]}.</b> {md(opt)}" for i, opt in enumerate(step.options)))
    return "\n\n".join(parts)


def quiz_result_text(step: QuizStep, chosen: int, header: str = "🧩 <b>Quick check</b>") -> str:
    correct = chosen == step.answer
    lines = []
    for i, option in enumerate(step.options):
        mark = "✅" if i == step.answer else ("❌" if i == chosen else "▫️")
        lines.append(f"{mark} <b>{LETTERS[i]}.</b> {md(option)}")
    verdict = (
        "✅ <b>Correct.</b>"
        if correct
        else f"❌ <b>Not quite.</b> The answer is {LETTERS[step.answer]}."
    )
    parts = [header, md(step.question)]
    if step.code:
        parts.append(code_block(step.code, step.language))
    parts += ["\n".join(lines), verdict, md(step.explanation)]
    return "\n\n".join(parts)


def think_text(step: ThinkStep) -> str:
    label = "🎤 <b>Interview question</b>" if step.interview else "🤔 <b>Think it through</b>"
    return (
        f"{label}\n\n{md(step.prompt)}\n\n"
        "<i>Type your answer in your own words — a few sentences is enough.</i>"
    )


def model_answer_text(step: ThinkStep) -> str:
    points = "\n".join(f"• {md(p)}" for p in step.key_points)
    return f"💡 <b>Model answer</b>\n\n{md(step.model_answer)}\n\n<b>Key points</b>\n{points}"


def render_step(step: Step) -> list[Out]:
    """Render a step without navigation buttons (the engine adds those)."""
    if isinstance(step, TextStep):
        title = f"<b>{escape(step.title)}</b>\n\n" if step.title else ""
        return [Out(title + md(step.text))]
    if isinstance(step, CodeStep):
        parts = []
        if step.title:
            parts.append(f"<b>{escape(step.title)}</b>")
        if step.text:
            parts.append(md(step.text))
        parts.append(code_block(step.code, step.language))
        if step.after:
            parts.append(md(step.after))
        return [Out("\n\n".join(parts))]
    if isinstance(step, ImageStep):
        return [Out(md(step.caption), image=Path(step.image))]
    if isinstance(step, QuizStep):
        return [Out(quiz_text(step))]
    if isinstance(step, ThinkStep):
        return [Out(think_text(step))]
    if isinstance(step, LaptopStep):
        where = f"\n\n📂 <code>{escape(step.repo_path)}</code>" if step.repo_path else ""
        return [
            Out(
                f"💻 <b>For your laptop: {escape(step.title)}</b>\n\n{md(step.task)}{where}\n\n"
                "<i>Saved to your /later list — keep going on your phone.</i>"
            )
        ]
    if isinstance(step, RecapStep):
        points = "\n".join(f"• {md(p)}" for p in step.points)
        return [Out(f"📌 <b>Recap</b>\n\n{points}")]
    raise TypeError(f"unknown step type {type(step).__name__}")
