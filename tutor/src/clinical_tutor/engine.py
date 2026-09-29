"""The tutor's brain: where each learner is, what to send next, and how to react to answers.

The engine is transport-agnostic. Each public method takes a user id (plus action details) and
returns a list of ``Out`` messages; ``bot.py`` turns those into Telegram calls.

Position model: each learner has a current lesson and a ``cursor`` pointing at the next item to
show. Item 0 is the lesson header, items 1..n are the lesson's steps, and item n+1 is the
completion screen. Navigation buttons carry the (lesson index, cursor) they were created at, so a
stale or double-tapped button can never skip content.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import yaml

from .content import (
    STEP_ADAPTER,
    CodeStep,
    Course,
    ImageStep,
    LaptopStep,
    Lesson,
    QuizStep,
    RecapStep,
    Step,
    TextStep,
    ThinkStep,
    load_lesson_file,
)
from .llm import Tutor, TutorUnavailable
from .render import escape, md, progress_bar
from .steps import (
    LETTERS,
    Button,
    Out,
    model_answer_text,
    quiz_result_text,
    quiz_text,
    render_step,
    think_text,
)
from .store import DAY, Store, User

log = logging.getLogger(__name__)

Notify = Callable[[str], Awaitable[None]]

WELCOME_BACK_AFTER = 3 * 3600
NUDGE_QUIET_PERIOD = 20 * 3600
SCORE_DOTS = ["○○○", "●○○", "●●○", "●●●"]
CONTINUE_WORDS = {"next", "n", "continue", "c", "go", "ok", "▶", "resume"}


def step_plain(step: Step) -> str:
    """A plain-text rendering of a step, used as context for the AI tutor."""
    if isinstance(step, TextStep):
        return f"{step.title or ''}\n{step.text}".strip()
    if isinstance(step, CodeStep):
        return "\n".join(p for p in [step.title, step.text, step.code, step.after] if p)
    if isinstance(step, ImageStep):
        return f"[figure] {step.caption}"
    if isinstance(step, QuizStep):
        options = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(step.options))
        code = f"\n{step.code}" if step.code else ""
        return (
            f"Quiz: {step.question}{code}\n{options}\n"
            f"Answer: {LETTERS[step.answer]}. {step.explanation}"
        )
    if isinstance(step, ThinkStep):
        return f"Open question: {step.prompt}"
    if isinstance(step, LaptopStep):
        return f"Laptop task: {step.title}. {step.task}"
    if isinstance(step, RecapStep):
        return "Recap:\n" + "\n".join(f"- {p}" for p in step.points)
    return ""


def _clean_generated_step(raw: dict[str, Any]) -> Step | None:
    data = {k: v for k, v in raw.items() if v not in ("", None)}
    if data.get("type") == "quiz" and not data.get("code"):
        data.pop("language", None)
    try:
        return STEP_ADAPTER.validate_python(data)
    except Exception:  # noqa: BLE001 - a malformed generated step is dropped, not fatal
        log.info("dropping invalid generated step: %s", str(raw)[:200])
        return None


class Engine:
    def __init__(
        self,
        course: Course,
        store: Store,
        tutor: Tutor | None,
        data_dir: Path,
        guide_text: str = "",
    ):
        self.course = course
        self.store = store
        self.tutor = tutor
        self.generated_dir = data_dir / "generated"
        self.guide_text = guide_text
        self._generation_locks: dict[str, asyncio.Lock] = {}

    # --- helpers ------------------------------------------------------------------------------

    def _index(self, lesson_id: str) -> int:
        return self.course.order.index(lesson_id)

    def _nav(self, user: User, label: str = "Continue ▶") -> Button:
        return Button(label, f"n:{self._index(user.lesson_id)}:{user.cursor}")

    async def _user(self, user_id: int, first_name: str | None = None) -> User:
        user = await self.store.get_user(user_id)
        if user is None:
            user = await self.store.create_user(user_id, first_name, self.course.first_lesson)
        return user

    async def _reload(self, user_id: int) -> User:
        user = await self.store.get_user(user_id)
        assert user is not None
        return user

    async def lesson(self, lesson_id: str, notify: Notify | None = None) -> Lesson:
        """Return a lesson, generating (and caching) seed lessons when the tutor is available."""
        lesson = self.course.lessons[lesson_id]
        if lesson.status == "authored":
            return lesson
        cached = self.generated_dir / f"{lesson_id}.yaml"
        if cached.is_file():
            try:
                return load_lesson_file(cached)
            except Exception:  # noqa: BLE001 - regenerate a corrupt cache
                log.warning("ignoring invalid generated lesson %s", cached)
        if self.tutor is None:
            return self._seed_fallback(lesson)

        lock = self._generation_locks.setdefault(lesson_id, asyncio.Lock())
        async with lock:
            if cached.is_file():
                return load_lesson_file(cached)
            if notify:
                await notify(
                    "✍️ <b>This lesson is being written for you right now.</b> It takes about a "
                    "minute; it's saved afterwards so it's instant next time."
                )
            try:
                raw_steps = await self.tutor.write_lesson(self._brief(lesson), self.guide_text)
            except TutorUnavailable as exc:
                log.warning("lesson generation failed for %s: %s", lesson_id, exc)
                return self._seed_fallback(lesson)
            steps = [s for s in (_clean_generated_step(r) for r in raw_steps) if s is not None]
            if len(steps) < 5:
                return self._seed_fallback(lesson)
            generated = lesson.model_copy(update={"status": "authored", "steps": steps})
            self.generated_dir.mkdir(parents=True, exist_ok=True)
            cached.write_text(
                yaml.safe_dump(
                    generated.model_dump(mode="json", exclude_none=True),
                    sort_keys=False,
                    allow_unicode=True,
                    width=100,
                ),
                encoding="utf-8",
            )
            return generated

    def _brief(self, lesson: Lesson) -> str:
        module = self.course.module_of[lesson.id]
        index = self._index(lesson.id)
        neighbours = [
            self.course.lessons[lid].title
            for lid in self.course.order[max(0, index - 2) : index + 3]
            if lid != lesson.id
        ]
        return yaml.safe_dump(
            {
                "course": self.course.title,
                "module": f"{module.id}: {module.title} — {module.summary}",
                "lesson_id": lesson.id,
                "title": lesson.title,
                "summary": lesson.summary,
                "why_it_matters": lesson.why_it_matters,
                "objectives": lesson.objectives,
                "key_points": lesson.key_points,
                "nearby_lessons": neighbours,
            },
            sort_keys=False,
            allow_unicode=True,
        )

    def _seed_fallback(self, lesson: Lesson) -> Lesson:
        steps: list[Step] = [
            TextStep(
                type="text",
                title="Overview",
                text=f"{lesson.summary}\n\n{lesson.why_it_matters}\n\n"
                "_This lesson hasn't been fully written yet. Here's its outline; with the AI tutor "
                "enabled it is generated automatically, and you can ask questions about any point._",
            ),
            TextStep(
                type="text",
                title="What to learn",
                text="\n".join(f"- {o}" for o in lesson.objectives),
            ),
            RecapStep(type="recap", points=lesson.key_points),
        ]
        return lesson.model_copy(update={"steps": steps})

    async def _context(self, user: User, focus_item: int | None = None) -> str:
        """Describe the learner's position for the AI tutor."""
        lesson = await self.lesson(user.lesson_id)
        module = self.course.module_of[lesson.id]
        shown = max(0, min(user.cursor - 1, len(lesson.steps)))
        focus = focus_item if focus_item is not None else shown
        lines = [
            f"Module {module.id}: {module.title}",
            f"Lesson: {lesson.title} ({lesson.id}): {lesson.summary}",
            "Objectives: " + "; ".join(lesson.objectives),
        ]
        if lesson.key_points:
            lines.append("Key points: " + "; ".join(lesson.key_points))
        if 1 <= focus <= len(lesson.steps):
            earlier = lesson.steps[max(0, focus - 4) : focus - 1]
            if earlier:
                lines.append(
                    "Steps just before:\n" + "\n---\n".join(step_plain(s)[:600] for s in earlier)
                )
            lines.append(
                f"Current step ({focus}/{len(lesson.steps)}):\n{step_plain(lesson.steps[focus - 1])}"
            )
        elif user.cursor == 0:
            lines.append("The learner is about to start this lesson.")
        return "\n\n".join(lines)

    async def _lesson_digest(self, user: User, limit: int = 14000) -> str:
        lesson = await self.lesson(user.lesson_id)
        upto = max(1, min(user.cursor - 1, len(lesson.steps)))
        body = "\n\n".join(step_plain(s) for s in lesson.steps[:upto])
        return f"{await self._context(user)}\n\nLesson content so far:\n{body[-limit:]}"

    # --- entry points -------------------------------------------------------------------------

    async def start(self, user_id: int, first_name: str | None) -> list[Out]:
        existing = await self.store.get_user(user_id)
        if existing is not None:
            return await self.resume(user_id)
        user = await self._user(user_id, first_name)
        name = f", {escape(first_name)}" if first_name else ""
        n_lessons = len(self.course.order)
        n_modules = len(self.course.spec.modules)
        tutor_line = (
            "• <b>Ask me anything</b>: just type a question at any point and your AI tutor "
            "answers in the context of where you are.\n"
            if self.tutor
            else ""
        )
        text = (
            f"👋 Welcome{name}!\n\n<b>{escape(self.course.title)}</b>\n"
            f"{escape(self.course.spec.description)}\n\n"
            f"{n_modules} modules · {n_lessons} lessons · built for your phone.\n\n"
            "<b>How it works</b>\n"
            "• It's one continuous path. Tap <b>Continue</b> to move on, and stop whenever you "
            "like. I remember exactly where you were.\n"
            f"{tutor_line}"
            "• Quizzes and interview questions you miss come back later for spaced review "
            "(/review).\n"
            "• Hands-on coding tasks are saved to /later for when you're at your laptop.\n\n"
            "Commands: /continue · /map · /review · /interview · /later · /progress · /help"
        )
        return [Out(text, [[self._nav(user, "Start the course ▶")]])]

    async def help(self, user_id: int) -> list[Out]:
        user = await self._user(user_id)
        ask = "Type any question to ask the AI tutor.\n" if self.tutor else ""
        text = (
            "<b>Commands</b>\n"
            "/continue: pick up exactly where you left off\n"
            "/map: the course map and jumping to any lesson\n"
            "/review: spaced review of things you missed\n"
            "/interview: a mock interview question on what you've covered\n"
            "/later: your laptop to-do list\n"
            "/progress: your stats\n"
            + ("/quiz: fresh questions on the current lesson\n" if self.tutor else "")
            + f"\n{ask}Typing <i>next</i> also continues."
        )
        return [Out(text, [[self._nav(user)]])]

    async def resume(self, user_id: int, notify: Notify | None = None) -> list[Out]:
        user = await self._user(user_id)
        away = time.time() - user.last_seen
        await self.store.touch(user_id)
        awaiting = user.awaiting or {}
        if awaiting.get("kind") == "quiz":
            record = await self.store.get_quiz(awaiting["quiz_id"])
            if record is not None and record.chosen is None:
                step = QuizStep.model_validate(record.payload)
                return [
                    Out("You still have this question open 👇"),
                    self._quiz_out(step, record.id),
                ]
        if awaiting.get("kind") == "think":
            step = ThinkStep.model_validate(awaiting["step"])
            return [Out("Picking up with this question 👇"), self._think_out(step)]

        out: list[Out] = []
        if away > WELCOME_BACK_AFTER and user.cursor > 1:
            lesson = await self.lesson(user.lesson_id)
            recap = ""
            if self.tutor is not None:
                try:
                    recap = md(await self.tutor.welcome_back(await self._context(user)))
                except TutorUnavailable:
                    recap = ""
            position = f"{min(user.cursor - 1, len(lesson.steps))}/{len(lesson.steps)}"
            out.append(
                Out(
                    f"👋 Welcome back. You're in <b>{escape(lesson.title)}</b>, step {position}."
                    + (f"\n\n{recap}" if recap else "")
                )
            )
        return out + await self._show_next(user, notify)

    async def advance(
        self, user_id: int, lesson_index: int, cursor: int, notify: Notify | None = None
    ) -> list[Out]:
        user = await self._user(user_id)
        await self.store.touch(user_id)
        if (lesson_index, cursor) != (self._index(user.lesson_id), user.cursor):
            # A stale button: show where the learner actually is instead of skipping ahead.
            return [Out("You've already moved past that point. Here's where you are 👇")] + (
                await self._show_next(user, notify)
            )
        if (user.awaiting or {}).get("kind") in {"quiz", "think"}:
            await self.store.set_awaiting(user_id, None)
            user = await self._reload(user_id)
        return await self._show_next(user, notify)

    # --- lesson flow --------------------------------------------------------------------------

    async def _show_next(self, user: User, notify: Notify | None = None) -> list[Out]:
        lesson = await self.lesson(user.lesson_id, notify)
        index = self._index(lesson.id)
        item = user.cursor
        if item == 0:
            await self.store.set_position(user.user_id, lesson.id, 1)
            return [self._header(lesson, index, len(lesson.steps))]
        if item > len(lesson.steps):
            return await self._complete(user, lesson)

        step = lesson.steps[item - 1]
        next_cursor = item + 1
        await self.store.set_position(user.user_id, lesson.id, next_cursor)
        nav = Button("Continue ▶", f"n:{index}:{next_cursor}")
        extras: list[Button] = []
        if self.tutor is not None and not isinstance(step, (RecapStep, LaptopStep)):
            extras = [Button("🔍 Go deeper", f"d:{index}:{item}")]

        if isinstance(step, QuizStep):
            quiz_id = await self.store.log_quiz(
                user.user_id,
                f"{lesson.id}#{item}",
                {"kind": "lesson", "li": index, "cursor": next_cursor},
                step.model_dump(),
            )
            await self.store.set_awaiting(user.user_id, {"kind": "quiz", "quiz_id": quiz_id})
            return [self._quiz_out(step, quiz_id, skip=Button("Skip", nav.data))]

        if isinstance(step, ThinkStep):
            await self.store.set_awaiting(
                user.user_id,
                {
                    "kind": "think",
                    "key": f"{lesson.id}#{item}",
                    "step": step.model_dump(),
                    "ctx": {"kind": "lesson", "li": index, "cursor": next_cursor},
                },
            )
            return [self._think_out(step)]

        if isinstance(step, LaptopStep):
            await self.store.add_later(
                user.user_id, lesson.id, step.title, step.task, step.repo_path
            )

        outs = render_step(step)
        if isinstance(step, ImageStep):
            outs = [Out(o.text, image=self.course.image_path(step.image)) for o in outs]
        progress = f"\n\n<i>{item}/{len(lesson.steps)}</i>" if item % 5 == 0 else ""
        last = outs[-1]
        last.text += progress
        last.buttons = [[nav, *extras]] if extras else [[nav]]
        return outs

    def _header(self, lesson: Lesson, index: int, n_steps: int) -> Out:
        module = self.course.module_of[lesson.id]
        in_module = module.lessons.index(lesson.id) + 1
        objectives = "\n".join(f"• {md(o)}" for o in lesson.objectives)
        text = (
            f"📘 <b>{escape(module.title)}</b> · lesson {in_module}/{len(module.lessons)}\n\n"
            f"<b>{escape(lesson.title)}</b>\n{md(lesson.summary)}\n\n"
            f"<b>Why it matters</b>\n{md(lesson.why_it_matters)}\n\n"
            f"<b>You'll be able to</b>\n{objectives}\n\n"
            f"⏱ ~{lesson.minutes} min · {n_steps} steps · course "
            f"{progress_bar(index, len(self.course.order))} {index}/{len(self.course.order)}"
        )
        return Out(text, [[Button("Start ▶", f"n:{index}:1")]])

    async def _complete(self, user: User, lesson: Lesson) -> list[Out]:
        await self.store.complete_lesson(user.user_id, lesson.id)
        answered, correct = await self.store.lesson_quiz_stats(user.user_id, lesson.id)
        due, _, _ = await self.store.review_counts(user.user_id)
        next_id = self.course.next_lesson(lesson.id)
        score = f"\nQuick checks: {correct}/{answered} correct." if answered else ""
        review = f"\nReview items due: {due}." if due else ""

        if next_id is None:
            text = (
                f"🎓 <b>You finished the whole course.</b>\n\n<b>{escape(lesson.title)}</b> "
                f"complete.{score}{review}\n\nKeep your skills sharp with /interview and /review, "
                "and work through your /later list at the laptop."
            )
            buttons = [[Button("🎤 Mock interview", "iv")]]
            if due:
                buttons[0].append(Button(f"🔁 Review ({due})", "rev"))
            return [Out(text, buttons)]

        await self.store.set_position(user.user_id, next_id, 0)
        next_lesson = self.course.lessons[next_id]
        new_module = self.course.module_of[next_id].id != self.course.module_of[lesson.id].id
        module_line = (
            f"\n\n🧭 Next up is a new module: <b>{escape(self.course.module_of[next_id].title)}</b>"
            if new_module
            else ""
        )
        text = f"✅ <b>Lesson complete: {escape(lesson.title)}</b>{score}{review}{module_line}"
        buttons = [[Button(f"▶ Next: {next_lesson.title[:40]}", f"n:{self._index(next_id)}:0")]]
        if due:
            buttons.append([Button(f"🔁 Review first ({due})", "rev")])
        return [Out(text, buttons)]

    def _quiz_out(
        self, step: QuizStep, quiz_id: int, header: str | None = None, skip: Button | None = None
    ) -> Out:
        text = quiz_text(step, header) if header else quiz_text(step)
        row = [Button(LETTERS[i], f"a:{quiz_id}:{i}") for i in range(len(step.options))]
        buttons = [row]
        if skip is not None:
            buttons.append([skip])
        return Out(text, buttons)

    def _think_out(self, step: ThinkStep, fresh: bool = False) -> Out:
        buttons = [[Button("💡 Show answer", "rv"), Button("⏭ Skip", "sk")]]
        if fresh and self.tutor is not None:
            buttons.append([Button("🎲 Different question", "ivg")])
        return Out(think_text(step), buttons)

    async def _continuation(self, user_id: int, ctx: dict[str, Any]) -> list[list[Button]]:
        kind = ctx.get("kind")
        if kind == "lesson":
            return [[Button("Continue ▶", f"n:{ctx['li']}:{ctx['cursor']}")]]
        if kind == "review":
            return [[Button("Next review ▶", "rev"), Button("Back to lesson", "go")]]
        if kind == "interview":
            return [[Button("🎤 Another question", "iv"), Button("Back to lesson", "go")]]
        user = await self._user(user_id)
        return [[self._nav(user)]]

    # --- quizzes ------------------------------------------------------------------------------

    async def answer_quiz(self, user_id: int, quiz_id: int, choice: int) -> list[Out]:
        record = await self.store.get_quiz(quiz_id)
        if record is None or record.user_id != user_id:
            return [Out("That question has expired.")]
        if record.chosen is not None:
            return []
        step = QuizStep.model_validate(record.payload)
        if not 0 <= choice < len(step.options):
            return []
        correct = choice == step.answer
        await self.store.answer_quiz(quiz_id, choice, correct)
        await self.store.touch(user_id)
        ctx = record.context
        kind = ctx.get("kind")

        if kind == "review":
            await self.store.record_review(user_id, record.item_key, correct)
        elif not correct:
            await self.store.schedule_review(user_id, record.item_key, "quiz", record.payload, 0)

        user = await self._user(user_id)
        if (user.awaiting or {}).get("quiz_id") == quiz_id:
            await self.store.set_awaiting(user_id, None)

        header = "🔁 <b>Review</b>" if kind == "review" else "🧩 <b>Quick check</b>"
        text = quiz_result_text(step, choice, header)
        if not correct and kind != "review":
            text += "\n\n<i>🔁 Added to your spaced review.</i>"
        buttons = [] if kind == "extra" else await self._continuation(user_id, ctx)
        return [Out(text, buttons, edit=True)]

    async def more_questions(self, user_id: int) -> list[Out]:
        user = await self._user(user_id)
        if self.tutor is None:
            return [Out("Fresh questions need the AI tutor (set ANTHROPIC_API_KEY).")]
        lesson = await self.lesson(user.lesson_id)
        existing = [s.question for s in lesson.steps if isinstance(s, QuizStep)]
        try:
            raw = await self.tutor.make_quiz(await self._lesson_digest(user), 3, existing)
        except TutorUnavailable as exc:
            return [Out(f"⚠️ {escape(str(exc))}", [[self._nav(user)]])]
        outs: list[Out] = []
        for item in raw:
            data = {k: v for k, v in item.items() if v != ""}
            try:
                step = QuizStep.model_validate({"type": "quiz", **data})
            except Exception:  # noqa: BLE001 - skip a malformed generated question
                continue
            key = f"gen:{lesson.id}:{time.time_ns()}"
            quiz_id = await self.store.log_quiz(user_id, key, {"kind": "extra"}, step.model_dump())
            outs.append(self._quiz_out(step, quiz_id, header="🧪 <b>Fresh question</b>"))
        if not outs:
            return [Out("⚠️ Couldn't generate good questions this time. Try again.")]
        outs.append(Out("Answer them above whenever you like.", [[self._nav(user)]]))
        return outs

    # --- open questions -----------------------------------------------------------------------

    async def text(self, user_id: int, text: str, notify: Notify | None = None) -> list[Out]:
        """Handle free text: an answer to an open question, a 'next', or a question for the tutor."""
        user = await self._user(user_id)
        await self.store.touch(user_id)
        awaiting = user.awaiting or {}
        if awaiting.get("kind") == "think" and not awaiting.get("revealed"):
            return await self._grade_think(user, awaiting, text)
        if text.strip().lower() in CONTINUE_WORDS:
            return await self.resume(user_id, notify)
        if self.tutor is None:
            return [
                Out(
                    "I can only follow the course right now. The AI tutor that answers questions "
                    "isn't enabled on this server (it needs ANTHROPIC_API_KEY).",
                    [[self._nav(user)]],
                )
            ]
        history = await self.store.recent_chat(user_id)
        try:
            reply = await self.tutor.answer(await self._context(user), history, text)
        except TutorUnavailable as exc:
            return [Out(f"⚠️ {escape(str(exc))}", [[self._nav(user)]])]
        await self.store.add_chat(user_id, "user", text)
        await self.store.add_chat(user_id, "assistant", reply)
        return [Out(md(reply), [[self._nav(user), Button("🧪 Quiz me", "mq")]])]

    async def _grade_think(self, user: User, awaiting: dict[str, Any], answer: str) -> list[Out]:
        step = ThinkStep.model_validate(awaiting["step"])
        ctx = awaiting.get("ctx", {})
        if self.tutor is None:
            await self.store.set_awaiting(
                user.user_id, {**awaiting, "revealed": True, "answer": answer}
            )
            return [
                Out(
                    model_answer_text(step) + "\n\n<b>How did your answer compare?</b>",
                    self._self_rate_buttons(),
                )
            ]
        try:
            grade = await self.tutor.grade(step.prompt, step.key_points, step.model_answer, answer)
        except TutorUnavailable as exc:
            await self.store.set_awaiting(
                user.user_id, {**awaiting, "revealed": True, "answer": answer}
            )
            return [
                Out(
                    f"⚠️ {escape(str(exc))} Here's the model answer instead.\n\n"
                    + model_answer_text(step)
                    + "\n\n<b>How did your answer compare?</b>",
                    self._self_rate_buttons(),
                )
            ]
        await self._record_think(user.user_id, awaiting, answer, grade.score)
        points = "\n".join(f"• {md(p)}" for p in step.key_points)
        text = (
            f"<b>Your answer</b> {SCORE_DOTS[grade.score]}\n\n{md(grade.feedback)}\n\n"
            f"💡 <b>Model answer</b>\n<blockquote expandable>{md(step.model_answer)}\n\n"
            f"<b>Key points</b>\n{points}</blockquote>"
        )
        if grade.score < 2 and ctx.get("kind") != "review":
            text += "\n\n<i>🔁 This one will come back in your spaced review.</i>"
        return [Out(text, await self._continuation(user.user_id, ctx))]

    def _self_rate_buttons(self) -> list[list[Button]]:
        return [
            [
                Button("✅ Nailed it", "sr:2"),
                Button("🟡 Partly", "sr:1"),
                Button("❌ Missed", "sr:0"),
            ]
        ]

    async def _record_think(
        self, user_id: int, awaiting: dict[str, Any], answer: str | None, score: int
    ) -> None:
        key = awaiting["key"]
        step = awaiting["step"]
        ctx = awaiting.get("ctx", {})
        await self.store.log_think(user_id, key, answer, score)
        if ctx.get("kind") == "review":
            await self.store.record_review(user_id, key, score >= 2)
        elif score < 2:
            await self.store.schedule_review(user_id, key, "think", step, 0)
        elif step.get("interview"):
            await self.store.schedule_review(user_id, key, "think", step, 1)
        await self.store.set_awaiting(user_id, None)

    async def reveal(self, user_id: int) -> list[Out]:
        user = await self._user(user_id)
        awaiting = user.awaiting or {}
        if awaiting.get("kind") != "think":
            return [Out("There's no open question right now.", [[self._nav(user)]])]
        step = ThinkStep.model_validate(awaiting["step"])
        await self.store.set_awaiting(user_id, {**awaiting, "revealed": True})
        return [
            Out(
                model_answer_text(step) + "\n\n<b>Could you have said this?</b>",
                self._self_rate_buttons(),
            )
        ]

    async def self_rate(self, user_id: int, rating: int) -> list[Out]:
        user = await self._user(user_id)
        awaiting = user.awaiting or {}
        if awaiting.get("kind") != "think":
            return []
        score = {0: 0, 1: 1, 2: 3}.get(rating, 1)
        await self._record_think(user_id, awaiting, awaiting.get("answer"), score)
        note = {
            0: "🔁 It'll come back in your spaced review.",
            1: "🔁 It'll come back in your spaced review.",
            2: "💪 Nice.",
        }[rating if rating in (0, 1, 2) else 1]
        return [Out(note, await self._continuation(user_id, awaiting.get("ctx", {})))]

    async def skip(self, user_id: int) -> list[Out]:
        user = await self._user(user_id)
        awaiting = user.awaiting or {}
        ctx = awaiting.get("ctx", {})
        await self.store.set_awaiting(user_id, None)
        return [Out("Skipped.", await self._continuation(user_id, ctx))]

    # --- deeper -------------------------------------------------------------------------------

    async def deeper(self, user_id: int, lesson_index: int, item: int) -> list[Out]:
        user = await self._user(user_id)
        if self.tutor is None:
            return [Out("Going deeper needs the AI tutor (set ANTHROPIC_API_KEY).")]
        focus = item if lesson_index == self._index(user.lesson_id) else None
        try:
            reply = await self.tutor.deepen(await self._context(user, focus))
        except TutorUnavailable as exc:
            return [Out(f"⚠️ {escape(str(exc))}", [[self._nav(user)]])]
        await self.store.add_chat(user_id, "user", "(Asked to go deeper on the current step.)")
        await self.store.add_chat(user_id, "assistant", reply)
        return [Out(f"🔍 {md(reply)}", [[self._nav(user), Button("🧪 Quiz me", "mq")]])]

    # --- spaced review ------------------------------------------------------------------------

    async def review(self, user_id: int) -> list[Out]:
        user = await self._user(user_id)
        await self.store.touch(user_id)
        item = await self.store.next_due_review(user_id)
        if item is None:
            _, total, next_due = await self.store.review_counts(user_id)
            if total == 0:
                text = "🔁 Nothing to review yet. Questions you miss will show up here."
            else:
                hours = max(1, round((next_due - time.time()) / 3600)) if next_due else 0
                when = f"{round(hours / 24)} day(s)" if hours >= 36 else f"{hours} hour(s)"
                text = (
                    f"🔁 All caught up. {total} item(s) scheduled; the next one is due in ~{when}."
                )
            return [Out(text, [[self._nav(user, "Back to lesson ▶")]])]

        due, _, _ = await self.store.review_counts(user_id)
        header = f"🔁 <b>Review</b> · {due} due"
        if item.kind == "quiz":
            step = QuizStep.model_validate(item.payload)
            quiz_id = await self.store.log_quiz(
                user_id, item.item_key, {"kind": "review"}, item.payload
            )
            return [self._quiz_out(step, quiz_id, header=header)]
        step = ThinkStep.model_validate(item.payload)
        await self.store.set_awaiting(
            user_id,
            {
                "kind": "think",
                "key": item.item_key,
                "step": item.payload,
                "ctx": {"kind": "review"},
            },
        )
        out = self._think_out(step)
        out.text = f"{header}\n\n{out.text}"
        return [out]

    # --- mock interview -----------------------------------------------------------------------

    async def interview(self, user_id: int, fresh: bool = False) -> list[Out]:
        user = await self._user(user_id)
        await self.store.touch(user_id)
        covered = await self._covered_lessons(user)
        answered = await self.store.answered_think_keys(user_id)

        pool: list[tuple[str, ThinkStep]] = []
        for lesson_id in covered:
            lesson = self.course.lessons[lesson_id]
            for i, step in enumerate(lesson.steps, 1):
                if isinstance(step, ThinkStep) and step.interview:
                    pool.append((f"{lesson_id}#{i}", step))
        unseen = [p for p in pool if p[0] not in answered]

        if (fresh or not unseen) and self.tutor is not None:
            topics = "\n".join(
                f"- {self.course.lessons[lid].title}: "
                + "; ".join(self.course.lessons[lid].key_points)
                for lid in covered or [user.lesson_id]
            )
            try:
                data = await self.tutor.interview_question(topics, [s.prompt for _, s in pool])
                step = ThinkStep(type="think", interview=True, **data)
                key = f"gen-iv:{time.time_ns()}"
            except (TutorUnavailable, ValueError) as exc:
                return [Out(f"⚠️ {escape(str(exc))}", [[self._nav(user, "Back to lesson ▶")]])]
        elif pool:
            key, step = random.choice(unseen or pool)
        else:
            return [
                Out(
                    "🎤 Interview questions unlock as you work through lessons. Keep going!",
                    [[self._nav(user)]],
                )
            ]

        await self.store.set_awaiting(
            user_id,
            {"kind": "think", "key": key, "step": step.model_dump(), "ctx": {"kind": "interview"}},
        )
        out = self._think_out(step, fresh=True)
        out.text = "🎤 <b>Mock interview</b>\n\n" + out.text.split("\n\n", 1)[1]
        return [out]

    async def _covered_lessons(self, user: User) -> list[str]:
        completed = await self.store.completed_lessons(user.user_id)
        covered = [lid for lid in self.course.order if lid in completed]
        if user.cursor > 1 and user.lesson_id not in completed:
            covered.append(user.lesson_id)
        return covered

    # --- navigation & stats -------------------------------------------------------------------

    async def course_map(self, user_id: int) -> list[Out]:
        user = await self._user(user_id)
        completed = await self.store.completed_lessons(user_id)
        lines = [f"🗺 <b>{escape(self.course.title)}</b>\n"]
        buttons: list[list[Button]] = []
        for mi, module in enumerate(self.course.spec.modules):
            done = sum(1 for lid in module.lessons if lid in completed)
            here = " 📍" if user.lesson_id in module.lessons else ""
            lines.append(
                f"<b>{escape(module.title)}</b>{here}\n"
                f"{progress_bar(done, len(module.lessons), 8)} {done}/{len(module.lessons)}"
            )
            buttons.append([Button(f"{module.title[:48]}", f"M:{mi}")])
        buttons.append([self._nav(user, "Continue where I was ▶")])
        return [Out("\n\n".join(lines), buttons)]

    async def module_view(self, user_id: int, module_index: int) -> list[Out]:
        user = await self._user(user_id)
        if not 0 <= module_index < len(self.course.spec.modules):
            return []
        module = self.course.spec.modules[module_index]
        completed = await self.store.completed_lessons(user_id)
        lines = [f"<b>{escape(module.title)}</b>\n{md(module.summary)}\n"]
        buttons = []
        for n, lesson_id in enumerate(module.lessons, 1):
            lesson = self.course.lessons[lesson_id]
            mark = (
                "✅" if lesson_id in completed else ("📍" if lesson_id == user.lesson_id else "▫️")
            )
            lines.append(f"{mark} {n}. {escape(lesson.title)}")
            buttons.append([Button(f"{n}. {lesson.title[:50]}", f"L:{self._index(lesson_id)}")])
        buttons.append([Button("« Map", "map"), self._nav(user, "Continue ▶")])
        return [Out("\n".join(lines), buttons)]

    async def jump(
        self, user_id: int, lesson_index: int, notify: Notify | None = None
    ) -> list[Out]:
        if not 0 <= lesson_index < len(self.course.order):
            return []
        await self._user(user_id)
        await self.store.set_awaiting(user_id, None)
        await self.store.set_position(user_id, self.course.order[lesson_index], 0)
        return await self._show_next(await self._reload(user_id), notify)

    async def progress(self, user_id: int) -> list[Out]:
        user = await self._user(user_id)
        completed = await self.store.completed_lessons(user_id)
        answered, correct = await self.store.quiz_stats(user_id)
        due, total, _ = await self.store.review_counts(user_id)
        later = await self.store.list_later(user_id)
        lesson = self.course.lessons[user.lesson_id]
        total_lessons = len(self.course.order)
        accuracy = f"{round(100 * correct / answered)}%" if answered else "no answers yet"
        days = max(1, round((time.time() - user.created_at) / DAY))
        text = (
            "📈 <b>Your progress</b>\n\n"
            f"{progress_bar(len(completed), total_lessons)} {len(completed)}/{total_lessons} lessons\n"
            f"📍 Now: {escape(lesson.title)}\n"
            f"🧩 Quick checks: {correct}/{answered} ({accuracy})\n"
            f"🔁 Review: {due} due, {total} scheduled\n"
            f"💻 Laptop tasks open: {len(later)}\n"
            f"📅 Day {days} of your journey"
        )
        buttons = [[self._nav(user)]]
        if due:
            buttons[0].append(Button(f"🔁 Review ({due})", "rev"))
        return [Out(text, buttons)]

    async def later(self, user_id: int) -> list[Out]:
        user = await self._user(user_id)
        tasks = await self.store.list_later(user_id)
        if not tasks:
            return [
                Out(
                    "💻 Your laptop list is empty. Hands-on tasks from lessons land here.",
                    [[self._nav(user)]],
                )
            ]
        lines = ["💻 <b>For your laptop</b>\n"]
        buttons = []
        for n, task in enumerate(tasks, 1):
            where = f"\n   📂 <code>{escape(task.repo_path)}</code>" if task.repo_path else ""
            lines.append(f"<b>{n}. {escape(task.title)}</b>\n{md(task.task)}{where}")
            buttons.append([Button(f"✅ Done: {n}. {task.title[:40]}", f"ld:{task.id}")])
        buttons.append([self._nav(user, "Back to lesson ▶")])
        return [Out("\n\n".join(lines), buttons)]

    async def finish_later(self, user_id: int, task_id: int) -> list[Out]:
        await self.store.finish_later(user_id, task_id)
        outs = await self.later(user_id)
        for out in outs:
            out.edit = True
        return outs

    # --- nudges -------------------------------------------------------------------------------

    async def nudges(self) -> list[tuple[int, list[Out]]]:
        """Messages for learners who have been away for a while (sent at most once a day)."""
        now = time.time()
        result = []
        for user in await self.store.list_users():
            if (
                now - user.last_seen < NUDGE_QUIET_PERIOD
                or now - user.last_nudge < NUDGE_QUIET_PERIOD
            ):
                continue
            lesson = self.course.lessons[user.lesson_id]
            due, _, _ = await self.store.review_counts(user.user_id)
            text = f"📚 Ready for a few minutes? You're on <b>{escape(lesson.title)}</b>."
            if due:
                text += f"\n🔁 {due} review item(s) are due."
            buttons = [[self._nav(user)]]
            if due:
                buttons[0].append(Button(f"🔁 Review ({due})", "rev"))
            await self.store.mark_nudged(user.user_id)
            result.append((user.user_id, [Out(text, buttons)]))
        return result
