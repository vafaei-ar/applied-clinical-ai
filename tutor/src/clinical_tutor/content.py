"""Course content model, loader, and validator.

The course lives in ``tutor/course/``:

* ``course.yaml`` lists modules and the order of their lessons.
* ``<module_id>/<lesson_id>.yaml`` holds one lesson: metadata plus an ordered list of steps.
* ``images/`` holds figures referenced by ``image`` steps.

A lesson with ``status: seed`` has no authored steps; the tutor generates it at runtime when an
LLM is configured and falls back to its objectives and key points otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

# Telegram limits: 4096 characters per message, 1024 per photo caption. Rendered HTML is a bit
# longer than the source, so authored text is capped comfortably below the hard limit.
MAX_TEXT = 3200
MAX_CAPTION = 900
MAX_CODE = 2800

DEFAULT_COURSE_DIR = Path(__file__).resolve().parents[2] / "course"


class _Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Optional hand-written narration for the audio version; replaces the generated text.
    speak: str | None = Field(default=None, max_length=3000)

    @field_validator("options", "points", "key_points", mode="before", check_fields=False)
    @classmethod
    def _stringify_items(cls, value):
        # YAML reads a bare `- 4` or `- 0.5` as a number; authors mean the text "4".
        # Anything else that isn't text is almost always a YAML slip, so reject it with a hint:
        # an unquoted `key: value` becomes a mapping, and `yes`/`no`/`on`/`off` become booleans.
        if not isinstance(value, list):
            return value
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                out.append(str(item))
            else:
                raise ValueError(
                    f"list item {item!r} is not text; quote it (for example an item containing "
                    "': ' or a bare yes/no must be in quotes)"
                )
        return out


class TextStep(_Step):
    type: Literal["text"]
    title: str | None = None
    text: str = Field(max_length=MAX_TEXT)


class CodeStep(_Step):
    type: Literal["code"]
    title: str | None = None
    text: str | None = Field(default=None, max_length=1200)
    language: str = "sql"
    code: str = Field(max_length=MAX_CODE)
    after: str | None = Field(default=None, max_length=1200)


class ImageStep(_Step):
    type: Literal["image"]
    image: str
    caption: str = Field(max_length=MAX_CAPTION)


class QuizStep(_Step):
    type: Literal["quiz"]
    question: str = Field(max_length=1500)
    code: str | None = Field(default=None, max_length=1500)
    language: str = "sql"
    options: list[str] = Field(min_length=2, max_length=6)
    answer: int
    explanation: str = Field(max_length=1500)

    @field_validator("options")
    @classmethod
    def _short_options(cls, options: list[str]) -> list[str]:
        for option in options:
            if len(option) > 300:
                raise ValueError(f"quiz option longer than 300 characters: {option[:40]}...")
        return options

    @model_validator(mode="after")
    def _answer_in_range(self) -> QuizStep:
        if not 0 <= self.answer < len(self.options):
            raise ValueError(f"answer index {self.answer} is out of range")
        return self


class ThinkStep(_Step):
    """An open question the learner answers in their own words."""

    type: Literal["think"]
    prompt: str = Field(max_length=1500)
    model_answer: str = Field(max_length=MAX_TEXT)
    key_points: list[str] = Field(min_length=1)
    interview: bool = False


class LaptopStep(_Step):
    """A hands-on task queued for when the learner is at a computer."""

    type: Literal["laptop"]
    title: str = Field(max_length=120)
    task: str = Field(max_length=1500)
    repo_path: str | None = None


class RecapStep(_Step):
    type: Literal["recap"]
    points: list[str] = Field(min_length=1)


Step = Annotated[
    TextStep | CodeStep | ImageStep | QuizStep | ThinkStep | LaptopStep | RecapStep,
    Field(discriminator="type"),
]
STEP_ADAPTER: TypeAdapter[Step] = TypeAdapter(Step)


class Lesson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    summary: str
    minutes: int = 20
    status: Literal["authored", "seed"] = "authored"
    why_it_matters: str
    objectives: list[str] = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)

    @model_validator(mode="after")
    def _steps_match_status(self) -> Lesson:
        if self.status == "authored" and not self.steps:
            raise ValueError("authored lessons need at least one step")
        if self.status == "seed" and not self.key_points:
            raise ValueError("seed lessons need key_points so they can be generated or summarized")
        return self


class Module(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    summary: str
    lessons: list[str] = Field(min_length=1)


class Case(BaseModel):
    """A realistic scenario that needs several lessons at once ("case of the day")."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = Field(max_length=100)
    scenario: str = Field(max_length=1000)
    question: str = Field(max_length=300)
    model_answer: str = Field(max_length=MAX_TEXT)
    key_points: list[str] = Field(min_length=2)
    # Lessons whose ideas the case draws on; a case unlocks once most of them are covered.
    lessons: list[str] = Field(min_length=1)


def case_step(case: Case) -> ThinkStep:
    """The open question a learner sees for a case (scenario and task in one message)."""
    return ThinkStep(
        type="think",
        prompt=f"**{case.title}**\n\n{case.scenario.strip()}\n\n**Your task:** {case.question}",
        model_answer=case.model_answer,
        key_points=case.key_points,
    )


class CourseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    description: str
    modules: list[Module]


class Course:
    """The loaded course: modules, lessons in order, and lookup helpers."""

    def __init__(
        self,
        root: Path,
        spec: CourseSpec,
        lessons: dict[str, Lesson],
        cases: list[Case] | None = None,
    ):
        self.root = root
        self.spec = spec
        self.lessons = lessons
        self.cases: list[Case] = cases or []
        self.order: list[str] = [lid for module in spec.modules for lid in module.lessons]
        self.module_of: dict[str, Module] = {
            lid: module for module in spec.modules for lid in module.lessons
        }

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def first_lesson(self) -> str:
        return self.order[0]

    def next_lesson(self, lesson_id: str) -> str | None:
        index = self.order.index(lesson_id)
        return self.order[index + 1] if index + 1 < len(self.order) else None

    def image_path(self, name: str) -> Path:
        return self.root / "images" / name


def _lesson_files(root: Path) -> dict[str, Path]:
    return {path.stem: path for path in root.glob("*/*.yaml")}


def load_lesson_file(path: Path) -> Lesson:
    with path.open(encoding="utf-8") as handle:
        return Lesson.model_validate(yaml.safe_load(handle))


def load_course(root: Path = DEFAULT_COURSE_DIR) -> Course:
    """Load and validate the full course. Raises ``CourseError`` listing every problem found."""
    errors: list[str] = []
    with (root / "course.yaml").open(encoding="utf-8") as handle:
        spec = CourseSpec.model_validate(yaml.safe_load(handle))

    files = _lesson_files(root)
    lessons: dict[str, Lesson] = {}
    seen: set[str] = set()
    for module in spec.modules:
        for lesson_id in module.lessons:
            if lesson_id in seen:
                errors.append(f"{lesson_id}: listed twice in course.yaml")
            seen.add(lesson_id)
            path = files.get(lesson_id)
            if path is None:
                errors.append(f"{lesson_id}: no file {module.id}/{lesson_id}.yaml")
                continue
            if path.parent.name != module.id:
                errors.append(f"{lesson_id}: file is in {path.parent.name}/, expected {module.id}/")
            try:
                lesson = load_lesson_file(path)
            except Exception as exc:  # noqa: BLE001 - collect every validation problem
                errors.append(f"{path.relative_to(root)}: {exc}")
                continue
            if lesson.id != lesson_id:
                errors.append(f"{path.relative_to(root)}: id '{lesson.id}' != file name")
            for index, step in enumerate(lesson.steps):
                if isinstance(step, ImageStep) and not (root / "images" / step.image).is_file():
                    errors.append(f"{lesson_id} step {index}: missing image images/{step.image}")
                errors.extend(f"{lesson_id} step {index}: {p}" for p in _render_problems(step))
            lessons[lesson_id] = lesson

    for orphan in sorted(set(files) - seen):
        errors.append(f"{files[orphan].relative_to(root)}: not listed in course.yaml")
    cases = _load_cases(root, set(lessons), errors)
    if errors:
        raise CourseError(errors)
    return Course(root, spec, lessons, cases)


def _load_cases(root: Path, lesson_ids: set[str], errors: list[str]) -> list[Case]:
    path = root / "cases.yaml"
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cases = [Case.model_validate(item) for item in raw.get("cases", [])]
    except Exception as exc:  # noqa: BLE001 - report every validation problem
        errors.append(f"cases.yaml: {exc}")
        return []
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            errors.append(f"cases.yaml: duplicate case id {case.id}")
        seen.add(case.id)
        for lesson_id in case.lessons:
            if lesson_id not in lesson_ids:
                errors.append(f"cases.yaml: case {case.id} references unknown lesson {lesson_id}")
        for problem in _case_render_problems(case):
            errors.append(f"cases.yaml: case {case.id}: {problem}")
    return cases


def _case_render_problems(case: Case) -> list[str]:
    from .render import check_html
    from .steps import model_answer_text, think_text

    step = case_step(case)
    return check_html(think_text(step, "🧭 <b>Case of the day</b>")) + check_html(
        model_answer_text(step)
    )


def _render_problems(step: Step) -> list[str]:
    """Check that a step renders to HTML Telegram accepts (imported lazily to avoid a cycle)."""
    from .render import check_html
    from .steps import quiz_result_text, render_step

    problems = [p for out in render_step(step) for p in check_html(out.text)]
    if isinstance(step, ImageStep):
        problems = [p for p in problems if "visible characters" not in p]
    if isinstance(step, QuizStep):
        problems += check_html(quiz_result_text(step, step.answer))
    return problems


class CourseError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))
