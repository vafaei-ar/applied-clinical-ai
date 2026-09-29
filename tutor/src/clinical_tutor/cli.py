"""Command-line entry point: run the bot, validate the course, or preview a lesson."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .content import DEFAULT_COURSE_DIR, CourseError, ImageStep, load_course


def _validate(course_dir: Path) -> int:
    try:
        course = load_course(course_dir)
    except CourseError as exc:
        print(f"{len(exc.errors)} problem(s) found:")
        for error in exc.errors:
            print(f"  - {error}")
        return 1
    authored = sum(1 for lesson in course.lessons.values() if lesson.status == "authored")
    steps = sum(len(lesson.steps) for lesson in course.lessons.values())
    print(
        f"OK: {len(course.spec.modules)} modules, {len(course.order)} lessons "
        f"({authored} authored, {len(course.order) - authored} seed), {steps} steps."
    )
    return 0


def _outline(course_dir: Path) -> int:
    course = load_course(course_dir)
    for module in course.spec.modules:
        print(f"\n{module.id}  {module.title}")
        for lesson_id in module.lessons:
            lesson = course.lessons[lesson_id]
            mark = "*" if lesson.status == "authored" else " "
            print(f"  {mark} {lesson_id:42s} {len(lesson.steps):3d} steps  {lesson.title}")
    return 0


def _preview(course_dir: Path, lesson_id: str) -> int:
    """Print a lesson as the bot would send it, one step after another."""
    from .steps import render_step

    course = load_course(course_dir)
    lesson = course.lessons.get(lesson_id)
    if lesson is None:
        print(f"Unknown lesson {lesson_id}")
        return 1
    for index, step in enumerate(lesson.steps):
        print(f"\n──── step {index} · {step.type} " + "─" * 40)
        if isinstance(step, ImageStep):
            print(f"[image {step.image}]")
        for message in render_step(step):
            print(message.text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clinical-tutor")
    parser.add_argument("--course", type=Path, default=DEFAULT_COURSE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="start the Telegram bot")
    sub.add_parser("validate", help="validate every lesson in the course")
    sub.add_parser("outline", help="print the course outline")
    preview = sub.add_parser("preview", help="print a lesson as Telegram HTML")
    preview.add_argument("lesson_id")
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _validate(args.course)
    if args.command == "outline":
        return _outline(args.course)
    if args.command == "preview":
        return _preview(args.course, args.lesson_id)

    from .bot import run

    run(args.course)
    return 0


if __name__ == "__main__":
    sys.exit(main())
