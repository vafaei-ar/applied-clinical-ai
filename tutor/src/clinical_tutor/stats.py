"""Usage report over the learner database: where the course needs work.

Read-only. Everything stays on the computer that runs the bot; nothing is uploaded.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from .content import Course, QuizStep, ThinkStep
from .steps import LETTERS

DB_NAME = "tutor.sqlite3"


def _step_label(course: Course, lesson_id: str, item: int) -> str:
    lesson = course.lessons.get(lesson_id)
    if lesson is None or not 1 <= item <= len(lesson.steps):
        return f"{lesson_id}#{item}"
    step = lesson.steps[item - 1]
    if isinstance(step, QuizStep):
        text = step.question
    elif isinstance(step, ThinkStep):
        text = step.prompt
    else:
        text = getattr(step, "title", None) or getattr(step, "text", None) or step.type
    text = " ".join((text or step.type).split())
    return f"{lesson_id}#{item} [{step.type}] {text[:70]}"


def _split_key(item_key: str) -> tuple[str, int] | None:
    lesson_id, sep, item = item_key.partition("#")
    if sep and item.isdigit():
        return lesson_id, int(item)
    return None


def _rows(db: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Run a query, treating a table missing from an older database as empty."""
    try:
        return db.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []
        raise


def build_report(course: Course, db_path: Path, min_attempts: int = 1) -> str:
    if not db_path.is_file():
        return f"No learner data yet ({db_path} does not exist). Run the bot and study first."
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    lines: list[str] = []
    add = lines.append

    users = db.execute("SELECT * FROM users").fetchall()
    completions = db.execute("SELECT user_id, lesson_id FROM completions").fetchall()
    completed_by_lesson = Counter(r["lesson_id"] for r in completions)
    add(f"Learners: {len(users)}   lesson completions: {len(completions)}")
    add("")

    # Where learners are now ---------------------------------------------------------------
    add("WHERE LEARNERS ARE NOW")
    stalled: dict[str, list[str]] = defaultdict(list)
    for u in users:
        lesson = course.lessons.get(u["lesson_id"])
        total = len(lesson.steps) if lesson else 0
        idle_days = (time.time() - u["last_seen"]) / 86400
        stalled[u["lesson_id"]].append(
            f"step {max(0, u['cursor'] - 1)}/{total}, idle {idle_days:.1f}d"
        )
    for lesson_id in course.order:
        if lesson_id in stalled or completed_by_lesson[lesson_id]:
            where = "; ".join(stalled.get(lesson_id, [])) or "-"
            add(f"  {lesson_id:38s} completed {completed_by_lesson[lesson_id]:>3}   now: {where}")
    add("")

    # Quiz difficulty ----------------------------------------------------------------------
    rows = db.execute(
        "SELECT item_key, context, payload, chosen, correct FROM quiz_log WHERE chosen IS NOT NULL"
    ).fetchall()
    first: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        if json.loads(r["context"]).get("kind") in {"lesson", "testout"} and _split_key(
            r["item_key"]
        ):
            first[r["item_key"]].append(r)
    table = []
    for key, attempts in first.items():
        if len(attempts) < min_attempts:
            continue
        acc = sum(a["correct"] for a in attempts) / len(attempts)
        answer = QuizStep.model_validate(json.loads(attempts[0]["payload"])).answer
        wrong = Counter(a["chosen"] for a in attempts if not a["correct"])
        popular = ""
        if wrong:
            letter, count = wrong.most_common(1)[0]
            popular = f"most-picked wrong option {LETTERS[letter]} ({count}x; correct is {LETTERS[answer]})"
        table.append((acc, len(attempts), key, popular))
    table.sort()
    add(f"QUIZ STEPS BY FIRST-ATTEMPT ACCURACY (lowest first, min {min_attempts} attempts)")
    if not table:
        add("  no data yet")
    for acc, n, key, popular in table[:15]:
        lesson_id, item = _split_key(key)  # type: ignore[misc]
        add(f"  {acc:4.0%} of {n:<3} {_step_label(course, lesson_id, item)}")
        if popular:
            add(f"           {popular}")
    add("  (A step that everyone misses may have a wrong key or a misleading distractor;")
    add("   one that everyone gets right may be too easy. Check both.)")
    add("")

    # Open questions -----------------------------------------------------------------------
    scores: dict[str, list[int]] = defaultdict(list)
    for r in db.execute("SELECT item_key, score FROM think_log"):
        if _split_key(r["item_key"]):
            scores[r["item_key"]].append(r["score"])
    add("OPEN / INTERVIEW QUESTIONS BY AVERAGE SCORE (0-3, lowest first)")
    ranked = sorted(
        (sum(v) / len(v), len(v), k) for k, v in scores.items() if len(v) >= min_attempts
    )
    if not ranked:
        add("  no data yet")
    for avg, n, key in ranked[:10]:
        lesson_id, item = _split_key(key)  # type: ignore[misc]
        add(f"  {avg:3.1f} of {n:<3} {_step_label(course, lesson_id, item)}")
    add("")

    # Where learners ask for help ----------------------------------------------------------
    asked: Counter[tuple[str, int]] = Counter()
    for r in _rows(db, "SELECT lesson_id, item FROM questions"):
        asked[(r["lesson_id"], r["item"])] += 1
    add("STEPS THAT DREW QUESTIONS OR 'GO DEEPER' (most first)")
    if not asked:
        add("  no data yet (needs the AI tutor enabled)")
    for (lesson_id, item), n in asked.most_common(10):
        add(f"  {n:>3}x {_step_label(course, lesson_id, item)}")
    add("")

    # Flags --------------------------------------------------------------------------------
    flags: dict[tuple[str, int], list[sqlite3.Row]] = defaultdict(list)
    for r in _rows(db, "SELECT * FROM feedback ORDER BY created_at"):
        flags[(r["lesson_id"], r["item"])].append(r)
    add("FLAGGED STEPS (most flags first)")
    if not flags:
        add("  none")
    for (lesson_id, item), items in sorted(flags.items(), key=lambda kv: -len(kv[1])):
        reasons = Counter(f["reason"] for f in items)
        summary = ", ".join(f"{k} x{v}" for k, v in reasons.most_common())
        add(f"  {len(items):>3}x {_step_label(course, lesson_id, item)}")
        add(f"        {summary}")
        for f in items:
            if f["note"]:
                add(f'        note: "{f["note"][:200]}"')
    add("")

    # Backlog ------------------------------------------------------------------------------
    due, total = db.execute(
        "SELECT COALESCE(SUM(due_at <= ?), 0), COUNT(*) FROM reviews", (time.time(),)
    ).fetchone()
    open_later = db.execute("SELECT COUNT(*) FROM later WHERE done = 0").fetchone()[0]
    add(f"REVIEW BACKLOG: {due} due, {total} scheduled.   OPEN LAPTOP TASKS: {open_later}")
    db.close()
    return "\n".join(lines)
