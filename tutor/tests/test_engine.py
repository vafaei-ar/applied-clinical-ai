from __future__ import annotations

import time

from clinical_tutor.store import DAY

from .conftest import FakeTutor

UID = 42


def _button_data(outs) -> list[str]:
    return [b.data for o in outs for row in o.buttons for b in row]


def _continue(outs) -> tuple[int, int]:
    data = next(d for d in _button_data(outs) if d.startswith("n:"))
    _, li, cursor = data.split(":")
    return int(li), int(cursor)


async def test_full_lesson_offline(make_engine, store):
    engine = make_engine()
    outs = await engine.start(UID, "Sam")
    assert "Welcome, Sam" in outs[0].text

    outs = await engine.advance(UID, *_continue(outs))  # header
    assert "Alpha" in outs[0].text and "Why it matters" in outs[0].text

    outs = await engine.advance(UID, *_continue(outs))  # text step
    assert "<b>bold</b>" in outs[0].text

    outs = await engine.advance(UID, *_continue(outs))  # quiz
    answer = next(d for d in _button_data(outs) if d.startswith("a:"))
    quiz_id = int(answer.split(":")[1])
    result = await engine.answer_quiz(UID, quiz_id, 0)  # wrong answer
    assert result[0].edit and "Not quite" in result[0].text
    due, total, _ = await store.review_counts(UID)
    assert (due, total) == (0, 1)  # scheduled for tomorrow
    assert await engine.answer_quiz(UID, quiz_id, 1) == []  # can't answer twice

    outs = await engine.advance(UID, *_continue(result))  # think step
    assert "Interview question" in outs[0].text
    outs = await engine.text(UID, "my answer")  # offline: shows model answer + self-rating
    assert "Model answer" in outs[0].text
    outs = await engine.self_rate(UID, 2)
    user = await store.get_user(UID)
    assert user.awaiting is None

    outs = await engine.advance(UID, *_continue(outs))  # laptop step
    assert "For your laptop" in outs[0].text
    assert len(await store.list_later(UID)) == 1

    outs = await engine.advance(UID, *_continue(outs))  # recap
    outs = await engine.advance(UID, *_continue(outs))  # completion
    assert "Lesson complete" in outs[0].text
    assert "m01-01-alpha" in await store.completed_lessons(UID)

    outs = await engine.advance(UID, *_continue(outs))  # seed lesson header (offline fallback)
    assert "Beta" in outs[0].text


async def test_stale_button_never_skips(make_engine):
    engine = make_engine()
    outs = await engine.start(UID, None)
    first = _continue(outs)
    await engine.advance(UID, *first)  # header shown, cursor -> 1
    outs = await engine.advance(UID, *first)  # same stale button again
    assert "already moved past" in outs[0].text
    # the learner sees the next item (step 1), not step 2
    assert "Hello" in outs[1].text


async def test_resume_reopens_pending_quiz(make_engine):
    engine = make_engine()
    outs = await engine.start(UID, None)
    outs = await engine.advance(UID, *_continue(outs))
    outs = await engine.advance(UID, *_continue(outs))
    outs = await engine.advance(UID, *_continue(outs))  # quiz pending
    outs = await engine.resume(UID)
    assert "still have this question" in outs[0].text
    assert any(d.startswith("a:") for d in _button_data(outs))


async def test_review_cycle(make_engine, store):
    engine = make_engine()
    await engine.start(UID, None)
    await store.schedule_review(
        UID,
        "k1",
        "quiz",
        {"type": "quiz", "question": "Q", "options": ["a", "b"], "answer": 0, "explanation": "e"},
        0,
    )
    await store.db.execute("UPDATE reviews SET due_at = ?", (time.time() - 1,))
    await store.db.commit()
    outs = await engine.review(UID)
    quiz_id = int(next(d for d in _button_data(outs) if d.startswith("a:")).split(":")[1])
    await engine.answer_quiz(UID, quiz_id, 0)
    item = await store.get_review(UID, "k1")
    assert item.box == 1 and item.due_at > time.time() + 2 * DAY


async def test_tutor_grading_and_questions(make_engine, store):
    tutor = FakeTutor()
    engine = make_engine(tutor)
    outs = await engine.start(UID, None)
    for _ in range(3):
        outs = await engine.advance(UID, *_continue(outs))
    quiz_id = int(next(d for d in _button_data(outs) if d.startswith("a:")).split(":")[1])
    outs = await engine.answer_quiz(UID, quiz_id, 1)
    outs = await engine.advance(UID, *_continue(outs))  # think
    outs = await engine.text(UID, "a wrong answer")
    assert "●○○" in outs[0].text and "spaced review" in outs[0].text

    outs = await engine.text(UID, "What is alpha?")
    assert "Answer to: What is alpha?" in outs[0].text
    assert len(await store.recent_chat(UID)) == 2

    outs = await engine.more_questions(UID)
    assert "Fresh question" in outs[0].text

    outs = await engine.interview(UID)
    assert "Mock interview" in outs[0].text


async def test_seed_lesson_is_generated_and_cached(make_engine, store, tmp_path):
    tutor = FakeTutor()
    engine = make_engine(tutor)
    await engine.start(UID, None)
    notes: list[str] = []

    async def notify(text: str) -> None:
        notes.append(text)

    outs = await engine.jump(UID, 1, notify)
    assert "Beta" in outs[0].text and notes
    lesson = await engine.lesson("m01-02-beta")
    assert len(lesson.steps) == 6  # the malformed quiz was dropped
    assert (tmp_path / "data" / "generated" / "m01-02-beta.yaml").is_file()
    await engine.lesson("m01-02-beta")
    assert tutor.calls.count("write_lesson") == 1


async def test_nudge_once_per_quiet_period(make_engine, store):
    engine = make_engine()
    await engine.start(UID, None)
    await store.db.execute("UPDATE users SET last_seen = ?", (time.time() - 2 * DAY,))
    await store.db.commit()
    first = await engine.nudges()
    assert len(first) == 1 and first[0][0] == UID
    assert await engine.nudges() == []
