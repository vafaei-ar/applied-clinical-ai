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


async def test_flag_and_report_flow(make_engine, store):
    engine = make_engine()
    outs = await engine.start(UID, None)
    outs = await engine.advance(UID, *_continue(outs))  # header
    outs = await engine.advance(UID, *_continue(outs))  # step 1 (text)
    flag = next(d for d in _button_data(outs) if d.startswith("f:"))
    _, li, item = flag.split(":")
    menu = await engine.flag_menu(UID, int(li), int(item))
    reasons = [b.data for row in menu[0].buttons for b in row if b.data.startswith("fr:")]
    assert len(reasons) == 5
    _, _, _, code = reasons[1].split(":")
    done = await engine.flag(UID, int(li), int(item), int(code))
    assert done[0].edit and "Saved" in done[0].text

    empty = await engine.report(UID, "  ")
    assert "/report" in empty[0].text
    await engine.report(UID, "the wording here is confusing")
    rows = await (
        await store.db.execute("SELECT lesson_id, item, reason, note FROM feedback")
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        ("m01-01-alpha", 1, "wrong", None),
        ("m01-01-alpha", 1, "note", "the wording here is confusing"),
    ]
    assert await engine.flag(UID, 99, 1, 0) == []  # bad lesson index is ignored


async def test_quiz_result_offers_flag_on_the_quiz_step(make_engine):
    engine = make_engine()
    outs = await engine.start(UID, None)
    for _ in range(3):
        outs = await engine.advance(UID, *_continue(outs))  # header, text, quiz
    quiz_id = int(next(d for d in _button_data(outs) if d.startswith("a:")).split(":")[1])
    result = await engine.answer_quiz(UID, quiz_id, 1)
    flag = next(d for d in _button_data(result) if d.startswith("f:"))
    assert flag == "f:0:2"  # lesson 0, step 2 (the quiz)


async def test_stats_report(make_engine, store, tmp_path):
    from clinical_tutor.stats import build_report

    tutor = FakeTutor()
    engine = make_engine(tutor)
    outs = await engine.start(UID, None)
    for _ in range(3):
        outs = await engine.advance(UID, *_continue(outs))
    quiz_id = int(next(d for d in _button_data(outs) if d.startswith("a:")).split(":")[1])
    await engine.answer_quiz(UID, quiz_id, 2)  # wrong: picks C
    await engine.text(UID, "why is B right?")
    await engine.report(UID, "explanation is thin")

    report = build_report(engine.course, tmp_path / "db.sqlite3")
    assert "Learners: 1" in report
    assert "m01-01-alpha#2 [quiz] Pick B" in report
    assert "most-picked wrong option C (1x; correct is B)" in report
    assert "1x m01-01-alpha#2" in report  # the question is logged against the step being read
    assert 'note: "explanation is thin"' in report
    assert "no learner data" in build_report(engine.course, tmp_path / "missing.sqlite3").lower()


async def test_stats_tolerates_database_from_before_feedback_tables(make_engine, tmp_path):
    import sqlite3

    from clinical_tutor.stats import build_report

    engine = make_engine()
    await engine.start(UID, None)
    db = sqlite3.connect(tmp_path / "db.sqlite3")
    db.executescript("DROP TABLE feedback; DROP TABLE questions;")
    db.close()
    report = build_report(engine.course, tmp_path / "db.sqlite3")
    assert "FLAGGED STEPS" in report and "none" in report


async def _at_gamma_header(engine):
    outs = await engine.jump(UID, 2)  # lesson index 2 = gamma; shows its header
    return outs


async def test_header_offers_test_out_only_with_enough_quizzes(make_engine):
    engine = make_engine()
    await engine.start(UID, None)
    gamma = await _at_gamma_header(engine)
    assert "tt:2" in _button_data(gamma)
    alpha = await engine.jump(UID, 0)  # only one quiz
    assert not any(d.startswith("tt:") for d in _button_data(alpha))


async def _answer_testout(engine, outs, chooser):
    """Answer each test-out question; returns the outputs after the final 'See result'."""
    while True:
        datas = _button_data(outs)
        answers = [d for d in datas if d.startswith("a:")]
        assert answers, datas
        qid = int(answers[0].split(":")[1])
        record = await engine.store.get_quiz(qid)
        result = await engine.answer_quiz(UID, qid, chooser(record.payload["answer"]))
        outs = await engine.test_next(UID, qid)
        assert result[0].edit
        if not any(d.startswith("a:") for d in _button_data(outs)):
            return outs


async def test_test_out_pass_completes_the_lesson(make_engine, store):
    engine = make_engine()
    await engine.start(UID, None)
    await _at_gamma_header(engine)
    outs = await engine.test_out(UID, 2)
    assert "Test-out</b> 1/3" in outs[0].text
    final = await _answer_testout(engine, outs, lambda correct: correct)
    assert "Tested out" in final[0].text and "3/3" in final[0].text
    assert "m01-03-gamma" in await store.completed_lessons(UID)
    assert await store.review_counts(UID) == (0, 0, None)  # nothing missed


async def test_test_out_fail_sends_learner_to_the_lesson(make_engine, store):
    engine = make_engine()
    await engine.start(UID, None)
    await _at_gamma_header(engine)
    outs = await engine.test_out(UID, 2)
    final = await _answer_testout(engine, outs, lambda correct: (correct + 1) % 3)
    assert "0/3" in final[0].text and "worth doing" in final[0].text
    assert "m01-03-gamma" not in await store.completed_lessons(UID)
    _, total, _ = await store.review_counts(UID)
    assert total == 3  # every miss is queued for spaced review
    start = next(d for d in _button_data(final) if d.startswith("n:"))
    outs = await engine.advance(UID, *map(int, start.split(":")[1:]))
    assert "Intro." in outs[0].text  # the full lesson starts from step 1


async def test_test_out_is_ignored_when_stale(make_engine):
    engine = make_engine()
    await engine.start(UID, None)
    await _at_gamma_header(engine)
    outs = await engine.test_out(UID, 0)  # a card for a different lesson
    assert "out of date" in outs[0].text
