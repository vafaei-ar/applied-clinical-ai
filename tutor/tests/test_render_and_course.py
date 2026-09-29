from __future__ import annotations

from clinical_tutor.content import load_course
from clinical_tutor.render import md, split_message
from clinical_tutor.steps import render_step


def test_markdown_subset():
    out = md("A **bold** and _italic_ with `a<b` and snake_case_name.\n\n- item one\n- item two")
    assert "<b>bold</b>" in out
    assert "<i>italic</i>" in out
    assert "<code>a&lt;b</code>" in out
    assert "snake_case_name" in out
    assert "• item one\n• item two" in out


def test_soft_wraps_join_but_lists_and_code_stay():
    out = md("one line\nwrapped here\n\n- a\n- b\n\n```sql\nSELECT 1\nFROM t\n```")
    assert "one line wrapped here" in out
    assert "• a\n• b" in out
    assert "SELECT 1\nFROM t" in out


def test_split_message_respects_limit():
    text = "\n\n".join("x" * 1000 for _ in range(10))
    chunks = split_message(text, limit=3900)
    assert all(len(c) <= 3900 for c in chunks)
    assert sum(c.count("x") for c in chunks) == 10_000


def test_real_course_is_valid_and_renders():
    course = load_course()
    assert course.order, "course has lessons"
    for lesson in course.lessons.values():
        for step in lesson.steps:
            for out in render_step(step):
                assert len(out.text) < 4096 or step.type in {"text", "code"}
