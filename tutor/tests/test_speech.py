from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from clinical_tutor.content import (
    CodeStep,
    ImageStep,
    LaptopStep,
    QuizStep,
    RecapStep,
    TextStep,
    ThinkStep,
    load_course,
)
from clinical_tutor.speech import (
    lesson_parts,
    load_lexicon,
    script_seconds,
    speakable,
    step_script,
)
from clinical_tutor.tts import MacSay, make_tts

LEX = load_lexicon()
BENCH = Path(__file__).resolve().parents[1] / "tts_bench" / "cases.yaml"


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        (
            "SNOMED CT, LOINC, RxNorm, ICD-10-CM, CPT.",
            "snow med C T, L O I N C, R X norm, I C D ten C M, C P T.",
        ),
        ("adds I11–I13 and I15–I16.", "adds I eleven to I thirteen and I fifteen to I sixteen."),
        (
            "`ADT^A01` (admit) and `ORU^R01`",
            "A D T A zero one (admit) and O R U R zero one",
        ),
        (
            "history codes `Z86.73` and `I69.3-`",
            "history codes Z eighty-six point seven three and I sixty-nine point three dash",
        ),
        ("uses `70450` (CT head)", "uses seven zero four five zero (C T head)"),
        ("LOINC 2160-0", "L O I N C two one six zero dash zero"),
        ("`latest_ldl_prior_365d`", "latest L D L prior 365 days"),
        (
            "model A has AUROC 0.720 (95% CI 0.700–0.740)",
            "model A has A U ROC 0.720 (95 percent confidence interval 0.700 to 0.740)",
        ),
        ("from mg/dL to mmol/L", "from milligrams per deciliter to millimoles per liter"),
        ("0.95 x 0.05 = 0.50", "0.95 times 0.05 equals 0.50"),
        ("2019-01-01 to 2021-12-31", "January 1, 2019 to December 31, 2021"),
        ("version 1.3.1 (November 2024)", "version 1 point 3 point 1 (November 2024)"),
        ("see m01-05 and Module 04", "see module one, lesson five and Module four"),
        ("then HR 0.76 on zetagrel", "then hazard ratio 0.76 on zeta grel"),
        ("Costs $8.0 million or 2.0¢ each", "Costs 8.0 million dollars or 2.0 cents each"),
        ("p ≤ α/m", "p is at most alpha over m"),
        ("the check #5", "the check number 5"),
        ("nDCG@5", "N D C G at 5"),
        ("AUROCs and CIs", "A U ROCs and confidence intervals"),
        ("FROM → WHERE → GROUP BY", "from then where then group by"),
        ("`WHERE icd10_code LIKE 'I63%'`", "where I C D 10 code like I sixty-three wildcard"),
        ("`MAX(CASE WHEN cond THEN 1 ELSE 0 END)`", "max case when cond then 1 else 0 end"),
        ("`x NOT IN (a, b, NULL)`", "X not in A, B, null"),
        ("TPR/FPR trade off", "T P R slash F P R trade off"),
    ],
)
def test_hard_cases_are_rewritten_for_speech(written: str, spoken: str) -> None:
    assert speakable(written, LEX) == spoken


def test_markup_and_code_fences_never_reach_the_voice() -> None:
    out = speakable(
        "**Bold**, _italic_, a [link](https://x.org), 🚩 and\n```sql\nSELECT 1\n```", LEX
    )
    assert out == "Bold, italic, a link, and"


def test_wrapped_bullet_items_are_not_split() -> None:
    from clinical_tutor.speech import _paragraphs

    script = _paragraphs(
        "Steps:\n- **Clip** each gradient to a maximum norm, which\n  bounds influence.\n- **Add** noise.",
        LEX,
    )
    spoken = [x for x in script if isinstance(x, str)]
    assert spoken == [
        "Steps:",
        "Clip each gradient to a maximum norm, which bounds influence.",
        "Add noise.",
    ]


def test_speak_override_replaces_generated_narration() -> None:
    step = TextStep(
        type="text", text="Original with `code`.", speak="Say this instead, about AUROC."
    )
    assert [x for x in step_script(step, LEX) if isinstance(x, str)] == [
        "Say this instead, about A U ROC."
    ]


def test_step_scripts_cover_every_step_type() -> None:
    quiz = QuizStep(
        type="quiz",
        question="Pick one?",
        options=["Left", "Right"],
        answer=1,
        explanation="Because.",
    )
    spoken = " ".join(x for x in step_script(quiz, LEX) if isinstance(x, str))
    assert "The answer is B." in spoken and "A. Left." in spoken and "B. Right." in spoken
    assert any(isinstance(x, float) and x >= 5 for x in step_script(quiz, LEX))  # time to think

    think = ThinkStep(
        type="think", prompt="Why?", model_answer="Because.", key_points=["one"], interview=True
    )
    assert "Interview question." in step_script(think, LEX)[0]
    code = CodeStep(type="code", code="SELECT 1", text="Before.", after="After.")
    assert "A code example is shown in the text version." in step_script(code, LEX)
    assert "Figure." in step_script(ImageStep(type="image", image="x.png", caption="Cap."), LEX)
    assert step_script(RecapStep(type="recap", points=["Point."]), LEX)[0] == "Recap."
    assert "For your laptop: Do it." in step_script(
        LaptopStep(type="laptop", title="Do it", task="Now."), LEX
    )


def test_lesson_audio_is_split_into_parts_and_skips_laptop_tasks() -> None:
    course = load_course()
    lesson = course.lessons["m01-05-temporal-leakage"]
    parts = lesson_parts(lesson, LEX)
    assert 3 <= len(parts) <= 9
    text = " ".join(x for part in parts for x in part if isinstance(x, str))
    assert "For your laptop" not in text
    assert text.startswith("Temporal leakage and time-zero thinking")
    assert 10 <= sum(script_seconds(p) for p in parts) / 60 <= 40


def test_no_markup_or_symbols_survive_in_any_narration() -> None:
    """Scans every step of every lesson: nothing a voice would read as noise may remain."""
    forbidden = re.compile(r"[*`_|→≤≥≈≠±×·÷…−–—-^~@#<>{}\[\]\\]|\[\[")
    course = load_course()
    problems = []
    for lesson_id in course.order:
        for number, step in enumerate(course.lessons[lesson_id].steps, 1):
            script = step_script(step, LEX)
            if not any(isinstance(x, str) for x in script):
                problems.append(f"{lesson_id}#{number}: empty narration")
            for item in script:
                if isinstance(item, str) and (m := forbidden.search(item)):
                    problems.append(
                        f"{lesson_id}#{number}: {m.group(0)!r} in ...{item[max(0, m.start() - 30) : m.end() + 30]}"
                    )
    assert not problems, "\n".join(problems[:15])


def test_macos_say_script_rendering_blocks_embedded_commands() -> None:
    rendered = MacSay.render(["Hello [[volm 0]] there.", 1.5, "Done."])
    assert rendered == "Hello [volm 0] there. [[slnc 1500]] Done."


def test_make_tts_respects_none_and_unknown_backends() -> None:
    from clinical_tutor.tts import TTSError

    assert make_tts("none") is None
    with pytest.raises(TTSError, match="Unknown"):
        make_tts("espeak")


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say"), reason="needs macOS say")
async def test_macos_say_renders_a_playable_m4a(tmp_path: Path) -> None:
    out = tmp_path / "clip.m4a"
    await MacSay(None, 200).synthesize(["Testing one two three.", 0.3, "Done."], out)
    assert out.stat().st_size > 2000


# --- the challenge set used to choose a voice ---------------------------------------------------


def _step_texts(step) -> list[str]:
    if isinstance(step, TextStep):
        return [step.text]
    if isinstance(step, CodeStep):
        return [x for x in (step.text, step.after) if x]
    if isinstance(step, QuizStep):
        return [step.question, step.explanation, *step.options]
    if isinstance(step, ThinkStep):
        return [step.prompt, step.model_answer, *step.key_points]
    if isinstance(step, RecapStep):
        return list(step.points)
    if isinstance(step, ImageStep):
        return [step.caption]
    if isinstance(step, LaptopStep):
        return [step.task]
    return []


def test_challenge_set_is_real_lesson_text() -> None:
    cases = yaml.safe_load(BENCH.read_text(encoding="utf-8"))["cases"]
    course = load_course()
    assert len(cases) >= 50 and len({c["id"] for c in cases}) == len(cases)
    assert len({c["category"] for c in cases}) >= 8
    for case in cases:
        lesson_id, number = case["source"].split("#")
        step = course.lessons[lesson_id].steps[int(number) - 1]
        assert case["terms"], f"{case['id']} has no terms to check"
        assert case["why"], f"{case['id']} should say why it is hard"
        if case["kind"] == "text":
            assert any(case["text"] in t for t in _step_texts(step)), (
                f"{case['id']}: text is not verbatim in {case['source']}"
            )
        else:
            assert case["kind"] == "step" and step.type in {"quiz", "think", "recap"}


def test_lexicon_never_uses_a_lowercase_a_as_a_letter() -> None:
    """A lone lowercase 'a' is read as the article 'uh': 'S a M D' came out as 'S-uh-M-D'."""
    for term, spoken in LEX.terms.items():
        assert "a" not in spoken.split(), (
            f"{term!r}: {spoken!r} (use a capital A to spell the letter)"
        )
