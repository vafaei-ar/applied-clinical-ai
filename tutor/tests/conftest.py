from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from clinical_tutor.content import load_course
from clinical_tutor.engine import Engine
from clinical_tutor.llm import Grade
from clinical_tutor.store import Store

LESSON_A = {
    "id": "m01-01-alpha",
    "title": "Alpha",
    "summary": "First lesson.",
    "why_it_matters": "Because.",
    "objectives": ["Learn alpha"],
    "steps": [
        {"type": "text", "title": "Hello", "text": "Some **bold** text."},
        {
            "type": "quiz",
            "question": "Pick B",
            "options": ["A", "B", "C"],
            "answer": 1,
            "explanation": "B is right.",
        },
        {
            "type": "think",
            "interview": True,
            "prompt": "Explain alpha.",
            "model_answer": "Alpha is first.",
            "key_points": ["first"],
        },
        {"type": "laptop", "title": "Do alpha", "task": "Write code.", "repo_path": "x.py"},
        {"type": "recap", "points": ["done"]},
    ],
}

LESSON_B = {
    "id": "m01-02-beta",
    "title": "Beta",
    "summary": "Seed lesson.",
    "status": "seed",
    "why_it_matters": "Because.",
    "objectives": ["Learn beta"],
    "key_points": ["beta point"],
}


def _quiz(question: str, answer: int) -> dict:
    return {
        "type": "quiz",
        "question": question,
        "options": ["A-opt", "B-opt", "C-opt"],
        "answer": answer,
        "explanation": "Because.",
    }


LESSON_C = {
    "id": "m01-03-gamma",
    "title": "Gamma",
    "summary": "Has three quizzes so it can be tested out of.",
    "why_it_matters": "Because.",
    "objectives": ["Learn gamma"],
    "steps": [
        {"type": "text", "text": "Intro."},
        _quiz("Q1", 0),
        _quiz("Q2", 1),
        {"type": "text", "text": "More."},
        _quiz("Q3", 2),
        {"type": "recap", "points": ["done"]},
    ],
}


@pytest.fixture
def course_dir(tmp_path: Path) -> Path:
    root = tmp_path / "course"
    (root / "m01").mkdir(parents=True)
    (root / "images").mkdir()
    (root / "course.yaml").write_text(
        yaml.safe_dump(
            {
                "title": "Test course",
                "description": "A tiny course.",
                "modules": [
                    {
                        "id": "m01",
                        "title": "Module one",
                        "summary": "Summary.",
                        "lessons": ["m01-01-alpha", "m01-02-beta", "m01-03-gamma"],
                    }
                ],
            }
        )
    )
    (root / "m01" / "m01-01-alpha.yaml").write_text(yaml.safe_dump(LESSON_A))
    (root / "m01" / "m01-02-beta.yaml").write_text(yaml.safe_dump(LESSON_B))
    (root / "m01" / "m01-03-gamma.yaml").write_text(yaml.safe_dump(LESSON_C))
    (root / "cases.yaml").write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "id": "test-case",
                        "title": "A test case",
                        "scenario": "Something went wrong.",
                        "question": "What do you check?",
                        "model_answer": "Check the data first.",
                        "key_points": ["data", "labels"],
                        "lessons": ["m01-01-alpha", "m01-03-gamma"],
                    }
                ]
            }
        )
    )
    return root


class FakeTutor:
    """Deterministic stand-in for the AI tutor."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def answer(self, context: str, history: list[dict[str, str]], question: str) -> str:
        self.calls.append("answer")
        return f"Answer to: {question}"

    async def deepen(self, context: str) -> str:
        self.calls.append("deepen")
        return "Deeper insight."

    async def welcome_back(self, context: str) -> str:
        return "You were learning alpha."

    async def grade(
        self, prompt: str, key_points: list[str], model_answer: str, answer: str
    ) -> Grade:
        self.calls.append("grade")
        return Grade(score=1 if "wrong" in answer else 3, feedback="Feedback.")

    async def make_quiz(
        self, context: str, n: int = 3, avoid: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return [
            {
                "question": "Gen Q",
                "code": "",
                "options": ["x", "y"],
                "answer": 0,
                "explanation": "x.",
            }
        ]

    async def interview_question(
        self, context: str, avoid: list[str] | None = None
    ) -> dict[str, Any]:
        return {"prompt": "Gen interview?", "model_answer": "Answer.", "key_points": ["k"]}

    async def write_lesson(self, brief: str, guide: str) -> list[dict[str, Any]]:
        self.calls.append("write_lesson")
        steps: list[dict[str, Any]] = [
            {"type": "text", "title": "", "text": f"Generated {i}"} for i in range(5)
        ]
        steps.append({"type": "recap", "points": ["generated"]})
        steps.append(
            {
                "type": "quiz",
                "question": "bad",
                "code": "",
                "language": "",
                "options": ["only"],
                "answer": 3,
                "explanation": "",
            }
        )
        return steps


@pytest.fixture
async def store(tmp_path: Path):
    s = await Store(tmp_path / "db.sqlite3").open()
    yield s
    await s.close()


class FakeAudio:
    """Stands in for AudioCache: writes a tiny file per script and remembers what it was asked."""

    def __init__(self, directory: Path, fail_with: Exception | None = None) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scripts: list[list] = []
        self.fail_with = fail_with

    def cached(self, script: list) -> bool:
        return script in self.scripts

    async def get(self, script: list) -> Path:
        if self.fail_with is not None:
            raise self.fail_with
        if script not in self.scripts:
            self.scripts.append(script)
        path = self.directory / f"clip{self.scripts.index(script)}.m4a"
        path.write_bytes(b"fake audio")
        return path


@pytest.fixture
def make_engine(course_dir: Path, store: Store, tmp_path: Path):
    def _make(tutor: Any = None, audio: Any = None) -> Engine:
        return Engine(load_course(course_dir), store, tutor, tmp_path / "data", audio=audio)

    return _make
