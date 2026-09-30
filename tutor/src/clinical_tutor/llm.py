"""The AI tutor: answers questions, goes deeper, grades answers, and writes new material.

Everything here is optional. The bot works without it, using only authored content.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import anthropic

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

STYLE_RULES = """\
Formatting (the reply is shown in Telegram on a phone):
- Use only this Markdown subset: **bold**, _italic_, `inline code`, fenced ``` code blocks, and "- " bullets.
- No headings, no tables, no HTML, no horizontal rules.
- Short paragraphs separated by blank lines. Keep code snippets under ~20 lines.
"""

TUTOR_SYSTEM = (
    """\
You are the tutor inside "Applied Clinical AI", an open-source, phone-first course delivered through
Telegram. The learner is a researcher with a quantitative background who is building the technical
skills for clinical AI roles: clinical data engineering and SQL, clinical ML with PyTorch,
production ML, clinical trials and survival analysis, clinical LLM evaluation, healthcare standards.

The course follows one running project: a synthetic stroke cohort (repo folder
01-clinical-data-engineering/, tables patients, coverage, encounters, diagnoses, procedures,
medications, labs, claims), a 30-day readmission model built on it, that model shipped as a service,
a synthetic secondary-stroke-prevention trial, and an LLM assistant over stroke guidelines.

How to teach:
- Be a sharp senior colleague: precise, practical, warm, never condescending.
- Focus on what matters in real projects and in interviews: traps, trade-offs, vocabulary, judgment.
- Tie answers to the learner's current lesson and to the running project when it helps.
- Be accurate. If something is uncertain or depends on context, say so briefly. Never invent
  citations, statistics, or URLs.
- Default to ~120-250 words. Go longer only when asked or when a worked example truly needs it.
- End with a short question or next step only when it adds value, not by reflex.

"""
    + STYLE_RULES
)


@dataclass
class Grade:
    score: int  # 0 = missed, 1 = partial, 2 = good, 3 = excellent
    feedback: str


class TutorUnavailable(Exception):
    """Raised when the model call fails in a way the learner should hear about."""


_QUIZ_ITEM = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "code": {
            "type": "string",
            "description": "optional snippet shown with the question, or ''",
        },
        "options": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "integer", "description": "0-based index of the correct option"},
        "explanation": {"type": "string"},
    },
    "required": ["question", "code", "options", "answer", "explanation"],
    "additionalProperties": False,
}

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": _QUIZ_ITEM}},
    "required": ["questions"],
    "additionalProperties": False,
}

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "0 missed, 1 partial, 2 good, 3 excellent"},
        "feedback": {"type": "string"},
    },
    "required": ["score", "feedback"],
    "additionalProperties": False,
}

THINK_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "model_answer": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prompt", "model_answer", "key_points"],
    "additionalProperties": False,
}


def _obj(kind: str, **props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"type": {"const": kind}, **props},
        "required": ["type", *props],
        "additionalProperties": False,
    }


_S = {"type": "string"}
_SL = {"type": "array", "items": {"type": "string"}}

LESSON_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "anyOf": [
                    _obj("text", title=_S, text=_S),
                    _obj("code", title=_S, text=_S, language=_S, code=_S, after=_S),
                    _obj(
                        "quiz",
                        question=_S,
                        code=_S,
                        language=_S,
                        options=_SL,
                        answer={"type": "integer"},
                        explanation=_S,
                    ),
                    _obj(
                        "think",
                        prompt=_S,
                        model_answer=_S,
                        key_points=_SL,
                        interview={"type": "boolean"},
                    ),
                    _obj("laptop", title=_S, task=_S, repo_path=_S),
                    _obj("recap", points=_SL),
                ]
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


PROFILE_INSTRUCTIONS = """
<learner_profile_usage>
The learner has shared a private profile of their background (below). Use it to calibrate:
- Skip explanations of things listed under strengths; go straight to trade-offs, pitfalls, and
  what is different in clinical or industry settings.
- Bridge new ideas to their experience where the analogy is genuinely accurate; drop it if it
  isn't. Don't flatter, and don't recite their background back to them.
- Spend the time on the "less practiced" areas and on interview-style framing.
- Treat the "less practiced" list as a hypothesis: if they show fluency, update accordingly.
</learner_profile_usage>
"""

MAX_PROFILE_CHARS = 6000


class Tutor:
    def __init__(
        self,
        api_key: str,
        model: str,
        effort: str = "medium",
        language: str = "English",
        profile: str | None = None,
    ):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.effort = effort
        self.language = language
        self.profile = (profile or "").strip()[:MAX_PROFILE_CHARS] or None

    def _system(self) -> str:
        system = TUTOR_SYSTEM
        if self.language.lower() != "english":
            system += f"\nReply in {self.language}. Keep technical terms, code, and SQL in English."
        if self.profile:
            system += (
                f"{PROFILE_INSTRUCTIONS}\n<learner_profile>\n{self.profile}\n</learner_profile>\n"
            )
        return system

    async def _call(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4000,
        schema: dict[str, Any] | None = None,
        effort: str | None = None,
    ) -> str:
        output_config: dict[str, Any] = {"effort": effort or self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        try:
            # Streaming avoids HTTP timeouts on long generations (whole lessons).
            async with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=self._system(),
                messages=messages,
                output_config=output_config,
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                response = await stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise TutorUnavailable(
                "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise TutorUnavailable(
                "The tutor is rate-limited right now. Try again in a minute."
            ) from exc
        except anthropic.APIStatusError as exc:
            log.warning("tutor call failed: %s %s", exc.status_code, exc.message)
            raise TutorUnavailable("The tutor had a problem answering. Try again shortly.") from exc
        except anthropic.APIConnectionError as exc:
            raise TutorUnavailable("Couldn't reach the tutor service. Check the network.") from exc

        if response.stop_reason == "refusal":
            raise TutorUnavailable("The tutor declined to answer that one.")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise TutorUnavailable("The tutor returned an empty answer. Try rephrasing.")
        return text

    async def _json(self, prompt: str, schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        text = await self._call([{"role": "user", "content": prompt}], schema=schema, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise TutorUnavailable("The tutor returned malformed output. Try again.") from exc

    # --- conversation -------------------------------------------------------------------------

    async def answer(self, context: str, history: list[dict[str, str]], question: str) -> str:
        """Answer a free-form question in the context of the learner's current position."""
        messages: list[dict[str, Any]] = []
        for turn in history:
            if messages and messages[-1]["role"] == turn["role"]:
                messages[-1]["content"] += "\n\n" + turn["content"]
            else:
                messages.append({"role": turn["role"], "content": turn["content"]})
        if messages and messages[0]["role"] == "assistant":
            messages.insert(0, {"role": "user", "content": "(earlier conversation)"})
        messages.append(
            {
                "role": "user",
                "content": f"<where_i_am>\n{context}\n</where_i_am>\n\n{question}",
            }
        )
        return await self._call(messages, max_tokens=3000)

    async def deepen(self, context: str) -> str:
        prompt = (
            f"<where_i_am>\n{context}\n</where_i_am>\n\n"
            "Go one level deeper on the step I just read. Give me what a senior practitioner "
            "knows that the step left out: a subtlety, a real-world failure, a concrete worked "
            "example, or how this comes up in interviews. Don't repeat the step. ~200-300 words."
        )
        return await self._call([{"role": "user", "content": prompt}], max_tokens=3000)

    async def welcome_back(self, context: str) -> str:
        prompt = (
            f"<where_i_am>\n{context}\n</where_i_am>\n\n"
            "I'm coming back to the course after a break. In 2-4 short sentences, remind me what "
            "I was learning and the key idea just before where I stopped, so I can pick up "
            "smoothly. No greeting, no question at the end."
        )
        return await self._call([{"role": "user", "content": prompt}], max_tokens=800, effort="low")

    # --- assessment ---------------------------------------------------------------------------

    async def grade(
        self, prompt: str, key_points: list[str], model_answer: str, answer: str
    ) -> Grade:
        request = (
            "Grade a learner's answer to an open question.\n\n"
            f"<question>\n{prompt}\n</question>\n\n"
            f"<key_points>\n" + "\n".join(f"- {p}" for p in key_points) + "\n</key_points>\n\n"
            f"<reference_answer>\n{model_answer}\n</reference_answer>\n\n"
            f"<learner_answer>\n{answer}\n</learner_answer>\n\n"
            "Score 0-3: 0 = misses the core idea or is wrong, 1 = partially right with important "
            "gaps, 2 = solid and covers most key points, 3 = excellent, interview-ready. Judge "
            "substance, not wording or length; credit correct points that aren't in the key "
            "points. In `feedback` (80-160 words, same Markdown subset), say what was strong, "
            "then exactly what was missing or wrong and how a strong candidate would phrase it."
        )
        data = await self._json(request, GRADE_SCHEMA, max_tokens=2000)
        return Grade(score=max(0, min(3, int(data["score"]))), feedback=data["feedback"])

    async def make_quiz(
        self, context: str, n: int = 3, avoid: list[str] | None = None
    ) -> list[dict[str, Any]]:
        avoid_text = ""
        if avoid:
            avoid_text = "\n\nDon't repeat these existing questions:\n" + "\n".join(
                f"- {q}" for q in avoid
            )
        request = (
            f"<where_i_am>\n{context}\n</where_i_am>\n\n"
            f"Write {n} new multiple-choice questions that test real understanding of this "
            "lesson. They should test application and judgment, not recall. Use code-reading "
            "questions where the topic has code. Use 4 options each, with plausible distractors "
            "built from real misconceptions, and vary which option is correct. Keep options "
            "under ~120 characters. The explanation says why the answer is right and why the "
            "most tempting wrong option is wrong (under ~500 characters). Use '' for code when "
            "there is no snippet." + avoid_text
        )
        data = await self._json(request, QUIZ_SCHEMA, max_tokens=6000)
        return data["questions"]

    async def interview_question(
        self, context: str, avoid: list[str] | None = None
    ) -> dict[str, Any]:
        avoid_text = ""
        if avoid:
            avoid_text = "\n\nAvoid these questions I've already practised:\n" + "\n".join(
                f"- {q}" for q in avoid[-20:]
            )
        request = (
            f"<topics_i_have_covered>\n{context}\n</topics_i_have_covered>\n\n"
            "Ask me one realistic technical interview question for a clinical AI role, drawn from "
            "these topics. Mix concept, judgment, and design; phrase it the way an interviewer "
            "would. Provide a model answer a strong senior candidate would give (150-250 words) "
            "and 3-5 key points a grader should look for." + avoid_text
        )
        return await self._json(request, THINK_SCHEMA, max_tokens=4000)

    # --- content generation -------------------------------------------------------------------

    async def write_lesson(self, brief: str, guide: str) -> list[dict[str, Any]]:
        request = (
            "Write a complete lesson for the course as a list of steps.\n\n"
            f"<authoring_guide>\n{guide}\n</authoring_guide>\n\n"
            f"<lesson_brief>\n{brief}\n</lesson_brief>\n\n"
            "Follow the authoring guide's pedagogy and step mix: 18-26 steps, a concrete hook "
            "first, one idea per text step (under 1800 characters each), frequent quizzes with "
            "4 options, at least two `think` steps with interview=true, optionally one laptop "
            "task, and end with a recap. For fields that don't apply, use '' (for example `title` "
            "on an untitled text step, `code` on a quiz without a snippet, `repo_path` when there "
            "is none). `language` is the code language such as 'sql' or 'python'."
        )
        data = await self._json(request, LESSON_SCHEMA, max_tokens=32000, effort="high")
        return data["steps"]
