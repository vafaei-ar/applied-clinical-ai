# Writing lessons for the tutor

Every lesson is one YAML file in `course/<module>/<lesson-id>.yaml`. The bot reads it step by step
on a phone. Anyone can improve a lesson with a pull request; run `clinical-tutor validate` first.

## What a great lesson does

The learner is a researcher who already has a quantitative background and wants to become
genuinely employable in clinical AI roles. They read on a phone in spare moments, so:

1. **Lead with the why.** Open with a concrete situation (a failed model, a bad cohort, an
   interview question) that makes the concept feel necessary before it is explained.
2. **Teach the 20% that matters.** Focus on what shows up in real projects and interviews:
   the traps, the trade-offs, the vocabulary a senior colleague would use. Skip trivia.
3. **Use the repo.** Module 01's synthetic stroke dataset (`patients`, `coverage`, `encounters`,
   `diagnoses`, `procedures`, `medications`, `labs`, `claims`) is the running example. Later
   modules build on it: the stroke cohort feeds the readmission model, the model becomes a
   service, and so on. Connect lessons to each other explicitly.
4. **Make the learner work.** A step every 1–3 screens should ask for effort: a quiz,
   a code-reading question ("what does this return?", "where is the bug?"), or an open question.
5. **Interview-ready.** Each lesson includes at least two `think` steps with `interview: true`,
   phrased the way an interviewer would ask them, with a model answer a strong candidate would give.
6. **Honest and precise.** Clinical and statistical claims must be correct. When something is a
   simplification, say so. Cite well-known sources by name (for example, "Obermeyer et al., Science
   2019") rather than inventing URLs.
7. **Phone-sized.** One idea per `text` step. Short paragraphs. Code snippets under ~25 lines.

A lesson typically has 18–30 steps and takes 20–30 minutes. Aim for roughly: 8–12 `text`,
2–5 `code`, 4–7 `quiz`, 2–4 `think`, 0–2 `image`, 0–2 `laptop`, 1 `recap` at the end.

## File format

```yaml
id: m01-05-temporal-leakage        # must equal the file name
title: Temporal leakage and time-zero thinking
summary: One sentence shown in the lesson header.
minutes: 25
status: authored                    # "seed" = not written yet, generated at runtime
why_it_matters: Two sentences on career/project relevance, shown in the header.
objectives:
  - What the learner will be able to do
key_points:                         # optional for authored lessons; the tutor uses them as context
  - Core idea
tags: [leakage, interview]
steps:
  - type: text
    title: Optional bold title
    text: |
      Paragraphs in light Markdown.
```

### Step types

| type | fields | how the bot uses it |
|---|---|---|
| `text` | `text`, optional `title` | a message with **Continue** and **Go deeper** buttons |
| `code` | `code`, `language` (default `sql`), optional `title`, `text` (before), `after` | a message with a code block |
| `image` | `image` (file name in `course/images/`), `caption` | a photo |
| `quiz` | `question`, optional `code` + `language`, `options` (2–6), `answer` (0-based index), `explanation` | lettered buttons; the message is edited to show the result and explanation; misses go to spaced review |
| `think` | `prompt`, `model_answer`, `key_points`, `interview` (bool) | the learner types an answer; the tutor grades it against `key_points` (or shows the model answer offline); interview questions enter spaced review |
| `laptop` | `title`, `task`, optional `repo_path` | a hands-on task saved to the learner's `/later` list |
| `recap` | `points` | bullet summary; put one at the end |

### Formatting

Text fields use a small Markdown subset: `**bold**`, `_italic_`, `` `inline code` ``,
fenced code blocks, `[links](https://...)`, and `- ` bullets. No headings, tables, or HTML.
Emoji are welcome in moderation.

Single line breaks inside a paragraph are joined (as in Markdown), so wrap YAML text freely.
Leave a blank line between paragraphs and before a bullet list. A wrapped line that starts with
`- `, `* `, or a number followed by `.` or `)` (such as `02)`) is treated as a list item and keeps
its line break, so rewrap to avoid starting a line that way.

Limits (enforced by the validator): `text` ≤ 3200 characters, `code` ≤ 2800, image
`caption` ≤ 900, quiz options ≤ 300 each. Keep quiz options short enough to read on a phone.

### YAML pitfalls

- Use block scalars (`text: |`) for anything longer than a few words. Inside a block scalar,
  any characters are safe.
- A plain one-line value containing `: ` or ending in `:` must be quoted:
  `text: "From sql/02_features.sql:"`. This includes `title:` values, quiz options, and recap
  points (for example an option that starts "Correct: ..."). The validator reports these as
  "mapping values are not allowed here" or "list item ... is not text; quote it".
- A wrapped line that starts with a number and a period or bracket (such as `40. With ...` or
  `02)`) is read as a list item and keeps its line break; rewrap to avoid starting a line that way.
- A list item that starts with `*`, `` ` ``, `&`, `!`, `%`, `@`, `[`, `{`, or `"` must be quoted:
  `- "**Bold** start of a recap point"`.

### Quizzes that teach

A lesson's quiz steps double as its **test-out**: learners can tap "Test me first" and skip the
lesson by answering 4 to 5 of them well. So the quizzes as a set should cover the lesson's key
ideas, and include some genuinely hard ones. A lesson needs at least two quizzes to offer test-out.

- Distractors should be the mistakes people actually make, not obviously silly options.
- The `explanation` explains why the right answer is right **and** why the tempting wrong one is wrong.
- Vary the position of the correct answer.
- Code-reading quizzes are excellent: show 5–15 lines and ask what they return or what is wrong.

### Figures

Figures are generated by code so they stay reproducible and editable. Each lesson has its own
figure file, `figures/<lesson id with - replaced by _>.py` (for example
`figures/m01_05_temporal_leakage.py`). Each `fig_*` function in it draws with the shared style in
`figures/_style.py` and saves to `course/images/<lesson-id-prefix>-<name>.png` (for example
`m01-05-leakage-timeline.png`). Render with:

```bash
python figures/render_all.py                          # all figures
python figures/render_all.py m01_05_temporal_leakage  # one lesson's figures
```

Design figures for a phone: one message per figure, large fonts, few labels, high contrast,
white background. Timelines, curves (ROC, calibration, Kaplan–Meier), and architecture sketches
work well. Reference the file with an `image` step.

## Cases of the day

`course/cases.yaml` holds scenarios that make the learner combine several lessons, served by
`/case`. Each case has an `id`, a `title`, a short `scenario` (up to 1000 characters), the
`question` (up to 300), a `model_answer`, at least two `key_points` a grader looks for, and the
`lessons` it draws on (a case unlocks once at least half of them are covered). Good cases read like
something that actually happened: a metric moved, a pipeline broke, a stakeholder asked a hard
question. The strongest model answers give an ordered plan ("first check X, because it would show
Y"), not a list of facts. The validator checks that every listed lesson exists.

## Workflow

```bash
clinical-tutor validate              # schema, limits, missing images, course.yaml consistency
clinical-tutor outline               # course outline with step counts
clinical-tutor preview m01-05-temporal-leakage   # print a lesson as Telegram HTML
```

`course/course.yaml` controls module and lesson order. A lesson file that is not listed there is
reported as an error.
