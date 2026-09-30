# Applied Clinical AI: the Telegram tutor

Learn the whole course from your phone. The tutor walks you through one continuous path of
lessons, quizzes, figures, code-reading exercises, and interview questions, and it remembers
exactly where you stopped. Hands-on coding tasks are saved for when you're at a laptop.

The course content lives in [`course/`](course/) as plain YAML, so anyone can read it, fix it,
or add to it. Clone the repo, add your own Telegram bot token, and you have your own tutor.

## What it does

- **A continuous path.** 8 modules, 41 lessons, from clinical data engineering and SQL to
  PyTorch, production ML, trials and survival analysis, LLM evaluation, standards and privacy,
  and system-design interviews. Tap **Continue** to move on and stop whenever you like.
  `/continue` picks up where you left off, with a short recap if you've been away.
- **Active learning.** Quick-check quizzes (with explanations), code-reading questions, and open
  "think it through" and interview questions that you answer in your own words.
- **Spaced review.** Anything you miss comes back after 1, 3, 7, and 21 days (`/review`).
  Interview questions come back even when you get them right.
- **Test out of what you know.** Every lesson header has **⚡ Test me first**: 4 to 5 of the
  lesson's hardest quiz questions. Pass (at most one miss) and the lesson is marked complete;
  miss and the questions are queued for review and you start the lesson.
- **Laptop queue.** Coding tasks that need a computer land in `/later` with the repo path.
- **Mock interviews.** `/interview` asks questions from the lessons you've covered.
- **A daily nudge** (optional) if you've been away for a day.

With an Anthropic API key, the **AI tutor** also turns on:

- Type any question at any point and get an answer in the context of the step you're on.
- **🔍 Go deeper** on any step for the subtlety a senior practitioner would add.
- Your typed answers to open and interview questions are **graded** against the key points, with
  feedback.
- **🧪 Quiz me** / `/quiz` writes fresh questions on the current lesson.
- `/interview` can invent new questions on the topics you've covered.
- Lessons that haven't been written yet (`status: seed`) are **drafted automatically** the first
  time you reach them and cached in `data/generated/`.

**Personalize it.** Put a short description of your background in `data/profile.md` (strengths to
treat as known, areas you want to build, how you like to learn). The AI tutor reads it to skip
what you already know and bridge from your experience. That folder is gitignored, so it stays on
your computer.

Without a key, everything authored still works. Open questions show a model answer and you rate
yourself.

## Quick start (about 5 minutes)

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

1. **Create a bot.** In Telegram, message [@BotFather](https://t.me/BotFather), send `/newbot`,
   and copy the token it gives you.
2. **Find your user ID.** Message [@userinfobot](https://t.me/userinfobot) and copy the number.
3. **Install:**

   ```bash
   git clone https://github.com/vafaei-ar/applied-clinical-ai.git
   cd applied-clinical-ai/tutor
   uv venv && uv pip install -e .
   cp .env.example .env    # then edit .env: token, your user ID, optional ANTHROPIC_API_KEY
   ```

4. **Run:**

   ```bash
   .venv/bin/clinical-tutor run
   ```

5. Open your bot in Telegram and send `/start`.

The bot uses long polling, so it works from a home computer with no public IP, open ports, or
domain. Only the user IDs in `ALLOWED_TELEGRAM_USER_IDS` can use it.

## Keep it running on a Mac

```bash
./deploy/install-macos-service.sh            # starts now and at every login, restarts on crash
tail -f logs/tutor.log
./deploy/install-macos-service.sh uninstall
```

The service runs the bot under `caffeinate -i`, so the Mac won't idle-sleep while it's running.
Closing the lid of a laptop that isn't on power and an external display still sleeps it, and the
bot pauses until it wakes. Nothing is lost, because messages sent in the meantime are delivered when
it comes back. For 24/7 availability, run the same commands on any small Linux server or VM.

## Commands

| Command | What it does |
|---|---|
| `/start` | Welcome and first lesson (or resume if you've started) |
| `/continue` | Pick up exactly where you left off (typing `next` works too) |
| `/map` | Course map with progress; jump to any module or lesson |
| `/review` | Spaced review of items that are due |
| `/interview` | A mock interview question on what you've covered |
| `/quiz` | Fresh questions on the current lesson (AI tutor) |
| `/later` | Your laptop to-do list |
| `/progress` | Lessons completed, quiz accuracy, reviews due |
| `/report <note>` | Flag a problem with the step you're on (or tap 🚩 under any step) |

## Costs and privacy

- Your progress stays on your computer in `data/tutor.sqlite3`.
- With the AI tutor on, the relevant lesson context and your messages are sent to the Anthropic
  API. Normal study use makes small requests; drafting a whole unwritten lesson is the largest
  request, and it happens once per lesson. `TUTOR_EFFORT` and `TUTOR_MODEL` in `.env` trade depth
  for cost. The requests opt into Anthropic's server-side fallback, so a declined request is
  retried on another model automatically.
- The course uses synthetic data only. Don't send real patient information to the bot.

## Improving the course from real usage

Tap **🚩** under any step (or send `/report your note`) to flag a step as confusing, wrong, a typo,
too easy, or too hard. Then see where the course needs work:

```bash
.venv/bin/clinical-tutor stats
```

The report lists where learners currently are, the quiz steps with the lowest first-attempt
accuracy and the wrong option picked most often (a step everyone misses may have a bad key or a
misleading distractor), open questions with low scores, the steps that drew tutor questions or
"Go deeper" taps, every flagged step with its notes, and the review backlog. It reads only the
local database; nothing is uploaded.

## Contributing lessons

Read [`CONTENT_GUIDE.md`](CONTENT_GUIDE.md). In short: each lesson is one YAML file of steps
(`text`, `code`, `image`, `quiz`, `think`, `laptop`, `recap`), and figures are generated by code
in `figures/`. Before opening a pull request:

```bash
uv pip install -e ".[dev,figures]"
.venv/bin/clinical-tutor validate          # schema, Telegram limits, HTML, missing images
.venv/bin/clinical-tutor preview <lesson-id>
.venv/bin/python figures/render_all.py     # if you changed figures
.venv/bin/pytest -q
```

A good first contribution is taking an auto-drafted lesson from `data/generated/`, reviewing and
improving it, and moving it into `course/` with `status: authored`.

## Layout

```text
tutor/
├── course/                 # the curriculum: course.yaml, one folder per module, images/
├── figures/                # code that renders every figure in course/images/
├── src/clinical_tutor/
│   ├── content.py          # lesson schema, loader, validator
│   ├── engine.py           # position, flow, quizzes, review, interviews (transport-agnostic)
│   ├── llm.py              # AI tutor (Anthropic API)
│   ├── bot.py              # Telegram transport (aiogram, long polling)
│   ├── store.py            # SQLite persistence
│   ├── stats.py            # usage report (clinical-tutor stats)
│   ├── render.py, steps.py # Markdown → Telegram HTML, step rendering
│   └── cli.py              # run / validate / outline / preview
├── deploy/                 # macOS background service
└── tests/
```
