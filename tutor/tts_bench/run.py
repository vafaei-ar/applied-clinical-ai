"""Compare text-to-speech voices on the hard cases from our own lessons.

For every case in ``cases.yaml`` and every voice/rate you ask for, this renders the audio, then
(optionally) transcribes it back with a local speech-to-text model and checks whether the key terms
survived ("SNOMED", "I sixty-three", "milligrams per deciliter", "0.923"). It writes:

  results.json   every transcript and score
  report.md      scores per voice and category, plus the most-missed terms
  listen.html    a blind listening page: rate each clip without knowing which voice it is

Usage (from the tutor/ directory; needs `pip install -e ".[bench]"` for the scoring step):

  python tts_bench/run.py --voices Samantha,Daniel,Karen --rates 175
  python tts_bench/run.py --voices english --category "medical codes" --asr none   # audio only
  python tts_bench/run.py --voices Samantha --input raw    # skip our text rewriting, to see what
                                                           # the voice can do on its own

Add an engine by pointing --engine at ``module:factory`` where ``factory(voice, rate)`` returns an
object with ``name``, ``fingerprint`` and ``async synthesize(script, out_path)`` (see tts.py).
"""

from __future__ import annotations

import argparse
import asyncio
import html
import importlib
import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from clinical_tutor.content import load_course  # noqa: E402
from clinical_tutor.speech import Script, _paragraphs, load_lexicon, step_script  # noqa: E402
from clinical_tutor.tts import MacSay, macos_voices  # noqa: E402

NOVELTY = {
    "Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Good News", "Grandma",
    "Grandpa", "Jester", "Junior", "Organ", "Superstar", "Trinoids", "Whisper", "Wobble", "Zarvox",
}  # fmt: skip


# --- cases -------------------------------------------------------------------------------------


def load_cases(only_ids: list[str] | None, category: str | None) -> list[dict]:
    cases = yaml.safe_load((HERE / "cases.yaml").read_text(encoding="utf-8"))["cases"]
    if only_ids:
        cases = [c for c in cases if c["id"] in only_ids]
    if category:
        cases = [c for c in cases if c["category"] == category]
    return cases


def raw_script(text: str) -> Script:
    """Minimal clean-up only (no pronunciation rewriting): what a voice does on its own."""
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    t = re.sub(r"[*_`]", "", t)
    return [re.sub(r"\s+", " ", t).strip()]


def case_script(case: dict, mode: str, course, lex) -> Script:
    if case["kind"] == "step":
        lesson_id, step_no = case["source"].split("#")
        step = course.lessons[lesson_id].steps[int(step_no) - 1]
        return step_script(step, lex)
    if mode == "raw":
        return raw_script(case["text"])
    return _paragraphs(case["text"], lex)


# --- scoring -----------------------------------------------------------------------------------


def skeleton(text: str) -> str:
    t = text.lower()
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)  # 3,000 -> 3000
    t = re.sub(r"(\d+\.\d*?)0+\b", lambda m: m.group(1).rstrip("."), t)  # 0.720 -> 0.72
    return re.sub(r"[^a-z0-9]", "", t)


try:
    from rapidfuzz import fuzz

    def _close(a: str, b: str) -> bool:
        return fuzz.partial_ratio(a, b) >= 82
except ImportError:  # slower fallback
    from difflib import SequenceMatcher

    def _close(a: str, b: str) -> bool:
        n = len(a)
        return any(
            SequenceMatcher(None, a, b[i : i + n + 1]).ratio() >= 0.82
            for i in range(0, max(1, len(b) - n + 1))
        )


def term_heard(spec: str, transcript_skel: str) -> bool:
    """True if any of the 'a|b|c' renderings is present, exactly or with small ASR slips."""
    for alt in spec.split("|"):
        skel = skeleton(alt)
        if not skel:
            continue
        if skel in transcript_skel:
            return True
        if len(skel) >= 5 and _close(skel, transcript_skel):
            return True
    return False


class Transcriber:
    def __init__(self, model_name: str):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    @staticmethod
    def _load(path: Path):
        import numpy as np

        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
            capture_output=True,
            check=True,
        ).stdout
        return np.frombuffer(raw, dtype=np.float32)

    def __call__(self, path: Path) -> str:
        segments, _ = self.model.transcribe(self._load(path), language="en", beam_size=1)
        return " ".join(s.text.strip() for s in segments)


# --- voices ------------------------------------------------------------------------------------


def resolve_voices(spec: str) -> list[str]:
    installed = macos_voices()
    if spec == "english":
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        names = []
        for line in out.splitlines():
            if " en_" in line.split("#")[0]:
                names.append(line.split("#")[0].rstrip().rsplit(None, 1)[0].split(" (")[0])
        return [n for n in dict.fromkeys(names) if n not in NOVELTY]
    voices = [v.strip() for v in spec.split(",") if v.strip()]
    missing = [v for v in voices if v not in installed]
    if missing:
        raise SystemExit(f"Not installed: {missing}. Installed English voices: try `say -v '?'`.")
    return voices


def make_backend(engine: str, voice: str, rate: int):
    if engine == "macos":
        return MacSay(voice, rate)
    module, _, factory = engine.partition(":")
    return getattr(importlib.import_module(module), factory)(voice, rate)


# --- reporting ---------------------------------------------------------------------------------


def write_report(out: Path, rows: list[dict], configs: list[str], cases: list[dict], asr: bool) -> None:
    lines = ["# Voice comparison on challenge cases", ""]
    if asr:
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by[r["config"]][r["category"]].append(r["score"])
        cats = sorted({c["category"] for c in cases})
        lines.append("Share of key terms recognized after a round trip through speech-to-text.")
        lines.append("Higher is better; this measures intelligibility of the hard material, not")
        lines.append("naturalness. Listen too (listen.html).")
        lines += ["", "| voice | overall | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
        ranking = []
        for cfg in configs:
            allscores = [s for cat in by[cfg].values() for s in cat]
            ranking.append((sum(allscores) / len(allscores), cfg))
        for overall, cfg in sorted(ranking, reverse=True):
            cells = [f"{sum(by[cfg][c]) / len(by[cfg][c]):.2f}" if by[cfg][c] else "-" for c in cats]
            lines.append(f"| {cfg} | **{overall:.2f}** | " + " | ".join(cells) + " |")
        lines += ["", "## Terms missed by most voices", ""]
        missed = defaultdict(int)
        for r in rows:
            for term in r["missed"]:
                missed[(r["case"], term)] += 1
        for (case_id, term), n in sorted(missed.items(), key=lambda kv: -kv[1])[:25]:
            lines.append(f"- `{case_id}`: **{term}** missed by {n} of {len(configs)} voices")
    else:
        lines.append("Audio only (no scoring). Open listen.html to compare by ear.")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_listen_page(out: Path, rows: list[dict], configs: list[str], cases: list[dict], seed: int) -> None:
    rng = random.Random(seed)
    labels = {cfg: chr(65 + i) for i, cfg in enumerate(rng.sample(configs, len(configs)))}
    items = []
    for case in cases:
        clips = [r for r in rows if r["case"] == case["id"]]
        clips.sort(key=lambda r: labels[r["config"]])
        players = "".join(
            f'<div class="clip"><b>{labels[r["config"]]}</b> '
            f'<audio controls preload="none" src="{html.escape(r["audio"])}"></audio> '
            f'<select data-label="{labels[r["config"]]}" data-case="{html.escape(case["id"])}">'
            f'<option value="">rate</option>{"".join(f"<option>{i}</option>" for i in range(1, 6))}'
            f"</select></div>"
            for r in clips
        )
        shown = case.get("text") or f"(whole lesson step {case['source']})"
        items.append(
            f'<section><h3>{html.escape(case["id"])} <small>{html.escape(case["category"])}</small></h3>'
            f'<p class="why">{html.escape(case["why"])}</p>'
            f'<pre>{html.escape(shown)}</pre>{players}</section>'
        )
    legend = json.dumps({v: k for k, v in labels.items()})
    page = f"""<!doctype html><meta charset="utf-8"><title>Voice listening test</title>
<style>body{{font:16px system-ui;max-width:860px;margin:2rem auto;padding:0 1rem}}
section{{border-top:1px solid #ccc;padding:1rem 0}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:.6rem}}
.why{{color:#555;font-style:italic;margin:.2rem 0}}.clip{{margin:.3rem 0}}small{{color:#888;font-weight:normal}}
button{{padding:.5rem 1rem;font-size:1rem}}table{{border-collapse:collapse}}td,th{{padding:.2rem .8rem;border:1px solid #ccc}}</style>
<h1>Voice listening test</h1>
<p>Clips are labelled by letter, in a shuffled order, so you don't know which voice is which. Rate each
1 (hard to follow or wrong) to 5 (clear and natural). Ratings are saved in your browser. When you have
rated enough, reveal the averages.</p>
<button onclick="reveal()">Reveal averages by voice</button><div id="out"></div>
{"".join(items)}
<script>
const legend = {legend};
document.querySelectorAll('select').forEach(s => {{
  const key = 'rate:' + s.dataset.case + ':' + s.dataset.label;
  s.value = localStorage.getItem(key) || '';
  s.onchange = () => localStorage.setItem(key, s.value);
}});
function reveal() {{
  const sums = {{}};
  document.querySelectorAll('select').forEach(s => {{
    if (!s.value) return;
    const o = sums[s.dataset.label] = sums[s.dataset.label] || [0, 0];
    o[0] += +s.value; o[1] += 1;
  }});
  const rows = Object.entries(sums).map(([l, [t, n]]) => [legend[l], l, t / n, n])
    .sort((a, b) => b[2] - a[2]);
  document.getElementById('out').innerHTML = '<table><tr><th>voice</th><th>label</th><th>mean</th><th>rated</th></tr>' +
    rows.map(r => `<tr><td>${{r[0]}}</td><td>${{r[1]}}</td><td>${{r[2].toFixed(2)}}</td><td>${{r[3]}}</td></tr>`).join('') + '</table>';
}}
</script>"""
    (out / "listen.html").write_text(page, encoding="utf-8")


# --- main --------------------------------------------------------------------------------------


async def synth_all(jobs: list[tuple], concurrency: int) -> None:
    slots = asyncio.Semaphore(concurrency)

    async def one(backend, script, path):
        async with slots:
            if not path.exists():
                await backend.synthesize(script, path)

    await asyncio.gather(*(one(*j) for j in jobs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--voices", default="Samantha", help="comma list, or 'english' for all non-novelty English voices")
    ap.add_argument("--rates", default="175", help="comma list of words per minute")
    ap.add_argument("--engine", default="macos")
    ap.add_argument("--input", choices=["spoken", "raw"], default="spoken")
    ap.add_argument("--asr", default="small.en", help="faster-whisper model, or 'none' for audio only")
    ap.add_argument("--category")
    ap.add_argument("--cases", help="comma list of case ids")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()

    course, lex = load_course(), load_lexicon()
    cases = load_cases(args.cases.split(",") if args.cases else None, args.category)
    voices = resolve_voices(args.voices) if args.engine == "macos" else [v for v in args.voices.split(",")]
    rates = [int(r) for r in args.rates.split(",")]
    configs = [f"{v} @{r}" for v in voices for r in rates]
    out = args.out or (HERE.parent / "data" / "tts-bench" / time.strftime("%Y%m%d-%H%M%S"))
    (out / "audio").mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} cases x {len(configs)} voices = {len(cases) * len(configs)} clips -> {out}")

    jobs, plan = [], []
    for voice in voices:
        for rate in rates:
            backend = make_backend(args.engine, voice, rate)
            for case in cases:
                script = case_script(case, args.input, course, lex)
                rel = f"audio/{case['id']}__{voice.replace(' ', '_')}_{rate}.m4a"
                jobs.append((backend, script, out / rel))
                plan.append((f"{voice} @{rate}", case, rel))
    started = time.time()
    asyncio.run(synth_all(jobs, args.jobs))
    print(f"synthesized in {time.time() - started:.0f}s")

    asr = args.asr.lower() != "none"
    transcriber = Transcriber(args.asr) if asr else None
    rows = []
    for n, (config, case, rel) in enumerate(plan, 1):
        row = {"config": config, "case": case["id"], "category": case["category"], "audio": rel,
               "transcript": "", "score": None, "missed": []}
        if transcriber:
            text = transcriber(out / rel)
            skel = skeleton(text)
            misses = [t for t in case["terms"] if not term_heard(t, skel)]
            row.update(transcript=text, missed=misses, score=1 - len(misses) / len(case["terms"]))
        rows.append(row)
        if n % 50 == 0:
            print(f"  scored {n}/{len(plan)}", flush=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    write_report(out, rows, configs, cases, asr)
    write_listen_page(out, rows, configs, cases, args.seed)
    print(f"done: {out}/report.md  {out}/listen.html")


if __name__ == "__main__":
    main()
