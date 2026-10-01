# Choosing a voice: the challenge set

Lessons are full of things a speech engine gets wrong: ICD codes (`I63.9`), CPT codes, acronyms read
as words (SNOMED, LOINC, FHIR), snake_case column names, units (mg/dL), statistics with symbols,
drug names, ISO dates. `cases.yaml` holds 62 of the hardest cases, **all taken verbatim from our own
lessons** (`pytest tests/test_speech.py` checks that), in ten categories, each with the terms a
listener should hear.

`run.py` renders every case with every voice you ask for, transcribes the audio back with a local
speech-to-text model, and reports how many of the key terms survived. It also writes a blind
listening page, because intelligibility is not the same as sounding pleasant.

```bash
pip install -e ".[bench]"          # faster-whisper + rapidfuzz; also needs ffmpeg on the PATH
python tts_bench/run.py --voices english                  # all non-novelty English macOS voices
python tts_bench/run.py --voices "Samantha,Daniel" --rates 160,175,190
python tts_bench/run.py --voices Samantha --category "medical codes" --asr none   # audio only
python tts_bench/run.py --voices Samantha --input raw     # skip our rewriting: voice on its own
```

Output goes to `data/tts-bench/<timestamp>/`: `report.md` (scores by voice and category, and the
terms most voices miss), `results.json` (every transcript), and `listen.html` (open it in a
browser, rate clips 1 to 5 without knowing which voice is which, then reveal the averages).

## Two inputs, two questions

- `--input spoken` (default) feeds the voice our rewritten text (`speech.py` and
  `course/pronunciations.yaml`). It answers: *which voice is best given our text rewriting?*
- `--input raw` feeds only de-markdowned text. It answers: *how much does a voice need our
  rewriting?* A neural voice with good built-in text normalization will score closer to `spoken`
  here, and might need less of the lexicon.

## Adding a voice or engine

Any object with `name`, `fingerprint`, and `async synthesize(script, out_path)` works (see
`src/clinical_tutor/tts.py`; a script is a list of strings to speak and floats of seconds of
silence). Point `--engine` at a factory:

```bash
python tts_bench/run.py --engine mypackage.voices:make --voices alloy,nova
```

where `make(voice, rate)` returns the backend. Run it against the same cases to compare it with
the built-in voices on equal terms.

## What the score does and doesn't mean

The score is the share of expected terms that a transcription model recognized. It catches
mispronounced or dropped codes, numbers, and names; it cannot judge pacing, naturalness, or
listening fatigue, and the transcriber has its own errors (it sometimes drops a zero from a long
number). Use it to rule voices out and to find lexicon gaps, then decide among the survivors by
ear.

## Findings so far (macOS built-in voices, 175 words per minute)

Scores are the share of key terms recognized after a round trip through `faster-whisper small.en`,
on the lexicon as committed. Daniel, Moira, Karen, and Tessa score 0.99, Samantha 0.97; earlier
runs put Tara at 0.91 and Kathy, Ralph, Sandy, Rocko, Reed, Flo, Shelley, Eddy, Aman, and Fred
between 0.72 and 0.82. A difference of one term is about 0.4 points, so the top group is a tie on
intelligibility: choose among them by ear with `listen.html`. Daniel is the default because he
leads on medical codes and acronyms; set `TTS_VOICE` to change it.

What this exercise taught us, which is why the notes below matter more than the ranking:

- **Check the scorer before trusting the ranking.** The first run had universal "misses" that
  were the transcriber's fault (it writes "kappa" as "cap", `Parquet` as "per K", drops a repeated
  digit from `93000`) or bad expected terms. A bigger transcriber (`small.en`, not `base.en`) and
  alternate renderings per term fixed most of it. When every voice misses a term, suspect the
  scorer, and listen.
- **Spell letters in CAPITALS in the lexicon.** A lowercase "a" is read as the article "uh":
  "S a M D" came out as "S-uh-M-D" and was heard as "SMD".
- **Letter-by-letter beats a guessed phonetic spelling.** "loynk" was recognized as LOINC for 2 of
  6 voices; "L O I N C" for all 6. Test a spelling on several voices before committing it.
- **Capital words with a run of four or more consonants are acronyms** (IMDRF, SDTM, HTTPS); the
  converter spells those and reads other capital words as words (UNKNOWN, STORE).
- **Accents reorder dates.** British, Irish, and South African voices say "31 December", so scoring
  accepts both orders. Tara said "January 1 of January 2019", which is a real glitch.

Not measured: naturalness, pacing, and listening fatigue over a 20-minute lesson, and anything about
voices outside the built-in set. Run the same cases against any engine you are considering.
