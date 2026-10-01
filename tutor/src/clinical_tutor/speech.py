"""Turn lesson steps into text that sounds natural when read aloud.

Lessons are written for the eye: Markdown, code, symbols, table-like lists, ICD codes, snake_case
column names. Read verbatim by a speech engine they come out badly ("I six three percent",
"ay you row see"). This module rewrites them into a *script*: a list of strings to speak and
floats (seconds of silence), which any voice backend can render.

Pronunciations live in ``course/pronunciations.yaml``. A step may carry a ``speak:`` field to
replace the generated narration with hand-written text (still passed through the lexicon).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .content import (
    DEFAULT_COURSE_DIR,
    CodeStep,
    ImageStep,
    LaptopStep,
    Lesson,
    QuizStep,
    RecapStep,
    Step,
    TextStep,
    ThinkStep,
)

Script = list["str | float"]

WORDS_PER_MINUTE = 165
QUIZ_THINK_SECONDS = 6.0
THINK_PAUSE_SECONDS = 8.0
PARAGRAPH_PAUSE = 0.6
BULLET_PAUSE = 0.35

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_DIGITS = "zero one two three four five six seven eight nine".split()
_MONTHS = (
    "January February March April May June July August September October November December"
).split()

# All-caps words that are real words (or SQL keywords) and must not be spelled out letter by letter.
_CAPS_WORDS = set(
    """A I AND OR NOT IN IS ALL ANY NULL TRUE FALSE SELECT FROM WHERE GROUP BY HAVING ORDER LIMIT
    JOIN LEFT RIGHT INNER OUTER ON AS CASE WHEN THEN ELSE END DISTINCT UNION OVER PARTITION ROWS
    RANGE BETWEEN PRECEDING FOLLOWING CURRENT ROW QUALIFY WITH INSERT UPDATE DELETE CREATE TABLE
    EXISTS LIKE COUNT SUM MAX MIN AVG UNKNOWN AT TO IF NO YES OK THE THIS THAT ONLY NEVER ALWAYS
    MUST BUT SO UP OFF IT BE AN OF WE FOR ARE WAS NOW NEW ADD SET KEY FULL CROSS USING NATURAL
    FIRST LAST ASC DESC TOP""".split()
)

# Words that read as plain English in a code span (SQL and common function names).
_CODE_WORDS = _CAPS_WORDS | set("ROW_NUMBER RANK DENSE_RANK LAG LEAD COALESCE CAST".split())

_MATH = {
    "≤": "is at most", "≥": "is at least", "≠": "is not equal to", "≈": "approximately",
    "±": "plus or minus", "×": "times", "·": "times", "÷": "divided by", "…": "and so on",
    "→": "to", "−": "minus",
}  # fmt: skip
_OPERATORS = {
    **_MATH,
    "<>": "not equal to",
    "!=": "not equal to",
    "<=": "less than or equal to",
    ">=": "greater than or equal to",
    "==": "equals",
    "=": "equals",
    "<": "less than",
    ">": "greater than",
    "->": "to",
    "=>": "to",
    "::": "as",
    "+": "plus",
    "*": "star",
    "%": "percent",
    "|": "or",
    "&": "and",
}
_PATH_EXT = r"\.(?:py|sql|yaml|yml|md|json|csv|toml|duckdb|parquet|dcm|html|txt|ipynb|sh|tf|cfg)"
_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "θ": "theta",
    "κ": "kappa", "λ": "lambda", "μ": "mu", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau",
    "φ": "phi", "χ": "chi", "ω": "omega", "Δ": "delta", "Σ": "sigma",
}  # fmt: skip
_SUBSCRIPTS = {
    "ₜ": " sub t",
    "ₑ": " sub e",
    "ᵢ": " sub i",
    "ⱼ": " sub j",
    "ₙ": " sub n",
    "ₘ": " sub m",
}
_SUPERSCRIPTS = {"²": " squared", "³": " cubed", "⁻¹": " inverse"}


def _two_digits(digits: str) -> str:
    """Say a two-digit code number the way people read ICD and CPT fragments: 63 -> sixty-three."""
    if digits.startswith("0"):
        return f"zero {_DIGITS[int(digits[1])]}"
    n = int(digits)
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def _caps_word(word: str) -> str:
    """Speak a run of capitals: short or vowel-less ones are acronyms (spell them), others words."""
    if len(word) == 1:
        return word
    if word in _CAPS_WORDS:
        return word.lower()
    # Longer words with a vowel are usually real words, unless they have a run of four or more
    # consonants, which no English word does (IMDRF, SDTM, HTTPS): those are acronyms.
    if len(word) >= 4 and re.search(r"[AEIOUY]", word) and not re.search(r"[^AEIOUY]{4}", word):
        return word.lower()
    return " ".join(word)


def _say_n(n: int) -> str:
    """Say a small whole number in words (module and lesson numbers)."""
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def _digit_by_digit(digits: str) -> str:
    return " ".join(_DIGITS[int(d)] for d in digits)


def _grouped_digits(digits: str) -> str:
    groups = [digits[max(0, i - 3) : i] for i in range(len(digits), 0, -3)][::-1]
    return ", ".join(_digit_by_digit(g) for g in groups)


@dataclass
class Lexicon:
    terms: dict[str, str]

    def __post_init__(self) -> None:
        keys = sorted(self.terms, key=len, reverse=True)
        body = "|".join(re.escape(k) for k in keys)
        self._pattern = (
            re.compile(rf"(?<![A-Za-z0-9_])({body})(s?)(?![A-Za-z0-9_])") if body else None
        )

    def lookup(self, token: str) -> str | None:
        return self.terms.get(token)

    def apply(self, text: str, protect) -> str:
        if self._pattern is None:
            return text

        def repl(m: re.Match[str]) -> str:
            key, plural = m.group(1), m.group(2)
            spoken = self.terms[key]
            if plural:  # keys such as "CIs" are listed explicitly and match as a whole first
                spoken += " s" if len(spoken.split()[-1]) == 1 else "s"
            return protect(spoken)

        return self._pattern.sub(repl, text)


def load_lexicon(path: Path | None = None) -> Lexicon:
    path = path or (DEFAULT_COURSE_DIR / "pronunciations.yaml")
    if not path.is_file():
        return Lexicon({})
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Lexicon({str(k): str(v) for k, v in (raw.get("terms") or {}).items()})


_PH_OPEN, _PH_CLOSE, _PH_BASE = "\ue000", "\ue001", 0xE100
_PH_RE = re.compile(f"{_PH_OPEN}([\ue100-\ue109]+){_PH_CLOSE}")


class _Protector:
    """Holds finished spoken fragments behind placeholders so later rules can't rewrite them.

    Placeholders use private-use characters, which no pattern here treats as a letter or digit.
    """

    def __init__(self) -> None:
        self.items: list[str] = []

    def __call__(self, spoken: str) -> str:
        self.items.append(spoken)
        digits = "".join(chr(_PH_BASE + int(d)) for d in str(len(self.items) - 1))
        return f"{_PH_OPEN}{digits}{_PH_CLOSE}"

    def restore(self, text: str) -> str:
        def unwrap(m: re.Match[str]) -> str:
            return self.items[int("".join(str(ord(c) - _PH_BASE) for c in m.group(1)))]

        for _ in range(5):  # a code span can contain a lexicon term, so unwind repeatedly
            new = _PH_RE.sub(unwrap, text)
            if new == text:
                break
            text = new
        return text


def _ident_words(ident: str, lex: Lexicon) -> str:
    """Speak an identifier: latest_ldl_prior_365d -> 'latest L D L prior 365 days'."""
    pieces: list[str] = []
    for part in re.split(r"_+", ident):
        if not part:
            continue
        pieces += re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+[a-z]?", part)
    if len(pieces) == 2 and all(len(p) == 1 for p in pieces):
        return f"{pieces[0]} sub {pieces[1]}"
    spoken = []
    for piece in pieces:
        m = re.fullmatch(r"(\d+)d", piece)
        if m:
            spoken.append(f"{m.group(1)} days")
        elif piece.isdigit():
            spoken.append(piece)
        elif lex.lookup(piece.upper()) and not (piece.isupper() and piece in _CODE_WORDS):
            spoken.append(lex.lookup(piece.upper()) or piece)
        elif lex.lookup(piece):
            spoken.append(lex.lookup(piece) or piece)
        elif len(piece) == 1:
            spoken.append(piece.upper())
        elif piece.isupper():
            spoken.append(_caps_word(piece))
        else:
            spoken.append(piece.lower())
    return " ".join(spoken)


_CODE_TOKEN_RE = re.compile(
    r"[A-Z]\d{2}(?:\.(?:\d+|x))?-?(?![\w])"  # ICD-style code: I10, Z86.73, I69.3-
    r"|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|<>|<=|>=|!=|==|->|=>|::|\S"
)
_ICD_TOKEN_RE = re.compile(r"[A-Z]\d{2}(?:\.(?:\d+|x))?-?")


def _code_span(span: str, lex: Lexicon) -> str:
    """Speak the contents of a `code span`: paths, SQL, identifiers, codes, and formulas."""
    s = span.strip()
    if not s:
        return ""
    if " " not in s and (
        ("/" in s and re.search(r"[A-Za-z]/[A-Za-z_]", s)) or re.search(_PATH_EXT + r"$", s)
    ):
        base = s.rstrip("/").split("/")[-1]
        stem, dot, ext = base.partition(".")
        what = "the file" if dot else "the folder"
        name = _ident_words(stem, lex)
        return f"{what} {name}" + (f" dot {_ident_words(ext, lex)}" if dot else "")
    like = re.fullmatch(r"'?([A-Z]\d+(?:\.\d+)?)%'?", s)
    if like:
        return f"any code starting with {_code_number(like.group(1))}"
    if re.fullmatch(r"[A-Z]{2,5}\^[A-Z]\d{2}", s):  # HL7 message types such as ADT^A01
        head, tail = s.split("^")
        return f"{_ident_words(head, lex)} {_code_number(tail)}"

    s = re.sub(r"\b([A-Za-z])\(([^()]+)\)", r"\1 of \2", s)  # p(1) -> p of 1
    for sym, word in _GREEK.items():
        s = s.replace(sym, f" {word} ")
    tokens = _CODE_TOKEN_RE.findall(s)

    def digits_chain(i: int) -> bool:
        before = i > 1 and tokens[i - 1] == "-" and tokens[i - 2].isdigit()
        after = i + 2 < len(tokens) and tokens[i + 1] == "-" and tokens[i + 2].isdigit()
        return before or after

    out: list[str] = []
    for i, tok in enumerate(tokens):
        if _ICD_TOKEN_RE.fullmatch(tok):
            out.append(_code_number(tok.rstrip("-")) + (" dash" if tok.endswith("-") else ""))
        elif re.match(r"[A-Za-z_]", tok):
            if tok.upper() in _CODE_WORDS and tok.isupper():
                out.append(tok.lower().replace("_", " "))
            elif lex.lookup(tok) and "_" not in tok:
                out.append(lex.lookup(tok) or tok)
            else:
                out.append(_ident_words(tok, lex))
        elif tok == "%":
            out.append("wildcard")
        elif tok in _OPERATORS:
            out.append(_OPERATORS[tok])
        elif tok == ".":
            prev_word = i > 0 and re.match(r"[A-Za-z_]", tokens[i - 1])
            next_word = i + 1 < len(tokens) and re.match(r"[A-Za-z_]", tokens[i + 1])
            out.append("dot" if prev_word and next_word else ",")
        elif tok == ",":
            out.append(",")
        elif tok == "-":
            neighbours = (
                i > 0
                and i + 1 < len(tokens)
                and tokens[i - 1].isdigit()
                and tokens[i + 1].isdigit()
            )
            out.append("dash" if neighbours else "minus")
        elif tok == "/":
            out.append("over")
        elif tok == "^":
            out.append("to the power")
        elif tok == "@":
            out.append("at")
        elif tok.isdigit() and (digits_chain(i) or len(tok) == 5):
            out.append(_digit_by_digit(tok))
        elif tok.isdigit() and len(tok) >= 6:
            out.append(_grouped_digits(tok))
        elif tok.isdigit() or re.fullmatch(r"\d+\.\d+", tok):
            out.append(tok)
        # brackets, quotes, semicolons and other punctuation are dropped
    return " ".join(w for w in out if w)


def _code_number(code: str) -> str:
    m = re.fullmatch(r"([A-Z])(\d{2})(?:\.(\d+|x))?", code)
    if not m:
        return code
    letter, digits, sub = m.groups()
    spoken = f"{letter} {_two_digits(digits)}"
    if sub:
        spoken += " point " + (_digit_by_digit(sub) if sub.isdigit() else sub)
    return spoken


_CODE_RE = re.compile(r"(?<![\w.])([A-Z])(\d{2})(?:\.(\d+|x))?(-)?(?![\w])")
_CODE_RANGE_RE = re.compile(
    r"(?<![\w.])([A-Z]\d{2}(?:\.\d+)?)\s?[–-]\s?([A-Z]\d{2}(?:\.\d+)?)(?![\w])"
)


def _emoji_free(text: str) -> str:
    return re.sub("[\U0001f300-\U0001faff☀-➿⭐⬆️‍]", "", text)


def speakable(text: str, lex: Lexicon) -> str:
    """Rewrite one block of lesson Markdown as plain, speakable text."""
    protect = _Protector()
    t = text
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"\[([^\]]+)\]\((?:https?://)[^)\s]+\)", r"\1", t)
    t = _emoji_free(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)
    t = re.sub(r"(?<![\w*])_(?!\s)((?:(?!\n\n).)+?)(?<!\s)_(?![\w*])", r"\1", t, flags=re.S)
    t = re.sub(r"`([^`\n]+)`", lambda m: protect(_code_span(m.group(1), lex)), t)

    # snake_case words outside backticks, and "check #5"-style numbers.
    t = re.sub(
        r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b",
        lambda m: protect(_ident_words(m.group(0), lex)),
        t,
    )
    t = re.sub(r"#(\d+)", r"number \1", t)
    t = re.sub(r"\b(\w+)\(\*\)", r"\1 star", t)

    # Lesson and module references.
    t = re.sub(
        r"\bm(\d{2})-(\d{2})(?:-[a-z0-9-]+)?\b",
        lambda m: protect(f"module {_say_n(int(m.group(1)))}, lesson {_say_n(int(m.group(2)))}"),
        t,
    )
    t = re.sub(
        r"\b(Module|module|Lesson|lesson)\s0?(\d{1,2})\b",
        lambda m: protect(f"{m.group(1)} {_say_n(int(m.group(2)))}"),
        t,
    )
    t = re.sub(
        r"\bM0?(\d{1,2})\b(?=[:\s)])",
        lambda m: protect(f"module {_say_n(int(m.group(1)))}"),
        t,
    )

    # Dates and versions.
    def iso_date(m: re.Match[str]) -> str:
        year, month, day = m.group(1), int(m.group(2)), m.group(3)
        if not 1 <= month <= 12:
            return m.group(0)
        return protect(f"{_MONTHS[month - 1]}{f' {int(day)},' if day else ''} {year}")

    t = re.sub(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", iso_date, t)
    t = re.sub(r"\b((?:19|20)\d{2})-(0[1-9]|1[0-2])\b(?!-)", lambda m: iso_date(m), t)
    t = re.sub(
        r"(\b[Vv]ersion\s+)?(?:\bv(\d+(?:\.\d+)*)\b|\b(\d+\.\d+\.\d+)\b)",
        lambda m: protect(
            ("version " if not m.group(1) else m.group(1))
            + (m.group(2) or m.group(3)).replace(".", " point ")
        ),
        t,
    )

    # Names and terms with a fixed pronunciation.
    t = lex.apply(t, protect)

    # Medical codes. Code ranges first (I11–I13), then single codes and long identifiers.
    t = _CODE_RANGE_RE.sub(
        lambda m: protect(f"{_code_number(m.group(1))} to {_code_number(m.group(2))}"), t
    )
    t = _CODE_RE.sub(
        lambda m: protect(_code_number(m.group(0).rstrip("-")) + (" dash" if m.group(4) else "")),
        t,
    )
    t = re.sub(r"\bC\d{7}\b", lambda m: protect("C " + _digit_by_digit(m.group(0)[1:])), t)
    t = re.sub(
        r"\b(\d{3,6})-(\d)\b",
        lambda m: protect(f"{_digit_by_digit(m.group(1))} dash {_DIGITS[int(m.group(2))]}"),
        t,
    )
    t = re.sub(
        r"(?<![\d,.$])\b(\d{6,})\b(?![,.]\d)", lambda m: protect(_grouped_digits(m.group(1))), t
    )
    t = re.sub(
        r"(?<![\d,.$])\b(\d{5})\b(?![,.]\d)", lambda m: protect(_digit_by_digit(m.group(1))), t
    )

    # Numbers, ranges, money, and math symbols.
    t = re.sub(r"(?<=\d)\s?[–-]\s?(?=\d)", " to ", t)
    t = re.sub(r"(\d)\s?%", r"\1 percent", t)
    t = re.sub(
        r"\$(\d[\d,]*(?:\.\d+)?)(\s*(?:million|billion|thousand))?",
        lambda m: f"{m.group(1)}{m.group(2) or ''} dollars",
        t,
    )
    t = re.sub(r"(\d+(?:\.\d+)?)\s?¢", r"\1 cents", t)
    t = re.sub(r"\bln\(", "natural log of (", t)
    t = re.sub(r"\blog\(", "log of (", t)
    for sym, spoken in {**_SUBSCRIPTS, **_SUPERSCRIPTS}.items():
        t = t.replace(sym, spoken)
    t = re.sub(r"\b([A-Za-z])_([A-Za-z0-9])\b", r"\1 sub \2", t)
    t = re.sub(r"\b([a-zA-Z])\((\w+)\)", r"\1 of \2", t)
    t = re.sub(r"(\S)\^(\S)", r"\1 to the power \2", t)
    for sym, spoken in _GREEK.items():
        t = t.replace(sym, f" {spoken} ")
    t = t.replace("~", " about ").replace("@", " at ")
    for sym, word in _MATH.items():
        if sym not in "→−":  # arrows and minus signs need context, handled below
            t = t.replace(sym, f" {word} ")
    t = re.sub(r"(?<=\d) x (?=\d)", " times ", t)  # 0.95 x 0.05
    t = re.sub(r"<(?=\d)", " less than ", t)
    t = re.sub(r">(?=\d)", " greater than ", t)
    t = re.sub(r"[\[\]{}]", " ", t)
    t = t.replace("*", " ").replace("`", "").replace("_", " ")
    t = re.sub(r"(^|\s)<(?=\s)", r"\1less than", t)
    t = re.sub(r"(^|\s)>(?=\s)", r"\1greater than", t)
    t = re.sub(r"(?<=\s)=(?=\s)", "equals", t)
    t = re.sub(r"(?<=\s)\+(?=\s)", "plus", t)
    t = re.sub(r"\s[−-]\s", " minus ", t)
    t = re.sub(r"(?<=[\s(])[−](?=\d)", "minus ", t)
    t = t.replace("−", " minus ")
    arrows = t.count("→")
    t = t.replace("→", " then " if arrows >= 2 else " to ")

    def slash(m: re.Match[str]) -> str:
        left, right = m.group(1), m.group(2)
        over = (
            len(left) == 1
            or len(right) == 1
            or left[-1] in "0123456789)"
            or right[0] in "0123456789("
            or " " in m.group(0)
        )
        return f"{left} {'over' if over else 'slash'} {right}"

    t = re.sub(r"([\w.)\ue000-\ue1ff]+)\s?/\s?([\w.(\ue000-\ue1ff]+)", slash, t)
    t = t.replace("–", " ").replace("—", ", ")

    # Anything still ALL-CAPS and not a real word is an acronym: spell it out.
    def spell(m: re.Match[str]) -> str:
        word, plural = m.group(1), m.group(2)
        if len(word) == 1:
            return m.group(0)
        spoken = _caps_word(word)
        if " " in spoken:  # an acronym, spelled out
            return protect(spoken + (" s" if plural else ""))
        return spoken + plural  # a real word written in capitals

    t = re.sub(r"(?<![A-Za-z0-9])([A-Z]{2,12})(s?)(?![A-Za-z0-9])", spell, t)

    # Lists and layout.
    t = re.sub(r"^\s*[-*]\s+", "", t, flags=re.M)
    t = re.sub(r"\s*\|\s*", ", ", t)
    t = protect.restore(t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t.strip()


def _lesson_no(digits: str) -> str:
    n = int(digits)
    return _DIGITS[n] if n < 10 else _two_digits(digits)


_BULLET_RE = re.compile(r"\s*(?:[-*]|\d+[.)])\s+")


def _paragraphs(text: str, lex: Lexicon) -> Script:
    """Speak Markdown paragraph by paragraph, with a short pause between paragraphs and bullets."""
    script: Script = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if not any(_BULLET_RE.match(ln) for ln in lines):
            spoken = speakable(" ".join(ln.strip() for ln in lines), lex)
            if spoken:
                script += [_sentence(spoken), PARAGRAPH_PAUSE]
            continue
        # Bullets: a line that doesn't start a bullet continues the previous item (a soft wrap).
        items: list[str] = []
        for ln in lines:
            if _BULLET_RE.match(ln) or not items:
                items.append(_BULLET_RE.sub("", ln, count=1).strip())
            else:
                items[-1] += " " + ln.strip()
        for item in items:
            spoken = speakable(item, lex)
            if spoken:
                script += [_sentence(spoken), BULLET_PAUSE]
    return script


def _sentence(text: str) -> str:
    text = text.strip()
    return text if re.search(r"[.!?:;,]$", text) else text + "."


_LETTERS = "ABCDEF"


def step_script(step: Step, lex: Lexicon) -> Script:
    """The narration for one lesson step."""
    override = getattr(step, "speak", None)
    if override:
        return _paragraphs(override, lex)

    s: Script = []
    if isinstance(step, TextStep):
        if step.title:
            s += [_sentence(speakable(step.title, lex)), 0.4]
        s += _paragraphs(step.text, lex)
    elif isinstance(step, CodeStep):
        if step.title:
            s += [_sentence(speakable(step.title, lex)), 0.4]
        if step.text:
            s += _paragraphs(step.text, lex)
        s += ["A code example is shown in the text version.", 0.5]
        if step.after:
            s += _paragraphs(step.after, lex)
    elif isinstance(step, ImageStep):
        s += ["Figure.", 0.3] + _paragraphs(step.caption, lex)
    elif isinstance(step, QuizStep):
        s += ["Quick check.", 0.4] + _paragraphs(step.question, lex)
        if step.code:
            s += ["A code snippet is shown in the text version.", 0.4]
        s += ["The options are.", 0.3]
        for i, option in enumerate(step.options):
            s += [_sentence(f"{_LETTERS[i]}. {speakable(option, lex)}"), 0.45]
        s += ["Take a moment to think.", QUIZ_THINK_SECONDS]
        s += [_sentence(f"The answer is {_LETTERS[step.answer]}"), 0.4]
        s += _paragraphs(step.explanation, lex)
    elif isinstance(step, ThinkStep):
        s += ["Interview question." if step.interview else "Think it through.", 0.4]
        s += _paragraphs(step.prompt, lex)
        s += ["Try answering out loud.", THINK_PAUSE_SECONDS, "Here is a model answer.", 0.4]
        s += _paragraphs(step.model_answer, lex)
        s += ["The key points.", 0.3]
        for point in step.key_points:
            s += [_sentence(speakable(point, lex)), BULLET_PAUSE]
    elif isinstance(step, LaptopStep):
        s += [_sentence(f"For your laptop: {speakable(step.title, lex)}"), 0.4]
        s += _paragraphs(step.task, lex)
    elif isinstance(step, RecapStep):
        s += ["Recap.", 0.4]
        for point in step.points:
            s += [_sentence(speakable(point, lex)), BULLET_PAUSE]
    return s


def script_words(script: Script) -> int:
    return sum(len(x.split()) for x in script if isinstance(x, str))


def script_seconds(script: Script) -> float:
    return script_words(script) / WORDS_PER_MINUTE * 60 + sum(
        x for x in script if isinstance(x, float)
    )


def lesson_parts(lesson: Lesson, lex: Lexicon, max_words: int = 750) -> list[Script]:
    """The whole lesson as a few audio parts that split at step boundaries."""
    intro: Script = [_sentence(speakable(lesson.title, lex)), 0.5]
    intro += _paragraphs(lesson.summary, lex)
    intro += ["Why it matters.", 0.3] + _paragraphs(lesson.why_it_matters, lex)
    intro += ["In this lesson you will learn to.", 0.3]
    for objective in lesson.objectives:
        intro += [_sentence(speakable(objective, lex)), BULLET_PAUSE]

    parts: list[Script] = []
    current: Script = intro
    for step in lesson.steps:
        if isinstance(step, LaptopStep):
            continue  # hands-on tasks are for the laptop; they stay in /later
        narration = step_script(step, lex)
        if not narration:
            continue
        if (
            script_words(current) + script_words(narration) > max_words
            and script_words(current) > 150
        ):
            parts.append(current)
            current = []
        current += narration + [1.0]
    if current:
        parts.append(current)
    return parts
