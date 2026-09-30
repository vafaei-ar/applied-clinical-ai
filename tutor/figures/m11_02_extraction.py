"""Figures for lesson m11-02-extraction."""

from __future__ import annotations

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, save
from matplotlib.patches import Rectangle


def fig_concept_crosswalk():
    """One clinical idea, several identifier systems (identifiers checked against public sources)."""
    fig, ax = figure(10, 6.6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    cols = [0.25, 3.55, 5.85, 7.85]
    heads = ["Concept", "UMLS CUI", "SNOMED CT", "ICD-10-CM"]
    head_colors = [INK, BLUE, GREEN, ORANGE]
    y = 6.05
    for x, h, c in zip(cols, heads, head_colors, strict=True):
        ax.text(x, y, h, fontsize=16, weight="bold", color=c, va="center")
    ax.plot([0.2, 9.8], [5.7, 5.7], color=MUTED, lw=1.5)

    rows = [
        ("Stroke\n(cerebrovascular accident)", "C0038454", "230690007", "I63.9", "cerebral infarction,\nunspecified"),
        ("Atrial fibrillation", "C0004238", "49436004", "I48.91", "unspecified atrial\nfibrillation"),
        ("Transient ischemic\nattack", "C0007787", "266257000", "G45.9", "transient cerebral\nischemic attack, unsp."),
    ]
    y = 5.05
    for name, cui, sno, icd, icd_desc in rows:
        ax.text(cols[0], y, name, fontsize=15, va="center")
        ax.text(cols[1], y, cui, fontsize=16, va="center", family="DejaVu Sans Mono", color=BLUE, weight="bold")
        ax.text(cols[2], y, sno, fontsize=16, va="center", family="DejaVu Sans Mono", color=GREEN, weight="bold")
        ax.text(cols[3], y + 0.14, icd, fontsize=16, va="center", family="DejaVu Sans Mono", color=ORANGE,
                weight="bold")
        ax.text(cols[3], y - 0.3, icd_desc, fontsize=11.5, va="center", color=MUTED)
        y -= 0.98

    # Drug band
    ax.add_patch(Rectangle((0.2, 0.25), 9.6, 1.85, fc="#f4f1fb", ec=PURPLE, lw=1.6))
    ax.text(0.45, 1.8, "Drugs use RxNorm: a different ID space (RxCUI, not a UMLS CUI)", fontsize=14.5,
            weight="bold", color=PURPLE, va="center")
    drugs = [
        ("atorvastatin (ingredient)", "83367"),
        ("atorvastatin 40 MG Oral Tablet (clinical drug)", "617311"),
        ("aspirin (ingredient)", "1191"),
    ]
    yy = 1.3
    for label, rx in drugs:
        ax.text(0.45, yy, label, fontsize=14, va="center")
        ax.text(9.55, yy, rx, fontsize=15, va="center", ha="right", family="DejaVu Sans Mono", color=PURPLE,
                weight="bold")
        yy -= 0.4
    return save(fig, "m11-02-concept-crosswalk")


def fig_context_scope():
    """ConText-style trigger, scope, and terminator on two synthetic sentences."""
    fs = 19
    lab = 15
    fig, ax = figure(10, 5.4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    fig.tight_layout()  # fix layout first so measured widths stay valid
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def width(text: str, weight: str = "normal") -> float:
        t = ax.text(0, 0, text, fontsize=fs, weight=weight)
        bb = t.get_window_extent(renderer)
        t.remove()
        return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]

    bold_kinds = ("trigger", "stop", "pseudo")

    def row(y, words, x0=0.25):
        x = x0
        spans = {}
        gap = 0.3
        for text, kind in words:
            w = width(text, "bold" if kind in bold_kinds else "normal")
            color = {"trigger": RED, "stop": MUTED, "pseudo": ORANGE}.get(kind, INK)
            ax.text(x, y, text, fontsize=fs, color=color, weight="bold" if kind in bold_kinds else "normal",
                    va="center")
            spans[text] = (x, x + w)
            x += w + gap
        return spans

    def box(span, y, color):
        x0, x1 = span
        ax.add_patch(Rectangle((x0 - 0.07, y - 0.3), x1 - x0 + 0.14, 0.6, fc=color, alpha=0.2, ec=color, lw=1.8))

    def center(span):
        return (span[0] + span[1]) / 2

    # Row 1
    y1 = 3.9
    s1 = row(y1, [("No evidence of", "trigger"), ("hemorrhage", "neg"), ("but", "stop"),
                  ("new right arm weakness", "pos")])
    box(s1["hemorrhage"], y1, RED)
    box(s1["new right arm weakness"], y1, GREEN)
    a = s1["No evidence of"][0]
    ax.annotate("", xy=(s1["but"][0] - 0.05, y1 + 0.55), xytext=(a, y1 + 0.55),
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 2.4})
    ax.text(a, y1 + 0.85, "trigger opens a scope", fontsize=lab, color=RED, weight="bold", va="bottom")
    ax.text(center(s1["but"]), y1 + 0.85, "terminator", fontsize=lab, color=MUTED, weight="bold", va="bottom",
            ha="center")
    ax.text(center(s1["hemorrhage"]), y1 - 0.5, "negated", fontsize=lab, color=RED, weight="bold", va="top",
            ha="center")
    ax.text(center(s1["new right arm weakness"]), y1 - 0.5, "present", fontsize=lab, color=GREEN,
            weight="bold", va="top", ha="center")

    # Row 2
    y2 = 1.55
    s2 = row(y2, [("No change", "pseudo"), ("in the", "x"), ("infarct", "pos")])
    box(s2["infarct"], y2, GREEN)
    ax.text(center(s2["infarct"]), y2 - 0.5, "present", fontsize=lab, color=GREEN, weight="bold", va="top",
            ha="center")
    ax.text(s2["No change"][0], y2 + 0.5, "pseudo-trigger: has \"no\", but is not a negation of the finding",
            fontsize=lab - 1, color=ORANGE, weight="bold", va="bottom")

    ax.set_title("Negation by trigger, scope, and terminator", loc="left", pad=6)
    return save(fig, "m11-02-context-scope")
