"""Figures for lesson m11-01-notes-and-deid."""

from __future__ import annotations

import numpy as np
from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, save
from matplotlib.patches import FancyBboxPatch, Rectangle

MONO = "DejaVu Sans Mono"
CHAR_W = 0.602 / 72  # advance width of DejaVu Sans Mono in inches per point of font size


def fig_annotated_note():
    """A synthetic discharge summary with sections, and which mentions are not 'current, present'."""
    fs = 15
    fig, ax = figure(10, 9.4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.4)
    ax.axis("off")
    fig.tight_layout()  # fix the layout now so measured text widths stay valid when save() re-runs it
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    probe = ax.text(0, 0, "M" * 20, fontsize=fs, family=MONO)
    bb = probe.get_window_extent(renderer)
    cw = (inv.transform((bb.x1, bb.y0))[0] - inv.transform((bb.x0, bb.y0))[0]) / 20  # data units per char
    probe.remove()

    kinds = {
        "present": (GREEN, "current, present"),
        "negated": (RED, "negated"),
        "uncertain": (ORANGE, "uncertain"),
        "hypothetical": (PURPLE, "hypothetical"),
        "family": (BLUE, "family (someone else)"),
        "historical": (MUTED, "past history"),
    }

    # Each line: list of (text, kind or None, is_cue). Mentions are boxed, cues are bold colored.
    note = [
        ("H", "HISTORY OF PRESENT ILLNESS"),
        ("L", [("Woke with ", None, False), ("right arm weakness", "present", False),
               (".", None, False)]),
        ("L", [("No ", "negated", True), ("headache", "negated", False), (".", None, False)]),
        ("H", "FAMILY HISTORY"),
        ("L", [("Mother", "family", True), (" had a ", None, False), ("stroke", "family", False),
               (" at 60.", None, False)]),
        ("H", "PAST MEDICAL HISTORY"),
        ("L", [("Atrial fibrillation", "historical", False), (". ", None, False),
               ("TIA in 2019", "historical", False), (".", None, False)]),
        ("H", "HOSPITAL COURSE"),
        ("L", [("MRI: ", None, False), ("acute left MCA infarct", "present", False),
               (".", None, False)]),
        ("L", [("No ", "negated", True), ("hemorrhage", "negated", False), (".", None, False)]),
        ("L", [("Possible ", "uncertain", True), ("cardioembolic source", "uncertain", False),
               (".", None, False)]),
        ("H", "DISCHARGE PLAN"),
        ("L", [("Return if ", "hypothetical", True), ("new weakness", "hypothetical", False),
               (" develops.", None, False)]),
    ]

    ax.add_patch(FancyBboxPatch((0.25, 1.75), 9.5, 7.4, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#fbfbfa", ec=MUTED, lw=1.5))
    ax.text(0.6, 8.75, "Synthetic discharge summary (invented text)", fontsize=14, color=MUTED,
            style="italic", va="center")

    x0 = 0.75
    y = 8.05
    for kind, content in note:
        if kind == "H":
            y -= 0.08
            ax.text(x0, y, content, fontsize=13, weight="bold", color=MUTED, family=MONO, va="center")
            y -= 0.46
            continue
        x = x0
        for text, k, is_cue in content:
            width = len(text) * cw
            if k is not None and not is_cue:
                color = kinds[k][0]
                ax.add_patch(Rectangle((x - 0.04, y - 0.19), width + 0.08, 0.38, fc=color, alpha=0.22,
                                       ec=color, lw=1.8))
            weight = "bold" if is_cue else "normal"
            tcolor = kinds[k][0] if (k is not None and is_cue) else INK
            ax.text(x, y, text, fontsize=fs, family=MONO, va="center", color=tcolor, weight=weight)
            x += width
        y -= 0.42

    items = list(kinds.items())
    lx = [0.45, 3.65, 6.85]
    ly = [1.05, 0.5]
    for i, (_, (color, label)) in enumerate(items):
        col, row = i % 3, i // 3
        ax.add_patch(Rectangle((lx[col], ly[row] - 0.13), 0.36, 0.26, fc=color, alpha=0.3, ec=color,
                               lw=1.8))
        ax.text(lx[col] + 0.5, ly[row], label, fontsize=13.5, va="center")
    return save(fig, "m11-01-annotated-note")


def _simulate_deid(seed: int = 7):
    """Token-level scores from an imaginary de-identifier, 3% of tokens are PHI."""
    rng = np.random.default_rng(seed)
    n = 400_000
    is_phi = rng.random(n) < 0.03
    score = np.empty(n)

    # Easy non-PHI: ordinary words. Hard non-PHI: name-like clinical words (eponyms, brand names).
    hard_neg = (~is_phi) & (rng.random(n) < 0.015)
    easy_neg = (~is_phi) & ~hard_neg
    score[easy_neg] = rng.normal(-3.2, 1.1, easy_neg.sum())
    score[hard_neg] = rng.normal(0.6, 1.3, hard_neg.sum())

    # Easy PHI: dates, MRNs, common names. Hard PHI: rare names, initials, IDs inside free text.
    hard_pos = is_phi & (rng.random(n) < 0.02)
    easy_pos = is_phi & ~hard_pos
    score[easy_pos] = rng.normal(3.0, 1.1, easy_pos.sum())
    score[hard_pos] = rng.normal(-0.2, 1.3, hard_pos.sum())
    return is_phi, score


def deid_curve():
    """Return threshold-swept (recall, precision, thresholds, n_phi, n_neg) from the simulation."""
    is_phi, score = _simulate_deid()
    thresholds = np.linspace(-4.5, 4.5, 451)
    n_phi = int(is_phi.sum())
    n_neg = int((~is_phi).sum())
    recall, precision, false_pos = [], [], []
    for t in thresholds:
        flagged = score >= t
        tp = int((flagged & is_phi).sum())
        fp = int((flagged & ~is_phi).sum())
        recall.append(tp / n_phi)
        precision.append(tp / max(tp + fp, 1))
        false_pos.append(fp)
    return np.array(recall), np.array(precision), thresholds, np.array(false_pos), n_phi, n_neg


def _point_at_recall(recall, precision, target):
    """First point (highest threshold) that reaches the target recall."""
    idx = int(np.nonzero(recall >= target)[0].max())
    return idx, recall[idx], precision[idx]


def fig_deid_tradeoff():
    """Precision-recall curve of a simulated de-identifier with the cost asymmetry marked."""
    recall, precision, thr, fp, n_phi, n_neg = deid_curve()
    fig, ax = figure(10, 6.6)

    bar = 0.99
    ax.axvspan(0.80, bar, color=RED, alpha=0.07, lw=0)
    ax.axvline(bar, color=RED, lw=2, ls="--")
    ax.plot(recall, precision, color=BLUE, lw=3.5)
    ax.set_xlim(0.80, 1.003)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"])
    ax.set_xlabel("PHI recall (share of identifiers caught)")
    ax.set_ylabel("Precision")

    ax.text(0.845, 0.10, "left of the bar:\nidentifiers leak", ha="center", va="center", fontsize=15,
            color=RED, weight="bold")
    ax.annotate("more recall: fewer leaks,\nbut more useful text masked", xy=(0.975, 0.985),
                xytext=(0.807, 0.985), va="center", ha="left", fontsize=14, color=GREEN, weight="bold",
                arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 2.2})
    ax.text(bar - 0.002, 1.005, "example recall bar 0.99", ha="right", va="bottom", fontsize=12.5,
            color=RED)

    f1 = 2 * recall * precision / np.maximum(recall + precision, 1e-9)
    i_f1 = int(np.argmax(f1))
    ax.plot(recall[i_f1], precision[i_f1], "o", ms=14, color=ORANGE, zorder=5)
    ax.text(recall[i_f1] - 0.006, precision[i_f1] - 0.10,
            f"F1-optimal\nrecall {recall[i_f1]:.2f}\nprecision {precision[i_f1]:.2f}",
            ha="center", va="top", fontsize=14, color=ORANGE, weight="bold")

    _, r_hi, p_hi = _point_at_recall(recall, precision, bar)
    ax.plot(r_hi, p_hi, "s", ms=14, color=GREEN, zorder=5)
    ax.text(r_hi - 0.007, p_hi - 0.06, f"recall-first\nrecall {r_hi:.2f}\nprecision {p_hi:.2f}",
            ha="right", va="top", fontsize=14, color=GREEN, weight="bold")

    ax.set_title("Set the threshold by the cost of a miss", loc="left", pad=8)
    return save(fig, "m11-01-deid-tradeoff")
