"""Figures for lesson m01-04-cohort-design."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch, Rectangle

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save

# Counts from the default generator run (n_patients=5000, seed=20260908) with the logic of
# sql/01_index_stroke_cohort.sql. Re-derive them if the generator changes.
N_PATIENTS = 5000
N_STROKE = 400
N_ADULT = 400
N_COHORT = 333
EXCLUDED_LOOKBACK = 39
EXCLUDED_GAP = 15
EXCLUDED_ENDED = 13


def _box(ax, x, y, w, h, text, color, fontsize=15, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=color, alpha=0.14, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=color, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, color=INK)


def _down(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2})


def _side(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.8})


def fig_attrition_funnel():
    """CONSORT-style attrition flow for the index stroke cohort, with real default-run counts."""
    fig, ax = figure(11, 8.6)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.axis("off")

    cx, w, h = 0.2, 5.4, 1.25
    ys = [8.35, 5.95, 3.55, 1.0]
    labels = [
        f"{N_PATIENTS:,} patients in database",
        f"{N_STROKE} with an ED/inpatient\nI63 encounter (first one kept)",
        f"{N_ADULT} aged 18+ at index",
        f"{N_COHORT} in final cohort",
    ]
    colors = [BLUE, BLUE, BLUE, GREEN]
    for y, lab, col in zip(ys, labels, colors):
        _box(ax, cx, y, w, h, lab, col, fontsize=15.5)
    for y_top, y_next in zip(ys[:-1], ys[1:]):
        _down(ax, cx + w / 2, y_top - 0.05, y_next + h + 0.05)

    ex_x, ex_w = 6.4, 4.45
    mid = [(ys[0] + ys[1] + h) / 2, (ys[1] + ys[2] + h) / 2, (ys[2] + ys[3] + h) / 2]
    _side(ax, cx + w / 2, ex_x - 0.05, mid[0])
    _box(ax, ex_x, mid[0] - 0.5, ex_w, 1.0, f"{N_PATIENTS - N_STROKE:,} no qualifying stroke",
         MUTED, fontsize=14, weight="normal")
    _side(ax, cx + w / 2, ex_x - 0.05, mid[1])
    _box(ax, ex_x, mid[1] - 0.45, ex_w, 0.9, f"{N_STROKE - N_ADULT} under 18", MUTED,
         fontsize=14, weight="normal")
    _side(ax, cx + w / 2, ex_x - 0.05, mid[2])
    _box(ax, ex_x, mid[2] - 1.15, ex_w, 2.3,
         f"{N_ADULT - N_COHORT} no 365-day coverage:\n"
         f"{EXCLUDED_LOOKBACK}  enrolled < 1 year\n"
         f"{EXCLUDED_GAP}  gap in the lookback\n"
         f"{EXCLUDED_ENDED}  coverage ended first",
         ORANGE, fontsize=14, weight="normal")

    ax.text(0.2, 0.25, "default synthetic run (seed 20260908)", fontsize=11.5, color=MUTED,
            style="italic")
    ax.set_title("Every exclusion is counted", loc="left", pad=4)
    return save(fig, "m01-04-attrition-funnel")


def fig_index_then_eligibility():
    """Three patients: why the repo picks the first stroke, then checks eligibility."""
    fig, ax = figure(11, 6.6)
    x0, x1 = 2019.0, 2026.0
    ax.set_xlim(x0 - 0.05, x1 + 0.05)
    ax.set_ylim(-0.1, 3.2)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    ax.tick_params(axis="x", labelsize=13)

    rows = [
        # (y, label, coverage periods, strokes, verdict, verdict color)
        (2.4, "A", [(2019.2, 2025.8)], [2022.4], "in: covered 1 year before", GREEN),
        (1.3, "B", [(2019.6, 2025.7)], [2020.3, 2022.9], "out: 1st stroke lacks lookback", ORANGE),
        (0.2, "C", [(2019.1, 2021.9), (2022.3, 2025.8)], [2022.6], "out: gap in lookback", RED),
    ]
    for y, lab, periods, strokes, verdict, vcol in rows:
        ax.text(x0 - 0.02, y + 0.12, lab, ha="right", va="center", fontsize=18, weight="bold")
        for a, b in periods:
            ax.add_patch(Rectangle((a, y), b - a, 0.24, color=BLUE, alpha=0.28, lw=0))
        first = strokes[0]
        ax.add_patch(Rectangle((first - 1, y - 0.06), 1, 0.36, fill=False, ec=INK, lw=1.6,
                               ls="--"))
        for i, s in enumerate(strokes):
            ax.plot(s, y + 0.12, marker="v" if i else "D", ms=15 if i else 14,
                    color=INK if i == 0 else MUTED, zorder=5)
        ax.text(x1, y + 0.5, verdict, ha="right", va="bottom", fontsize=14.5, color=vcol,
                weight="bold")

    ax.text(2022.9, 1.3 - 0.12, "2nd stroke: a 'first eligible'\nrule would pick this one",
            ha="center", va="top", fontsize=12.5, color=MUTED)
    ax.text(x0, 3.05, "blue = coverage    dashed = 365-day lookback    ◆ = first stroke",
            fontsize=12.5, color=MUTED, va="bottom")
    ax.set_title("Index first, then eligibility", loc="left", pad=22)
    return save(fig, "m01-04-index-then-eligibility")
