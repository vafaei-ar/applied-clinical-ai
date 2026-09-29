"""Figures for lesson m00-01-landscape."""

from __future__ import annotations

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, save


def _box(ax, x, y, w, h, color, alpha=0.14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=color, alpha=alpha, edgecolor=color, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="none", edgecolor=color, lw=2))


def _arrow(ax, start, end, color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=22, color=color,
                                 lw=2.2, connectionstyle=f"arc3,rad={rad}"))


def fig_lifecycle():
    """Nine lifecycle stages in a 3x3 serpentine, each with its owner and course module."""
    fig, ax = figure(11, 8.4)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    stages = [
        ("1  Question", "clinician + informaticist", "M02 · M07", PURPLE),
        ("2  Data", "data engineer", "M01 · M06", BLUE),
        ("3  Cohort + label", "data scientist / biostat", "M01", BLUE),
        ("4  Features", "data scientist", "M01", BLUE),
        ("5  Model", "ML engineer / scientist", "M02", GREEN),
        ("6  Validation", "biostatistician", "M02 · M04", GREEN),
        ("7  Deployment", "ML / platform engineer", "M03", ORANGE),
        ("8  Monitoring", "MLOps + informatics", "M03", ORANGE),
        ("9  Evidence", "trialists + clinicians", "M04 · M03", RED),
    ]
    w, h = 3.3, 1.9
    xs = [0.35, 4.35, 8.35]
    ys = [6.6, 3.85, 1.1]
    order = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
    centers = []
    for (row, col), (name, owner, mod, color) in zip(order, stages):
        x, y = xs[col], ys[row]
        _box(ax, x, y, w, h, color)
        ax.text(x + 0.2, y + h - 0.35, name, fontsize=17, weight="bold", color=color, va="top")
        ax.text(x + 0.2, y + 0.95, owner, fontsize=13, color=INK, va="center")
        ax.text(x + 0.2, y + 0.38, mod, fontsize=13.5, color=MUTED, va="center", weight="bold")
        centers.append((x, y))

    # Serpentine arrows between neighbouring boxes
    for i in range(len(order) - 1):
        (r1, c1), (r2, c2) = order[i], order[i + 1]
        x1, y1 = centers[i]
        x2, y2 = centers[i + 1]
        if r1 == r2:
            if c2 > c1:
                _arrow(ax, (x1 + w + 0.05, y1 + h / 2), (x2 - 0.05, y2 + h / 2))
            else:
                _arrow(ax, (x1 - 0.05, y1 + h / 2), (x2 + w + 0.05, y2 + h / 2))
        else:
            _arrow(ax, (x1 + w / 2, y1 - 0.05), (x2 + w / 2, y2 + h + 0.05))

    ax.text(6.0, 0.45, "M05 (LLM systems) and M06 (standards, privacy) cut across every stage",
            ha="center", fontsize=13, color=MUTED, style="italic")
    ax.set_title("The clinical AI lifecycle: who owns what", loc="left", pad=6)
    return save(fig, "m00-01-lifecycle")


def fig_evidence_ladder():
    """Staircase of evidence, from retrospective AUROC to prospective impact."""
    fig, ax = figure(11, 6.6)
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 6.9)
    ax.axis("off")

    rungs = [
        ("Retrospective\nAUROC", "same site,\nrandom split", MUTED),
        ("Temporal +\nexternal", "later years,\nother hospitals", BLUE),
        ("Silent\ndeployment", "live data,\nalerts hidden", PURPLE),
        ("Prospective\nimpact", "trial or\nstaged rollout", GREEN),
    ]
    step_w, step_h = 2.8, 1.25
    for i, (name, sub, color) in enumerate(rungs):
        x = 0.1 + i * step_w
        y = 0.2
        height = 2.0 + step_h * i
        _box(ax, x, y, step_w - 0.2, height, color, alpha=0.18)
        ax.text(x + 0.18, y + height - 0.2, name, fontsize=16, weight="bold", color=color,
                va="top", linespacing=1.1)
        ax.text(x + 0.18, y + height - 1.1, sub, fontsize=13, color=INK, va="top",
                linespacing=1.1)

    ax.text(0.2, 2.45, "Most published\nmodels stop here", fontsize=14, color=RED,
            weight="bold", va="bottom", linespacing=1.1)
    ax.text(0.2, 6.3, "each rung answers a harder question  \u2192", fontsize=14, color=MUTED,
            style="italic")
    ax.set_title("The evidence ladder", loc="left", pad=6)
    return save(fig, "m00-01-evidence-ladder")
