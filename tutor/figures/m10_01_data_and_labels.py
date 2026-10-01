"""Figures for lesson m10-01-data-and-labels."""

from __future__ import annotations

import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, plt, save


def fig_dicom_hierarchy():
    """Patient > Study > Series > Instance, with where PHI and label shortcuts can hide."""
    fig, ax = figure(11, 8.2)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    levels = [
        ("Patient", "one person", BLUE),
        ("Study", "one stroke-code visit", GREEN),
        ("Series", "one scan run", ORANGE),
        ("Instance", "one slice = one file", PURPLE),
    ]
    phi = [
        "name, ID, birth date",
        "dates, accession no.,\nphysician names",
        "dose-report screenshot\n(text in pixels)",
        "burned-in text, face in\nvolume, private tags",
    ]
    lab = [
        "(none by itself)",
        "study description\n\"CODE STROKE\"",
        "which series exist\n(CTA present?), kernel",
        "scanner model,\nsite name, noise texture",
    ]

    ax.text(1.75, 7.75, "Level", ha="center", fontsize=15, weight="bold", color=INK)
    ax.text(5.95, 7.75, "PHI can hide in", ha="center", fontsize=15, weight="bold", color=RED)
    ax.text(9.15, 7.75, "Label or shortcut leaks", ha="center", fontsize=15, weight="bold", color=ORANGE)

    row_h, gap = 1.3, 0.38
    top = 7.3
    for i, (name, sub, col) in enumerate(levels):
        y = top - (i + 1) * row_h - i * gap
        ax.add_patch(FancyBboxPatch((0.15, y), 3.2, row_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=col, ec="none", alpha=0.18))
        ax.add_patch(FancyBboxPatch((0.15, y), 3.2, row_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="none", ec=col, lw=2.5))
        ax.text(1.75, y + row_h * 0.66, name, ha="center", va="center", fontsize=19, weight="bold", color=col)
        ax.text(1.75, y + row_h * 0.27, sub, ha="center", va="center", fontsize=13, color=INK)

        ax.add_patch(FancyBboxPatch((3.75, y), 4.2, row_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=RED, ec="none", alpha=0.10))
        ax.text(5.85, y + row_h / 2, phi[i], ha="center", va="center", fontsize=13.5, color=INK)

        ax.add_patch(FancyBboxPatch((8.05, y), 2.85, row_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=ORANGE, ec="none", alpha=0.14))
        ax.text(9.47, y + row_h / 2, lab[i], ha="center", va="center", fontsize=12.5,
                color=MUTED if i == 0 else INK)

        if i < len(levels) - 1:
            ax.annotate("", xy=(1.75, y - gap + 0.02), xytext=(1.75, y - 0.01),
                        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2})

    ax.text(0.15, 0.05, "Each level has its own UID (Study / Series / SOP Instance).", fontsize=12,
            color=MUTED, style="italic")
    ax.set_title("DICOM: one hierarchy, two kinds of leaks", loc="left", pad=6)
    return save(fig, "m10-01-dicom-hierarchy")


def _blob_mask(n, cx, cy, radius, phase, rng, boundary_sd):
    """A smooth irregular blob; boundary_sd is the reader's random boundary error in pixels."""
    yy, xx = np.mgrid[0:n, 0:n]
    dx, dy = xx - cx, yy - cy
    rho = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    r_true = radius * (1 + 0.12 * np.sin(3 * theta + phase))
    noise = np.zeros_like(theta)
    k_harm = int(np.clip(radius / 2, 2, 6))  # small lesions cannot carry high-frequency wiggle
    amp = boundary_sd * np.sqrt(2.0 / k_harm)
    for k in range(2, 2 + k_harm):
        noise += amp * np.sin(k * theta + rng.uniform(0, 2 * np.pi))
    return rho <= r_true + noise, r_true, theta


def _dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else np.nan


def fig_reader_ceiling():
    """Two readers with the same boundary disagreement: Dice depends on lesion size."""
    rng = np.random.default_rng(10)
    n = 160
    diameters = np.array([5, 7, 10, 14, 20, 28, 40, 56])
    curves = {}
    for sd in (0.5, 1.0):
        means = []
        for d in diameters:
            vals = []
            for _ in range(40):
                phase = rng.uniform(0, 2 * np.pi)
                a, _, _ = _blob_mask(n, n / 2, n / 2, d / 2, phase, rng, sd)
                b, _, _ = _blob_mask(n, n / 2, n / 2, d / 2, phase, rng, sd)
                vals.append(_dice(a, b))
            means.append(np.nanmean(vals))
        curves[sd] = np.array(means)

    fig = plt.figure(figsize=(11, 5.8))
    gs = GridSpec(2, 2, width_ratios=[1.55, 1], figure=fig, wspace=0.12, hspace=0.3,
                  left=0.09, right=0.98, top=0.84, bottom=0.14)
    fig.text(0.09, 0.93, "Same boundary error, very different Dice", fontsize=20, weight="bold")
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(diameters, curves[0.5], "-o", color=BLUE, lw=3, ms=8, label="boundary error 0.5 px")
    ax.plot(diameters, curves[1.0], "-o", color=ORANGE, lw=3, ms=8, label="boundary error 1.0 px")
    ax.axhline(0.8, color=MUTED, lw=1.5, ls="--")
    ax.text(57, 0.81, "0.80", color=MUTED, fontsize=12, ha="right", va="bottom")
    ax.set_xlabel("lesion diameter (pixels)")
    ax.set_ylabel("Dice between two readers")
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(2, 58)
    ax.legend(loc="lower right")

    demos = [(6, "small lesion"), (36, "large lesion")]
    for row, (d, label) in enumerate(demos):
        axd = fig.add_subplot(gs[row, 1])
        m = 96
        rs = np.random.default_rng(3 + row)
        phase = 0.7
        a, _, _ = _blob_mask(m, m / 2, m / 2, d / 2, phase, rs, 1.0)
        b, _, _ = _blob_mask(m, m / 2, m / 2, d / 2, phase, rs, 1.0)
        axd.contour(a.astype(float), levels=[0.5], colors=[BLUE], linewidths=3)
        axd.contour(b.astype(float), levels=[0.5], colors=[ORANGE], linewidths=3)
        half = d * 0.75
        axd.set_aspect("equal")
        axd.set_xlim(m / 2 - half, m / 2 + half)
        axd.set_ylim(m / 2 - half, m / 2 + half)
        axd.axis("off")
        axd.set_title(f"{label} ({d} px): Dice {_dice(a, b):.2f}", fontsize=14, pad=2)
    return save(fig, "m10-01-reader-ceiling")
