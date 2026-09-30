"""Figures for lesson m10-03-regulation-deployment."""

from __future__ import annotations

import heapq

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, save


def _box(ax, x, y, w, h, color, fill_alpha=0.16, lw=2.5, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="none", alpha=fill_alpha))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=color, lw=lw, ls=ls))


def _arrow(ax, x0, y0, x1, y1, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22,
                                 color=color, lw=2.2, shrinkA=0, shrinkB=0))


def fig_fda_pathway():
    """A decision sketch for an imaging AI product: claim, then route to market."""
    fig, ax = figure(12, 7.2)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, 7.2)
    ax.axis("off")

    # column 1: is it image analysis?
    _box(ax, 0.15, 3.0, 2.9, 1.9, BLUE)
    ax.text(1.6, 4.25, "Does the software\nanalyze a\nmedical image?", ha="center", va="center",
            fontsize=15, weight="bold", color=BLUE)
    ax.text(1.6, 3.35, "yes: device software", ha="center", va="center", fontsize=12.5, color=INK)
    ax.text(1.6, 2.5, "no: check the four\nnon-device CDS criteria\n(Module 03)", ha="center", va="top",
            fontsize=12, color=MUTED)

    # column 2: what does it claim?
    ax.text(5.65, 6.75, "What does it claim?", ha="center", fontsize=16, weight="bold", color=INK)
    claims = [
        ("Triage and notify", "CADt  ·  21 CFR 892.2080", GREEN),
        ("Mark findings", "CADe  ·  892.2070", ORANGE),
        ("Characterize a lesion", "CADx  ·  892.2060", PURPLE),
        ("Measure or process images", "892.2050  ·  code LLZ", BLUE),
    ]
    y0, h, gap = 5.25, 1.0, 0.33
    ys = []
    for i, (name, sub, col) in enumerate(claims):
        y = y0 - i * (h + gap)
        ys.append(y)
        _box(ax, 3.75, y, 3.8, h, col)
        ax.text(5.65, y + h * 0.68, name, ha="center", va="center", fontsize=14.5, weight="bold", color=col)
        ax.text(5.65, y + h * 0.27, sub, ha="center", va="center", fontsize=12.5, color=INK)
    _arrow(ax, 3.1, 3.95, 3.7, 3.95)

    # column 3: route to market
    ax.text(10.0, 6.75, "Route to market", ha="center", fontsize=16, weight="bold", color=INK)
    routes = [
        ("510(k)", "class and predicate exist", GREEN),
        ("De Novo", "new type, low to\nmoderate risk", ORANGE),
        ("PMA", "high risk", RED),
    ]
    ry0, rh, rgap = 5.05, 1.1, 0.4
    for i, (name, sub, col) in enumerate(routes):
        y = ry0 - i * (rh + rgap)
        _box(ax, 8.35, y, 3.5, rh, col)
        ax.text(10.1, y + rh * 0.68, name, ha="center", va="center", fontsize=16, weight="bold", color=col)
        ax.text(10.1, y + rh * 0.26, sub, ha="center", va="center", fontsize=12.5, color=INK)
    # bracket arrow from column 2 to column 3
    ax.plot([7.65, 7.95, 7.95], [ys[0] + 0.5, ys[0] + 0.5, 0.0], alpha=0)  # keep axes limits stable
    ax.plot([7.65, 7.95], [ys[0] + 0.5, ys[0] + 0.5], color=INK, lw=2.2)
    ax.plot([7.65, 7.95], [ys[-1] + 0.5, ys[-1] + 0.5], color=INK, lw=2.2)
    ax.plot([7.95, 7.95], [ys[-1] + 0.5, ys[0] + 0.5], color=INK, lw=2.2)
    _arrow(ax, 7.95, 3.6, 8.3, 3.6)

    # PCCP banner
    _box(ax, 3.75, 0.25, 8.1, 0.85, PURPLE, fill_alpha=0.10, ls="--")
    ax.text(7.8, 0.68, "Optional: a PCCP pre-authorizes planned model updates",
            ha="center", va="center", fontsize=13.5, color=PURPLE, weight="bold")
    ax.text(0.15, -0.25, "Simplified. The four classifications shown are Class II with special controls.  "
            "Checked against FDA and CFR pages on 2026-09-30.", fontsize=11, color=MUTED, style="italic",
            va="center")
    ax.set_title("Imaging AI: from claim to route to market", loc="left", pad=6)
    return save(fig, "m10-03-fda-pathway")


# ------------------------------------------------------------------ triage queue simulation
def _simulate(rho, triage, n=60000, seed=0, mean_read=10.0, prev=0.10, sens=0.90, fpr=0.08):
    """One radiologist, non-preemptive priority queue. Returns wait (min), truth, flag."""
    rng = np.random.default_rng(seed)
    lam = rho / mean_read
    arr = np.cumsum(rng.exponential(1 / lam, n))
    serv = rng.exponential(mean_read, n)
    pos = rng.random(n) < prev
    flag = np.where(pos, rng.random(n) < sens, rng.random(n) < fpr)
    prio = np.where(flag & triage, 0, 1)
    wait = np.zeros(n)
    t, i, done, heap = 0.0, 0, 0, []
    while done < n:
        if not heap and i < n and arr[i] > t:
            t = arr[i]
        while i < n and arr[i] <= t:
            heapq.heappush(heap, (prio[i], arr[i], i))
            i += 1
        _, a, j = heapq.heappop(heap)
        wait[j] = t - a
        t += serv[j]
        done += 1
    return wait, pos, flag


def fig_triage_queue():
    """Mean wait before a study is opened, with and without a triage flag, by reader busyness."""
    rhos = np.array([0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    fifo, tp, fn = [], [], []
    for k, rho in enumerate(rhos):
        w0, pos, flag = _simulate(rho, False, seed=k)
        w1, _, _ = _simulate(rho, True, seed=k)
        fifo.append(w0.mean())
        tp.append(w1[pos & flag].mean())
        fn.append(w1[pos & ~flag].mean())

    fig, ax = figure(11, 6.0)
    ax.plot(rhos, fifo, "--o", color=MUTED, lw=3, ms=8, label="no triage (first come, first served)")
    ax.plot(rhos, fn, "-o", color=RED, lw=3, ms=8, label="triage: positives the model misses")
    ax.plot(rhos, tp, "-o", color=BLUE, lw=3, ms=8, label="triage: flagged true positives")
    ax.set_xlabel("how busy the reader is (utilization)")
    ax.set_ylabel("mean wait before opening (min)")
    ax.set_xlim(0.25, 0.98)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=13.5)
    ax.set_title("A triage flag helps most when there is a queue", loc="left", fontsize=18, pad=6)
    ax.text(0.98, -0.2, "Simulation: one reader, 10 min per study, 10% positives, flag sensitivity 0.90,\n"
            "8% of negatives flagged. Illustrative, not a forecast.", transform=ax.transAxes, ha="right",
            va="top", fontsize=11.5, color=MUTED, style="italic")
    path = save(fig, "m10-03-triage-queue")
    print("    rho", rhos.tolist())
    print("    fifo", np.round(fifo, 1).tolist())
    print("    tp  ", np.round(tp, 1).tolist())
    print("    fn  ", np.round(fn, 1).tolist())
    return path
