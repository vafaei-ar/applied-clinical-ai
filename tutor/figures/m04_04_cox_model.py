"""Figures for lesson m04-04-cox-model."""

from __future__ import annotations

import numpy as np

from _style import BLUE, GRID, MUTED, ORANGE, figure, save


def _km(time: np.ndarray, event: np.ndarray):
    ts = np.unique(time[event == 1])
    s, out_t, out_s = 1.0, [], []
    for t in ts:
        n = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        s *= 1 - d / n
        out_t.append(t)
        out_s.append(s)
    return np.array(out_t), np.array(out_s)


def _weibull(rng, n, scale, shape, censor=36.0):
    t = scale * rng.weibull(shape, n)
    event = (t <= censor).astype(int)
    return np.minimum(t, censor), event


def fig_loglog():
    """log(-log S) vs log t: parallel lines under PH, crossing lines when PH fails."""
    rng = np.random.default_rng(404)
    panels = [
        ("PH holds: parallel", (_weibull(rng, 800, 60, 1.2), _weibull(rng, 800, 60 * 1.3, 1.2))),
        ("PH fails: lines cross", (_weibull(rng, 800, 60, 1.0), _weibull(rng, 800, 27, 1.8))),
    ]
    fig, axes = figure(12, 6.2)
    axes.remove()
    for i, (title, arms) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1)
        for (t, e), color, label in zip(arms, (ORANGE, BLUE), ("Control", "Treatment")):
            xs, s = _km(t, e)
            keep = (s > 0) & (s < 1) & (xs >= 1)
            ax.step(np.log(xs[keep]), np.log(-np.log(s[keep])), where="post", color=color, lw=3,
                    label=label)
        ax.set_title(title, loc="left", fontsize=18, pad=8)
        ax.set_xticks(np.log([1, 3, 10, 30]))
        ax.set_xticklabels(["1", "3", "10", "30"])
        ax.set_xlabel("Months (log scale)")
        ax.set_ylim(-4.6, 0.7)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel("log(−log S(t))")
            ax.legend(loc="lower right")
    fig.text(0.01, 0.005, "simulated Weibull data", fontsize=11, color=MUTED, style="italic")
    return save(fig, "m04-04-loglog")
