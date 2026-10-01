"""Figures for lesson m04-05-competing-risks-eval."""

from __future__ import annotations

import numpy as np

from _style import BLUE, INK, MUTED, RED, figure, save


def _simulate(n: int = 3000, seed: int = 7):
    """Recurrent stroke (cause 1) and death (cause 2) as competing exponential risks."""
    rng = np.random.default_rng(seed)
    t_stroke = rng.exponential(1 / 0.06, n)  # 6% per year
    t_death = rng.exponential(1 / 0.12, n)  # 12% per year (older post-stroke population)
    t_cens = rng.uniform(1.0, 6.0, n)  # staggered entry, administrative end of study
    t_event = np.minimum(t_stroke, t_death)
    time = np.minimum(t_event, t_cens)
    cause = np.where(t_cens < t_event, 0, np.where(t_stroke < t_death, 1, 2))
    return time, cause


def _one_minus_km(time, cause, k):
    """1 - KM for cause k, treating every other outcome (including death) as censoring."""
    order = np.argsort(time)
    t, c = time[order], cause[order]
    uniq = np.unique(t[c == k])
    surv, out = 1.0, []
    for u in uniq:
        at_risk = np.sum(t >= u)
        d = np.sum((t == u) & (c == k))
        surv *= 1 - d / at_risk
        out.append(1 - surv)
    return uniq, np.array(out)


def _aalen_johansen(time, cause, k):
    """Aalen-Johansen cumulative incidence for cause k: sum of S(t-) * d_k / n."""
    uniq = np.unique(time[cause > 0])
    s_prev, cif, out = 1.0, 0.0, []
    for u in uniq:
        at_risk = np.sum(time >= u)
        d_k = np.sum((time == u) & (cause == k))
        d_all = np.sum((time == u) & (cause > 0))
        cif += s_prev * d_k / at_risk
        s_prev *= 1 - d_all / at_risk
        out.append(cif)
    return uniq, np.array(out)


def fig_km_vs_cif():
    """1 - KM overstates recurrent-stroke risk when death competes; the CIF does not."""
    time, cause = _simulate()
    fig, ax = figure(10, 6.5)
    t1, km = _one_minus_km(time, cause, 1)
    t2, aj = _aalen_johansen(time, cause, 1)
    keep1, keep2 = t1 <= 5.0, t2 <= 5.0
    ax.step(t1[keep1], km[keep1], where="post", color=RED, lw=3)
    ax.step(t2[keep2], aj[keep2], where="post", color=BLUE, lw=3)
    ax.set_xlim(0, 6.1)
    ax.set_xticks(range(6))
    ax.set_ylim(0, 0.32)
    ax.set_xlabel("Years since randomization")
    ax.set_ylabel("Cumulative risk of recurrent stroke")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y")

    y_km = km[keep1][-1]
    y_aj = aj[keep2][-1]
    ax.text(1.9, 0.215, "1 − KM\n(death treated as censored)", color=RED,
            fontsize=15, weight="bold", ha="center")
    ax.text(3.4, 0.045, "Aalen–Johansen CIF\n(death as competing risk)", color=BLUE,
            fontsize=15, weight="bold", ha="center")
    ax.annotate("", xy=(5.1, y_km), xytext=(5.1, y_aj),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.8})
    ax.text(5.2, (y_km + y_aj) / 2, f"{y_km:.0%}\nvs\n{y_aj:.0%}", ha="left", va="center",
            fontsize=15, color=INK, weight="bold")
    ax.text(0.05, 0.3, "simulated: stroke 6%/yr, death 12%/yr", fontsize=12,
            color=MUTED, style="italic")
    ax.set_title("Ignoring death inflates stroke risk", loc="left")
    return save(fig, "m04-05-km-vs-cif")


if __name__ == "__main__":
    print(fig_km_vs_cif())
