"""Figures for lesson m10-02-evaluation."""

from __future__ import annotations

import numpy as np

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, plt, save


# ---------------------------------------------------------------- metrics (numpy only)
def _disc(n, cx, cy, r):
    yy, xx = np.mgrid[0:n, 0:n]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _surface(mask):
    """Boundary pixels: mask pixels with at least one 4-neighbour outside the mask."""
    p = np.pad(mask, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return np.argwhere(mask & ~inner).astype(float)


def dice(a, b):
    return 2.0 * (a & b).sum() / (a.sum() + b.sum())


def surface_distances(a, b):
    """Distances from each surface point of a to the nearest surface point of b, and back."""
    sa, sb = _surface(a), _surface(b)
    d = np.sqrt(((sa[:, None, :] - sb[None, :, :]) ** 2).sum(-1))
    return d.min(1), d.min(0)


def hausdorff(a, b):
    d_ab, d_ba = surface_distances(a, b)
    return max(d_ab.max(), d_ba.max())


def nsd(a, b, tol):
    """Normalized surface distance: fraction of surface points within tol of the other surface."""
    d_ab, d_ba = surface_distances(a, b)
    return ((d_ab <= tol).sum() + (d_ba <= tol).sum()) / (len(d_ab) + len(d_ba))


def _lesions_found(truth_blobs, pred):
    return sum(int((blob & pred).any()) for blob in truth_blobs), len(truth_blobs)


# ---------------------------------------------------------------- figure 1
def fig_dice_failure_cases():
    """Three synthetic cases where Dice and surface/detection metrics disagree."""
    n = 128
    big = _disc(n, 48, 64, 30)
    small = _disc(n, 108, 64, 4)
    blob_far = _disc(n, 106, 22, 8)
    tiny = _disc(n, 64, 64, 2)

    cases = []
    # A: large lesion segmented well, small second lesion missed completely.
    gt_a = big | small
    pred_a = _disc(n, 49, 64, 30)
    cases.append(("Misses the small lesion", gt_a, pred_a, [big, small]))
    # B: large lesion segmented well, plus a spurious blob far away.
    pred_b = _disc(n, 49, 64, 30) | blob_far
    cases.append(("Adds a far false positive", big, pred_b, [big]))
    # C: tiny lesion, prediction shifted by one pixel.
    pred_c = _disc(n, 65, 64, 2)
    cases.append(("One-pixel shift, tiny lesion", tiny, pred_c, [tiny]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6))
    results = []
    for ax, (title, gt, pred, blobs) in zip(axes, cases):
        ax.imshow(np.where(gt, 1.0, np.nan), cmap="Blues", vmin=0, vmax=1.6, alpha=0.75,
                  origin="upper", interpolation="nearest")
        ax.contour(pred.astype(float), levels=[0.5], colors=[ORANGE], linewidths=3)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(n - 1, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(MUTED)
        d, h, s2 = dice(gt, pred), hausdorff(gt, pred), nsd(gt, pred, 2.0)
        found, total = _lesions_found(blobs, pred)
        results.append((title, d, h, s2, found, total))
        ax.set_title(title, fontsize=15, pad=8)
        ax.set_xlabel(f"Dice {d:.2f}   HD {h:.0f} px\nNSD(2 px) {s2:.2f}   lesions found {found}/{total}",
                      fontsize=14, labelpad=10)
    # legend built by hand, top left of figure
    fig.text(0.02, 0.965, "Blue fill: reference.  Orange line: model.", fontsize=14, color=INK, va="top")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.2, wspace=0.08)
    path = save(fig, "m10-02-dice-failure-cases")
    for r in results:
        print("   ", r)
    return path


# ---------------------------------------------------------------- figure 2
def fig_spectrum_bias():
    """Same model, two test sets with different case mix: overall sensitivity differs."""
    rng = np.random.default_rng(2)
    n = 4000
    bins = [0, 1, 5, 30, np.inf]
    labels = ["<1 mL", "1-5 mL", "5-30 mL", ">30 mL"]

    def sens_curve(v):  # assumed detectability of a bleed of volume v (mL)
        return 1 / (1 + np.exp(-(np.log(v) - np.log(1.0)) / 0.9))

    sets = {
        "Curated set": rng.lognormal(np.log(12), 1.0, n),
        "Consecutive ED": rng.lognormal(np.log(4), 1.1, n),
    }
    per_bin_sens = None
    mix, overall = {}, {}
    for name, v in sets.items():
        detected = rng.random(n) < sens_curve(v)
        idx = np.digitize(v, bins[1:-1])
        mix[name] = np.bincount(idx, minlength=4) / n
        overall[name] = detected.mean()
        if per_bin_sens is None:
            allv = np.concatenate(list(sets.values()))
            alld = rng.random(allv.size) < sens_curve(allv)
            ai = np.digitize(allv, bins[1:-1])
            per_bin_sens = np.array([alld[ai == k].mean() for k in range(4)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})

    x = np.arange(4)
    ax1.bar(x, per_bin_sens, color=BLUE, width=0.65)
    for xi, s in zip(x, per_bin_sens):
        ax1.text(xi, s + 0.02, f"{s:.2f}", ha="center", fontsize=14, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=13)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("sensitivity of the same model")
    ax1.set_xlabel("hemorrhage volume")
    ax1.set_title("One model, by size", fontsize=17, loc="left")

    w = 0.36
    colors = [GREEN, RED]
    for k, (name, m) in enumerate(mix.items()):
        ax2.bar(x + (k - 0.5) * w, m * 100, width=w, color=colors[k],
                label=f"{name}: overall sens. {overall[name]:.2f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=13)
    ax2.set_ylabel("% of positive cases")
    ax2.set_xlabel("hemorrhage volume")
    ax2.set_ylim(0, 75)
    ax2.legend(loc="upper right", fontsize=13)
    ax2.set_title("Two case mixes", fontsize=17, loc="left")

    fig.suptitle("Overall sensitivity depends on case mix", x=0.02, ha="left", fontsize=20,
                 weight="bold", y=0.99)
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.08, right=0.98, wspace=0.28)
    path = save(fig, "m10-02-spectrum-bias")
    print("    per-bin sens", np.round(per_bin_sens, 3), "overall", {k: round(v, 3) for k, v in overall.items()},
          "mix", {k: np.round(v, 2) for k, v in mix.items()})
    return path
