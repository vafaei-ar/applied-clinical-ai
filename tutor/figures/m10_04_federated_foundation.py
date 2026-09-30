"""Figures for lesson m10-04-federated-foundation.

Everything here is simulated with numpy under fixed seeds. No real data or images.
"""

from __future__ import annotations

import numpy as np

from _style import BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RED, plt, save


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _aug(X):
    return np.c_[X, np.ones(len(X))]


def _auc(score, y):
    r = score.argsort().argsort() + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ------------------------------------------------------------------ federated averaging
def _define_sites(k, d, rng, noniid, w_star):
    """Define each site ONCE. Train and test data are drawn from the same site parameters."""
    sites = []
    for _ in range(k):
        if noniid:
            sites.append({
                "shift": rng.normal(0, 1, d) * 1.2,          # scanner-specific feature offset
                "bias": rng.uniform(-2.2, 0.8),              # site prevalence differs
                "w": w_star + rng.normal(0, 0.8, d),         # protocol or reader differences
            })
        else:
            sites.append({"shift": np.zeros(d), "bias": -0.7, "w": w_star})
    return sites


def _sample(site, n, rng):
    d = len(site["shift"])
    X = rng.normal(0, 1, (n, d)) + site["shift"]
    y = (rng.random(n) < _sigmoid(X @ site["w"] + site["bias"])).astype(float)
    return X, y


def _grad(w, X, y):
    Xb = _aug(X)
    return Xb.T @ (_sigmoid(Xb @ w) - y) / len(y)


def _loss(w, X, y):
    z = _aug(X) @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def _fedavg(train, rounds, local_steps, lr, eval_fn):
    d = train[0][0].shape[1]
    w = np.zeros(d + 1)
    ns = np.array([len(t[1]) for t in train], float)
    hist = []
    for _ in range(rounds):
        locals_ = []
        for X, y in train:
            wl = w.copy()
            for _ in range(local_steps):
                wl -= lr * _grad(wl, X, y)
            locals_.append(wl)
        w = np.average(locals_, axis=0, weights=ns)
        hist.append(eval_fn(w))
    return w, np.array(hist)


def _central(train, steps, lr):
    X = np.vstack([t[0] for t in train])
    y = np.concatenate([t[1] for t in train])
    w = np.zeros(X.shape[1] + 1)
    for _ in range(steps):
        w -= lr * _grad(w, X, y)
    return w


def _dispersion(train, w, local_steps, lr):
    """Client drift: mean distance of each site's locally trained model from their average."""
    ws = []
    for X, y in train:
        wl = w.copy()
        for _ in range(local_steps):
            wl -= lr * _grad(wl, X, y)
        ws.append(wl)
    ws = np.array(ws)
    return np.mean(np.linalg.norm(ws - ws.mean(0), axis=1))


def fig_fedavg_noniid():
    """FedAvg on 5 simulated sites: IID vs non-IID, few vs many local steps, and client drift."""
    k, d, n, lr = 5, 10, 300, 0.3
    seeds, rounds = 12, 300
    e_grid = [1, 2, 5, 10, 20, 50]
    curves = {"iid_50": [], "non_1": [], "non_50": []}
    disp = {"iid": [], "non": []}
    worst_gap = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        w_star = rng.normal(0, 1, d)
        for tag, noniid in (("iid", False), ("non", True)):
            sites = _define_sites(k, d, np.random.default_rng(seed + 100), noniid, w_star)
            train = [_sample(s, n, rng) for s in sites]
            test = [_sample(s, 3000, rng) for s in sites]
            Xt = np.vstack([t[0] for t in test])
            yt = np.concatenate([t[1] for t in test])
            eval_fn = lambda w: _loss(w, Xt, yt)  # noqa: E731
            w_c = _central(train, 3000, lr)
            base = eval_fn(w_c)
            if tag == "iid":
                _, h = _fedavg(train, rounds, 50, lr, eval_fn)
                curves["iid_50"].append(h - base)
            else:
                _, h1 = _fedavg(train, rounds, 1, lr, eval_fn)
                _, h50 = _fedavg(train, rounds, 50, lr, eval_fn)
                curves["non_1"].append(h1 - base)
                curves["non_50"].append(h50 - base)
                mean_auc = _auc(_aug(Xt) @ w_c, yt)
                worst_auc = min(_auc(_aug(x) @ w_c, y) for x, y in test)
                worst_gap.append((mean_auc, worst_auc))
            w10, _ = _fedavg(train, 10, 5, lr, eval_fn)
            disp[tag].append([_dispersion(train, w10, e, lr) for e in e_grid])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(1, rounds + 1)
    for key, col, lab, ls in (
        ("iid_50", MUTED, "IID sites, 50 local steps", "--"),
        ("non_1", BLUE, "non-IID, 1 local step", "-"),
        ("non_50", ORANGE, "non-IID, 50 local steps", "-"),
    ):
        ax1.plot(x, np.mean(curves[key], axis=0), ls, color=col, lw=3, label=lab)
    ax1.set_xscale("log")
    ax1.set_xticks([1, 10, 100, 300])
    ax1.set_xticklabels(["1", "10", "100", "300"])
    ax1.set_xlabel("communication rounds")
    ax1.set_ylabel("pooled test loss above\ncentralized training")
    ax1.set_ylim(-0.01, 0.3)
    ax1.legend(loc="upper right", fontsize=12.5)
    ax1.set_title("Many local steps stall above the optimum", fontsize=15, loc="left")

    for tag, col, lab in (("iid", MUTED, "IID sites"), ("non", ORANGE, "non-IID sites")):
        arr = np.array(disp[tag])
        ax2.plot(e_grid, arr.mean(0), "-o", color=col, lw=3, ms=8, label=lab)
    ax2.set_xscale("log")
    ax2.set_xticks(e_grid)
    ax2.set_xticklabels([str(e) for e in e_grid])
    ax2.set_xlabel("local steps per round")
    ax2.set_ylabel("client drift\n(spread of local models)")
    ax2.legend(loc="upper left", fontsize=12.5)
    ax2.set_title("Drift grows with local steps", fontsize=16, loc="left")

    fig.text(0.01, 0.01, "Simulation: 5 sites, logistic model, 300 patients per site, mean of 12 seeds.",
             fontsize=11.5, color=MUTED, style="italic")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.2, wspace=0.32)
    path = save(fig, "m10-04-fedavg-noniid")
    wg = np.array(worst_gap)
    print("    excess loss at round 300 (mean):", {k_: round(float(np.mean(v, axis=0)[-1]), 3) for k_, v in curves.items()})
    print("    excess loss at round 10  (mean):", {k_: round(float(np.mean(v, axis=0)[9]), 3) for k_, v in curves.items()})
    print("    drift E=50: iid %.3f  non-iid %.3f" % (np.mean(disp["iid"], 0)[-1], np.mean(disp["non"], 0)[-1]))
    print("    centralized non-IID: mean-site AUC %.3f, worst-site AUC %.3f" % tuple(wg.mean(0)))
    return path


# ------------------------------------------------------------------ differential privacy cost
def _eps_full_batch(sigma, steps, delta):
    """(eps, delta) after `steps` full-batch Gaussian-mechanism releases (add/remove adjacency).

    Renyi DP of the Gaussian mechanism is alpha / (2 sigma^2) per step; steps compose additively;
    eps = steps * alpha / (2 sigma^2) + log(1/delta) / (alpha - 1), minimized over alpha.
    This is a valid, simple upper bound. Subsampling (real DP-SGD) and tighter conversions give
    smaller eps for the same noise.
    """
    big_l = np.log(1 / delta)
    return steps / (2 * sigma ** 2) + np.sqrt(2 * steps * big_l) / sigma


def _sigma_for_eps(eps, steps, delta):
    lo, hi = 1e-3, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if _eps_full_batch(mid, steps, delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def _dp_gd(X, y, eps, rng, steps=100, clip=1.0, lr=20.0, delta=1e-5):
    n, d = X.shape
    Xb = _aug(X)
    w = np.zeros(d + 1)
    sigma = _sigma_for_eps(eps, steps, delta) if eps is not None else 0.0
    for _ in range(steps):
        g = (_sigmoid(Xb @ w) - y)[:, None] * Xb                      # per-example gradients
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        g = g * np.minimum(1.0, clip / np.maximum(norms, 1e-12))      # clip each to norm <= C
        total = g.sum(0)
        if eps is not None:
            total = total + rng.normal(0, sigma * clip, d + 1)        # noise on the sum
        w -= lr * total / n
    return w


def fig_dp_utility():
    """Test AUROC of a DP linear probe on 256-d unit-norm embeddings, by epsilon and training size."""
    d, scale = 256, 40.0
    rng0 = np.random.default_rng(0)
    u = rng0.normal(0, 1, d)
    u /= np.linalg.norm(u)

    def make(n, rng):
        X = rng.normal(0, 1, (n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        y = (rng.random(n) < _sigmoid(scale * (X @ u))).astype(float)
        return X, y

    Xte, yte = make(20000, np.random.default_rng(99))
    eps_grid = [1, 2, 4, 8, 16, 64]
    sizes = [(500, RED), (5000, ORANGE), (50000, BLUE)]
    results = {}
    for n, _ in sizes:
        row, nonpriv = [], []
        for eps in eps_grid + [None]:
            aucs = []
            for s in range(5):
                r = np.random.default_rng(1000 + s)
                X, y = make(n, r)
                w = _dp_gd(X, y, eps, r)
                aucs.append(_auc(_aug(Xte) @ w, yte))
            (nonpriv if eps is None else row).append(float(np.mean(aucs)))
        results[n] = (row, nonpriv[0])

    fig, ax = plt.subplots(figsize=(11, 6.0))
    for n, col in sizes:
        row, base = results[n]
        ax.plot(eps_grid, row, "-o", color=col, lw=3, ms=8, label=f"{n:,} training studies")
        ax.axhline(base, color=col, lw=1.3, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(eps_grid)
    ax.set_xticklabels([str(e) for e in eps_grid])
    ax.set_xlabel("privacy budget epsilon (smaller = stronger privacy)")
    ax.set_ylabel("test AUROC")
    ax.set_ylim(0.45, 0.95)
    ax.legend(loc="lower right", fontsize=13.5)
    ax.set_title("The same privacy costs small datasets far more", loc="left", fontsize=18, pad=6)
    ax.text(0.99, -0.2, "Simulation: linear probe on 256-d embeddings, delta = 1e-5, 100 full-batch steps.\n"
            "Dotted lines: no DP. Epsilon from a simple Renyi-DP bound (conservative).",
            transform=ax.transAxes, ha="right", va="top", fontsize=11.5, color=MUTED, style="italic")
    path = save(fig, "m10-04-dp-utility")
    for n, _ in sizes:
        print("    N=%d" % n, [round(v, 3) for v in results[n][0]], "no-DP %.3f" % results[n][1])
    return path


# ------------------------------------------------------------------ compute and emissions
def fig_compute_emissions():
    """Illustrative energy and CO2e for three training scenarios, with stated assumptions."""
    power_kw, pue, grid = 0.4, 1.2, 0.35   # kW per GPU (assumed), data-centre overhead, kg CO2e per kWh
    scenarios = [
        ("Fine-tune once\n1 GPU, 6 h", 1 * 6),
        ("Fine-tune with a search\n40 runs x 1 GPU x 6 h", 40 * 6),
        ("Pretrain from scratch\n64 GPUs x 7 days", 64 * 7 * 24),
    ]
    kg = [h * power_kw * pue * grid for _, h in scenarios]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = np.arange(len(scenarios))[::-1]
    cols = [GREEN, ORANGE, PURPLE]
    ax.barh(y, kg, color=cols, height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([s for s, _ in scenarios], fontsize=14)
    for yi, v, (_, h) in zip(y, kg, scenarios):
        label = f"{v:,.0f} kg CO2e" if v >= 10 else f"{v:.1f} kg CO2e"
        ax.text(v * 1.15, yi, f"{label}  ({h:,} GPU-h)", va="center", fontsize=13, color=INK)
    ax.set_xlim(0.5, 2e5)
    ax.set_xlabel("estimated emissions (kg CO2e, log scale)")
    ax.set_title("A search multiplies the cost; pretraining dominates", loc="left", fontsize=18, pad=6)
    ax.text(0.99, -0.22, "Assumptions (illustrative): 0.4 kW per GPU, PUE 1.2, 0.35 kg CO2e per kWh. "
            "Look up your own hardware and grid.", transform=ax.transAxes, ha="right", va="top",
            fontsize=11.5, color=MUTED, style="italic")
    path = save(fig, "m10-04-compute-emissions")
    print("    kg CO2e:", [round(v, 1) for v in kg])
    return path
