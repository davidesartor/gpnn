"""Figures for the README: python experiments/figures.py [name ...]"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gpnn import runner

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 5,
    }
)
TOYS = ["hetero", "skew", "heavy", "bimodal", "inverse", "friedman"]
TOY_TITLE = dict(
    hetero="hetero: Gaussian, varying width",
    skew="skew: one-sided noise",
    heavy="heavy: Student-t tails",
    bimodal="bimodal: one branch splits in two",
    inverse="inverse: up to three branches",
    friedman="friedman: 10 inputs, skewed noise",
)


def _mean(rows, **match):
    def get(metric):
        v = [r[metric] for r in rows if all(r[k] == val for k, val in match.items()) and metric in r]
        return float(np.mean(v)) if v else np.nan

    return get


def samples_needed():
    """Exp 1: distance to the true conditional vs number of sampled paths per example."""
    rows = runner.load("01_objectives")
    ks = [1, 2, 4, 8, 32]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
    for ax, toy in zip(axes.ravel(), TOYS):
        base = _mean(rows, dataset=toy, method="gauss_homo")("w1")
        iw = [_mean(rows, dataset=toy, method=f"iw:K={k}")("w1") for k in ks]
        es = [_mean(rows, dataset=toy, method=f"es_rb:K={k}")("w1") for k in ks[1:]]
        ax.axhline(base, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.plot(ks, iw, color=BLUE, marker="o", label="likelihood bound (importance-weighted)")
        ax.plot(ks[1:], es, color=ORANGE, marker="o", label="energy score")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_ylim(0, max(base, max(iw)) * 1.15)
        ax.set_title(TOY_TITLE[toy], loc="left")
        if toy == "hetero":
            ax.annotate("deterministic net + fixed noise", (32, base), xytext=(0, 4), textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("sampled paths per training example (K)")
    for ax in axes[:, 0]:
        ax.set_ylabel("W1 distance to true p(y|x)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("figures/samples_needed.png", dpi=170)


def gap():
    """Exp 7A: predictive bands where there is no training data."""
    z = np.load(runner.RESULTS / "07_gap.npz")
    x = np.linspace(-1.6, 1.6, 161)
    panels = [("sgn", "stochastic net"), ("sgn_ens5", "5 stochastic nets, pooled"), ("mdn", "mixture density net"), ("ensemble", "deep ensemble (5 Gaussian nets)"), ("gp", "Gaussian process")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.1), sharey=True)
    tq = z["truth|q"]
    for ax, (tag, title) in zip(axes, panels):
        q = z[f"{tag}|0|q"]
        for a, b in ((-1.6, -1.0), (-0.2, 0.4), (1.0, 1.6)):
            ax.axvspan(a, b, color="#efeeea", lw=0)
        ax.fill_between(x, q[0], q[4], color=BLUE, alpha=0.22, lw=0, label="predicted 90% interval")
        ax.plot(x, q[2], color=BLUE, label="predicted median")
        ax.plot(x, tq[0], color=INK, lw=1, ls=(0, (4, 3)), label="true 5% / 95% quantiles")
        ax.plot(x, tq[4], color=INK, lw=1, ls=(0, (4, 3)))
        ax.set_title(title, loc="left")
        ax.set_xlabel("x")
        ax.set_ylim(-3.2, 3.2)
        ax.grid(False)
    axes[0].set_ylabel("y")
    axes[0].annotate("no training data\nin shaded regions", (0.1, -3.0), ha="center", va="bottom", color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig("figures/gap.png", dpi=170)


def well():
    """Exp 5: sampled 12-step trajectories for one start point near the unstable equilibrium."""
    from gpnn import multi

    d = multi.Split("well", seed=0)
    i = int(np.argmin(np.abs(d.xte[:6, 0])))  # of the saved test points, the one starting closest to 0
    panels = [("truth", "true process"), ("sgn", "stochastic net"), ("mdn", "mixture density net"), ("mv_gauss", "full-covariance Gaussian"), ("quantile_indep", "per-step quantiles,\nindependent ranks")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.0), sharey=True)
    t = np.arange(1, 13)
    for ax, (tag, title) in zip(axes, panels):
        S = d.truth[i] if tag == "truth" else np.load(runner.RESULTS / f"05_samples_well_{tag}.npy")[i]
        S = S[:60] * d.ys + d.ym
        ax.plot(t, S.T, color=INK if tag == "truth" else BLUE, lw=0.7, alpha=0.45)
        ax.set_title(title, loc="left")
        ax.set_xlabel("step ahead")
        ax.set_xticks([1, 4, 8, 12])
        ax.set_ylim(-2.2, 2.2)
    axes[0].set_ylabel("y")
    fig.tight_layout()
    fig.savefig("figures/well.png", dpi=170)


def size():
    """Exp 7B: coverage of the nominal 90% interval as the training set shrinks."""
    sizes = [100, 300, 1000, 3000]
    series = [("sgn", "stochastic net", BLUE), ("sgn_temp", "stochastic net + held-out noise rescaling", ORANGE), ("ensemble", "deep ensemble", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True)
    for ax, ds in zip(axes, ("friedman", "hetero")):
        ax.axhline(0.9, color=MUTED, lw=1, ls=(0, (4, 3)))
        for tag, label, color in series:
            y = [_mean(runner.load(f"07_size_n{n}"), dataset=ds, method=tag)("cov90") for n in sizes]
            ax.plot(sizes, y, color=color, marker="o", label=label)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.set_xlabel("training points")
        ax.set_title(TOY_TITLE[ds], loc="left")
        ax.set_ylim(0.6, 1.0)
    axes[0].set_ylabel("test coverage of 90% interval")
    axes[0].annotate("nominal", (3000, 0.9), xytext=(0, 4), textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig("figures/size.png", dpi=170)


FIGS = dict(samples_needed=samples_needed, gap=gap, well=well, size=size)

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
        print("wrote", name)
