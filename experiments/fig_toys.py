"""Figure: what each method's predictive distribution looks like on the 1-d toys."""

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gpnn import runner

TOYS = ["hetero", "skew", "heavy", "bimodal", "inverse"]
METHODS = ["sgn", "mdn", "quantile", "gauss_hetero"]
LABEL = dict(sgn="stochastic graph net", mdn="mixture density net", quantile="quantile regression", gauss_hetero="heteroscedastic Gaussian")
RANGE = dict(inverse=(-0.15, 1.15))
G, MS = 160, 60  # grid points, samples drawn per grid point
OUT = runner.RESULTS / "fig_toys.npz"


def grid(toy):
    lo, hi = RANGE.get(toy, (-1, 1))
    return np.linspace(lo, hi, G)[:, None]


def one(job):
    toy, method, core = job
    os.sched_setaffinity(0, {core})
    import importlib

    import jax

    from gpnn import data

    build = importlib.import_module("experiments.03_baselines").build
    d = data.toy(toy, x_test=grid(toy), m_truth=MS)
    r = build(method)(d, jax.random.key(0))
    return toy, method, r["S"][:, :MS] * d.ys + d.ym, d.truth * d.ys + d.ym


if __name__ == "__main__":
    if "run" in sys.argv:
        jobs = [(t, m, i % 4) for i, (t, m) in enumerate((t, m) for t in TOYS for m in METHODS)]
        out = {}
        with ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as ex:
            for toy, method, S, T in ex.map(one, jobs):
                out[f"{toy}|{method}"], out[f"{toy}|truth"] = S, T
                print(toy, method, flush=True)
        np.savez_compressed(OUT, **out)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(OUT)
    rows = ["truth"] + METHODS
    fig, axes = plt.subplots(len(rows), len(TOYS), figsize=(2.6 * len(TOYS), 1.9 * len(rows)), sharex="col", sharey="col")
    for j, toy in enumerate(TOYS):
        x = np.repeat(grid(toy)[:, 0], MS)
        T = z[f"{toy}|truth"]
        lo, hi = np.quantile(T, 0.001), np.quantile(T, 0.999)
        for i, row in enumerate(rows):
            ax = axes[i, j]
            ax.scatter(x, z[f"{toy}|{row}"].ravel(), s=0.6, c="0.35" if row == "truth" else "C0", alpha=0.25, linewidths=0, rasterized=True)
            ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(toy, fontsize=10)
            if j == 0:
                ax.set_ylabel("true p(y|x)" if row == "truth" else LABEL[row], fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/toys.png", dpi=170)
