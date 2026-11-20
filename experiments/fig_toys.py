"""Samples from each method on a grid of inputs, for the `toys` figure (see figures.py)."""

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gpnn import runner

TOYS = ["hetero", "skew", "heavy", "bimodal", "inverse"]
METHODS = ["sgn", "mdn", "quantile", "gauss_hetero"]
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
    r = build(method)(d, jax.random.key(0), act="relu")
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
