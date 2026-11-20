"""What does a model with no epistemic uncertainty get wrong?

Part A (gap): training inputs avoid (-0.2, 0.4) and stop at |x| = 1. Inside
the hole and beyond the edge nothing constrains the fit, and a model that only
describes noise has no way to say so.

Part B (size): coverage as the training set shrinks. Fewer points mean a worse
fit, and the error of the fit is exactly what an aleatoric-only model omits.
"""

import importlib
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gpnn import methods, runner

OUT_A = runner.RESULTS / "07_gap.npz"
GRID = np.linspace(-1.6, 1.6, 161)[:, None]
REGIONS = {
    "seen": lambda x: ((x >= -1) & (x <= -0.2)) | ((x >= 0.4) & (x <= 1)),
    "hole": lambda x: (x > -0.2) & (x < 0.4),
    "beyond": lambda x: np.abs(x) > 1,
}
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
A_TAGS = ["sgn", "sgn_temp", "sgn_ens5", "mdn", "gauss_hetero", "ensemble", "gp"]
B_TAGS = ["sgn", "sgn_temp", "sgn_ens5", "mdn", "quantile", "gauss_hetero", "ensemble", "gp"]
SIZES = [100, 300, 1000, 3000]
SGN = {
    "sgn": dict(objective="es_rb", K=8, noise="input"),
    "sgn_temp": dict(objective="es_rb", K=8, noise="input", refit="temp"),
    "sgn_ens5": dict(objective="es_rb", K=8, noise="input", n_ens=5),
}
HP = dict(act="relu")


def build(tag):
    return methods.make_sgn(**SGN[tag]) if tag in SGN else methods.BASELINES[tag]


def gap_one(job):
    tag, seed, core = job
    os.sched_setaffinity(0, {core})
    import jax

    from gpnn import data

    d = data.toy("hetero_gap", seed=seed, x_test=GRID, m_truth=4000)
    S = build(tag)(d, jax.random.key(seed), **HP)["S"]
    lo, hi = np.quantile(S, 0.05, axis=1), np.quantile(S, 0.95, axis=1)
    cc = ((d.truth >= lo[:, None]) & (d.truth <= hi[:, None])).mean(1)  # true coverage at each x
    to_raw = lambda a: a * d.ys + d.ym
    return tag, seed, to_raw(np.quantile(S, QS, axis=1)), cc, to_raw(np.quantile(d.truth, QS, axis=1))


if __name__ == "__main__":
    if "gap" in sys.argv:
        jobs = [(t, s, i % 4) for i, (t, s) in enumerate((t, s) for t in A_TAGS for s in range(3))]
        out = {}
        with ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as ex:
            for tag, seed, q, cc, tq in ex.map(gap_one, jobs):
                out[f"{tag}|{seed}|q"], out[f"{tag}|{seed}|cc"], out["truth|q"] = q, cc, tq
                print(tag, seed, flush=True)
        np.savez_compressed(OUT_A, **out)
    if "size" in sys.argv:
        for n in SIZES:
            runner.run(f"07_size_n{n}", build, B_TAGS, [("toy", "friedman", dict(n=n)), ("toy", "hetero", dict(n=n))], seeds=range(5), hp=HP)

    if OUT_A.exists():
        z = np.load(OUT_A)
        x = GRID[:, 0]
        print("true coverage of the nominal 90% interval, by region (mean over 3 seeds)")
        print(f"{'':16}" + "".join(f"{r:>10}" for r in REGIONS))
        for tag in A_TAGS:
            cc = np.mean([z[f"{tag}|{s}|cc"] for s in range(3)], 0)
            print(f"{tag:16}" + "".join(f"{cc[m(x)].mean():10.3f}" for m in REGIONS.values()))
    for metric in ("cov90", "ccov_err90", "w1"):
        for ds in ("friedman", "hetero"):
            rows = {n: runner.load(f"07_size_n{n}") for n in SIZES if (runner.RESULTS / f"07_size_n{n}.jsonl").exists()}
            if not rows:
                continue
            print(f"\n{metric} on {ds} vs training-set size\n{'':16}" + "".join(f"{n:>9}" for n in rows))
            for tag in B_TAGS:
                print(f"{tag:16}" + "".join(f"{np.mean([r[metric] for r in rs if r['method'] == tag and r['dataset'] == ds] or [np.nan]):9.3f}" for rs in rows.values()))
