"""Do prior-sampled importance weights degenerate as the graph gets bigger?

A path's weight is p(y | path), which depends on the path only through the
output it produces. So the effective sample size (ESS) should not care about
depth or width, only about how concentrated y is relative to the model's
predictive spread, and about the dimension of y.

Part A: depth x width on scalar-output toys.
Part B: output dimension, using the first D steps of the double-well paths.
"""

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from gpnn import runner

OUT = runner.RESULTS / "06_importance.jsonl"
GRID = [(d, w) for d in (1, 2, 4, 8) for w in (16, 64, 256)]
DIMS = [1, 2, 4, 8, 12]


def one(job):
    os.sched_setaffinity(0, {job[-1]})
    import jax

    from gpnn import data, methods, multi

    part, name, a, b, seed, _ = job
    t = time.time()
    if part == "A":
        d = data.toy(name, seed=seed)
        r = methods.make_sgn("es_rb", K=8)(d, jax.random.key(seed), depth=a, width=b)
        row = dict(depth=a, width=b, dim=1)
    else:
        d = multi.Split("well", seed=seed, dims=a)
        r = multi.make_sgn("es", K=8)(d, jax.random.key(seed))
        row = dict(depth=2, width=64, dim=a)
    row.update(part=part, dataset=name, seed=seed, ess=r["info"]["ess"], sy=r["info"]["sy"], time=time.time() - t)
    return row


if __name__ == "__main__":
    if "run" in sys.argv:
        jobs = [("A", t, d, w, s) for t in ("skew", "inverse") for d, w in GRID for s in range(2)]
        jobs += [("B", "well", D, 0, s) for D in DIMS for s in range(2)]
        jobs = [(*j, i % 4) for i, j in enumerate(jobs)]
        with ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as ex, OUT.open("w") as f:
            for row in ex.map(one, jobs):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row, flush=True)
    import numpy as np

    rows = runner.load("06_importance")
    mean = lambda sel, k: np.mean([r[k] for r in rows if sel(r)])
    for name in ("skew", "inverse"):
        print(f"\n[{name}] median ESS fraction of 2000 prior paths (rows: depth, cols: width)")
        print(f"{'':8}" + "".join(f"{w:>9}" for w in (16, 64, 256)))
        for d in (1, 2, 4, 8):
            print(f"{d:<8}" + "".join(f"{mean(lambda r: r['dataset'] == name and r['depth'] == d and r['width'] == w, 'ess'):9.4f}" for w in (16, 64, 256)))
    print("\n[well] ESS fraction vs output dimension")
    for D in DIMS:
        sel = lambda r: r["part"] == "B" and r["dim"] == D
        print(f"D={D:<3} ess={mean(sel, 'ess'):.5f}  s_y={mean(sel, 'sy'):.3f}")
