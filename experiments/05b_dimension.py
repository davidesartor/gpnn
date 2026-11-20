"""Does the advantage of joint sampling grow with the dimension of y?

The first D steps of a 48-step double-well path, D in {4, 12, 24, 48}.
"""

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gpnn import runner

TAGS = ["sgn", "mdn", "mv_gauss"]
DIMS = [4, 12, 24, 48]
OUT = runner.RESULTS / "05b_dimension.jsonl"


def one(job):
    os.sched_setaffinity(0, {job[-1]})
    import jax

    from gpnn import multi

    tag, D, seed, _ = job
    d = multi.Split("well48", seed=seed, dims=D)
    t = time.time()
    r = multi.METHODS[tag](d, jax.random.key(seed), act="relu")
    row = dict(method=tag, dataset=f"D={D}", seed=seed, n=len(d.xtr), time=time.time() - t)
    row.update(multi.evaluate(r["S"], d))
    return row


if __name__ == "__main__":
    if "run" in sys.argv:
        jobs = [(t, D, s, i % 4) for i, (t, D, s) in enumerate((t, D, s) for D in DIMS for t in TAGS for s in range(3))]
        with ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as ex, OUT.open("w") as f:
            for row in ex.map(one, jobs):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row["method"], row["dataset"], row["seed"], f"{row['time']:.0f}s", flush=True)
    rows = runner.load("05b_dimension")
    for metric in ("energy", "sum_w1", "sum_cov90", "range_w1", "range_cov90", "max_cov90"):
        print(f"\n{metric}\n{'':12}" + "".join(f"{f'D={D}':>9}" for D in DIMS))
        for t in TAGS:
            print(f"{t:12}" + "".join(f"{np.mean([r[metric] for r in rows if r['method'] == t and r['dataset'] == f'D={D}']):9.3f}" for D in DIMS))
