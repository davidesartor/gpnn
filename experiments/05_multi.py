"""Vector-valued targets: joint samples, and intervals for functionals of y.

ring:  y in R^2 on a noisy circle whose radius depends on x.
well:  12 future steps of a double-well SDE; x = (start, noise level).
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

import numpy as np

from gpnn import runner

TAGS = ["sgn_es", "sgn_iw", "mv_gauss", "mdn", "quantile_indep", "quantile_comono"]
OUT = runner.RESULTS / "05_multi.jsonl"


def one(job):
    os.sched_setaffinity(0, {job[3]})
    import jax

    from gpnn import multi

    tag, name, seed, _ = job
    d = multi.Split(name, seed=seed)
    t = time.time()
    r = multi.METHODS[tag](d, jax.random.key(seed))
    row = dict(method=tag, dataset=name, seed=seed, n=len(d.xtr), time=time.time() - t)
    row.update(multi.evaluate(r["S"], d))
    if seed == 0:
        np.save(runner.RESULTS / f"05_samples_{name}_{tag}.npy", r["S"][:6].astype(np.float32))
    return row


if __name__ == "__main__":
    if "run" in sys.argv:
        done = set()
        if OUT.exists():
            done = {(r["method"], r["dataset"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())}
        jobs = [(t, n, s) for n in ("ring", "well") for t in TAGS for s in range(3) if (t, n, s) not in done]
        jobs = [(*j, i % 4) for i, j in enumerate(jobs)]
        with ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as ex, OUT.open("a") as f:
            for row in ex.map(one, jobs):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row["method"], row["dataset"], row["seed"], f"{row['time']:.0f}s", flush=True)
    rows = runner.load("05_multi")
    print(runner.table(rows, ["energy", "marg_w1", "sum_w1", "sum_cov90", "max_w1", "max_cov90", "range_w1", "range_cov90", "time"]))
