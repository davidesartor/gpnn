"""Central intervals vs highest-density regions when the prediction is multimodal."""

import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gpnn import runner

OUT = runner.RESULTS / "08_regions.jsonl"
TAGS = ["sgn", "mdn", "quantile"]


def one(job):
    tag, toy, seed, core = job
    os.sched_setaffinity(0, {core})
    import importlib

    import jax

    from gpnn import data, metrics

    d = data.toy(toy, seed=seed)
    S = importlib.import_module("experiments.03_baselines").build(tag)(d, jax.random.key(seed), act="relu")["S"]
    lo, hi = metrics.interval(S, 0.9)
    edges, mask = metrics.hdr(S, 0.9)
    T = d.truth
    return dict(
        method=tag,
        dataset=toy,
        seed=seed,
        n=len(d.xtr),
        central_cov=float(((T >= lo[:, None]) & (T <= hi[:, None])).mean()),
        central_len=float((hi - lo).mean() * d.ys),
        hdr_cov=float(metrics.hdr_contains(edges, mask, T).mean()),
        hdr_len=float(metrics.hdr_length(edges, mask).mean() * d.ys),
    )


if __name__ == "__main__":
    if "run" in sys.argv:
        jobs = [(t, toy, s, i % 4) for i, (t, toy, s) in enumerate((t, toy, s) for toy in ("bimodal", "inverse", "hetero") for t in TAGS for s in range(3))]
        with ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as ex, OUT.open("w") as f:
            for row in ex.map(one, jobs):
                f.write(json.dumps(row) + "\n")
                f.flush()
    print(runner.table(runner.load("08_regions"), ["central_cov", "central_len", "hdr_cov", "hdr_len"]))
