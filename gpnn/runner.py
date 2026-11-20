"""Run (method, dataset, seed) grids in parallel and append results as JSON lines."""

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _pin(cores):
    # One core per worker, set before XLA starts so its thread pool has size 1.
    os.sched_setaffinity(0, {cores.get()})


def _one(job):
    import jax
    import numpy as np

    from . import data, metrics

    build, tag, dataset, seed, hp = job
    kind, name, kw = dataset
    d = data.toy(name, seed=seed, **kw) if kind == "toy" else data.uci(name, seed=seed, **kw)
    method = build(tag)
    t = time.time()
    r = method(d, jax.random.key(seed), **hp)
    row = dict(method=tag, dataset=name, seed=seed, n=len(d.xtr), time=time.time() - t)
    row.update(metrics.summarize(r["S"], d.yte, d.ys))
    if r["logp"] is not None:
        row["nll"] = float(-np.mean(r["logp"]) + np.log(d.ys))
    if d.truth is not None:
        row.update(metrics.against_truth(r["S"], d.truth))
    info = {k: v for k, v in r["info"].items() if k != "hist"}
    row.update({f"info_{k}": v for k, v in info.items()})
    return row


def run(out, build, tags, datasets, seeds, hp=None, workers=4):
    """build(tag) -> method must be a picklable top-level function."""
    path = RESULTS / f"{out}.jsonl"
    path.parent.mkdir(exist_ok=True)
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["method"], r["dataset"], r["seed"]))
    jobs = [
        (build, tag, ds, seed, hp or {})
        for ds in datasets
        for tag in tags
        for seed in seeds
        if (tag, ds[1], seed) not in done
    ]
    print(f"{out}: {len(jobs)} jobs ({len(done)} cached)", flush=True)
    t0 = time.time()
    cores = mp.Queue()
    for c in sorted(os.sched_getaffinity(0))[:workers]:
        cores.put(c)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), initializer=_pin, initargs=(cores,)) as ex, path.open("a") as f:
        futs = {ex.submit(_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs)):
            j = futs[fut]
            try:
                row = fut.result()
            except Exception as e:  # keep the sweep alive, report the failure
                print(f"FAILED {j[1]} {j[2][1]} seed={j[3]}: {e!r}", file=sys.stderr, flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)


def load(out):
    return [json.loads(l) for l in (RESULTS / f"{out}.jsonl").read_text().splitlines()]


def table(rows, metrics, by="method", datasets=None, fmt="{:.3f}"):
    """Mean over seeds of each metric, one block per dataset."""
    import numpy as np

    names = datasets or sorted({r["dataset"] for r in rows})
    lines = []
    for ds in names:
        sub = [r for r in rows if r["dataset"] == ds]
        keys = list(dict.fromkeys(r[by] for r in sub))
        lines.append(f"\n[{ds}]  n_train={sub[0]['n']}  seeds={len({r['seed'] for r in sub})}")
        lines.append(f"{'':28}" + "".join(f"{m:>12}" for m in metrics))
        for k in keys:
            vals = []
            for m in metrics:
                v = [r[m] for r in sub if r[by] == k and m in r]
                vals.append(fmt.format(np.mean(v)) if v else "-")
            lines.append(f"{k:28}" + "".join(f"{v:>12}" for v in vals))
    return "\n".join(lines)
