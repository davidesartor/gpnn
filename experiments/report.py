"""Markdown tables for the README, straight from results/*.jsonl.

python experiments/report.py RESULT metric [--rows a,b,c] [--cols x,y] [--fmt .3f] [--best min|max|target:0.9]
"""

import argparse

import numpy as np

from gpnn.runner import load

p = argparse.ArgumentParser()
p.add_argument("result")
p.add_argument("metric")
p.add_argument("--rows")
p.add_argument("--cols")
p.add_argument("--fmt", default=".3f")
p.add_argument("--best", default="min")
p.add_argument("--sd", action="store_true", help="append the standard deviation over seeds")
a = p.parse_args()

rows = load(a.result)
methods = a.rows.split(",") if a.rows else list(dict.fromkeys(r["method"] for r in rows))
sets = a.cols.split(",") if a.cols else list(dict.fromkeys(r["dataset"] for r in rows))
vals = {(m, d): [r[a.metric] for r in rows if r["method"] == m and r["dataset"] == d and a.metric in r] for m in methods for d in sets}
mean = {k: (np.mean(v) if v else np.nan) for k, v in vals.items()}


def score(v):
    if a.best == "min":
        return v
    if a.best == "max":
        return -v
    return abs(v - float(a.best.split(":")[1]))


print("| | " + " | ".join(sets) + " |")
print("|---|" + "---:|" * len(sets))
for m in methods:
    cells = []
    for d in sets:
        v = mean[m, d]
        if np.isnan(v):
            cells.append("–")
            continue
        best = min(score(mean[k, d]) for k in methods if not np.isnan(mean[k, d]))
        txt = format(v, a.fmt) + (f" ±{np.std(vals[m, d]):{a.fmt}}" if a.sd else "")
        cells.append(f"**{txt}**" if abs(score(v) - best) < 1e-12 else txt)
    print(f"| {m} | " + " | ".join(cells) + " |")
