"""Print one metric as a (method x dataset) grid: python experiments/summary.py RESULT metric [metric ...]"""

import sys

import numpy as np

from gpnn.runner import load

rows = load(sys.argv[1])
ds = list(dict.fromkeys(r["dataset"] for r in rows))
tags = list(dict.fromkeys(r["method"] for r in rows))
for metric in sys.argv[2:]:
    print(f"\n{metric}  (mean over {len({r['seed'] for r in rows})} seeds)\n" + f"{'':20}" + "".join(f"{d:>10}" for d in ds))
    for t in tags:
        cells = []
        for d in ds:
            v = [r[metric] for r in rows if r["method"] == t and r["dataset"] == d and metric in r]
            cells.append(f"{np.mean(v):10.3f}" if v else f"{'-':>10}")
        print(f"{t:20}" + "".join(cells))
