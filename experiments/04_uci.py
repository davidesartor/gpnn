"""UCI regression benchmarks: marginal calibration and sharpness on real data.

Random 80/10/10 splits; every network shares architecture, optimiser and
early stopping on the validation split. No ground-truth conditional exists
here, so only marginal metrics are available.
"""

import importlib
import sys

from gpnn import runner

b3 = importlib.import_module("experiments.03_baselines")
TAGS = b3.TAGS
SETS = ["yacht", "boston", "energy", "concrete", "wine", "kin8nm", "power", "naval", "protein"]


def build(tag):
    return b3.build(tag)


if __name__ == "__main__":
    if "run" in sys.argv:
        only = [a for a in sys.argv[2:]] or SETS
        runner.run("04_uci", build, TAGS, [("uci", s, {}) for s in only], seeds=range(5), hp=dict(steps=8000))
    rows = runner.load("04_uci")
    print(runner.table(rows, ["crps", "rmse", "nll", "cov90", "width90", "iscore90", "cov95", "cal_err", "time"], datasets=[s for s in SETS if any(r["dataset"] == s for r in rows)]))
