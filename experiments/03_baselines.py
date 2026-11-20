"""Stochastic graph networks against standard ways of getting predictive intervals.

Synthetic problems with a known p(y|x), at two training-set sizes, so that
methods are judged on the conditional distribution they recover.
"""

import sys

from gpnn import methods, runner

SGN = {
    "sgn": dict(objective="es_rb", K=8),
    "sgn_het": dict(objective="es_rb", K=8, het=True, collapse=True),
}
TAGS = list(SGN) + ["gauss_homo", "conformal", "gauss_hetero", "mdn", "quantile", "ensemble", "gp"]
TOYS = ["hetero", "skew", "bimodal", "heavy", "inverse", "friedman"]


def build(tag):
    return methods.make_sgn(**SGN[tag]) if tag in SGN else methods.BASELINES[tag]


if __name__ == "__main__":
    if "run" in sys.argv:
        for n in (2000, 200):
            runner.run(f"03_baselines_n{n}", build, TAGS, [("toy", t, dict(n=n)) for t in TOYS], seeds=range(5))
    for n in (2000, 200):
        rows = runner.load(f"03_baselines_n{n}")
        print(f"\n================ n = {n}")
        print(runner.table(rows, ["crps", "w1", "ccov_err90", "cov90", "width90", "cal_err", "nll", "time"]))
