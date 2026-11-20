"""Stochastic graph networks against standard ways of getting predictive intervals.

Synthetic problems with a known p(y|x), at two training-set sizes, so that
methods are judged on the conditional distribution they recover.
"""

import sys

from gpnn import methods, runner

SGN = {
    "sgn": dict(objective="es_rb", K=8, noise="input"),  # noise appended to the input
    "sgn_temp": dict(objective="es_rb", K=8, noise="input", refit="temp"),  # + held-out noise rescaling
    "sgn_node": dict(objective="es_rb", K=8, noise="all"),  # noise at every hidden node
    "sgn_k2": dict(objective="es_rb", K=2, noise="input"),  # the minimum: two paths per example
    "sgn_pit": dict(objective="es_rb", K=8, noise="input", refit="pit"),  # + post-hoc PIT recalibration
}
TAGS = list(SGN) + ["gauss_homo", "conformal", "gauss_hetero", "mdn", "quantile", "ensemble", "gp"]
TOYS = ["hetero", "skew", "bimodal", "heavy", "inverse", "friedman"]
RUNS = [(n, act, seeds) for n in (2000, 200) for act, seeds in (("relu", 5), ("tanh", 3))]


def build(tag):
    return methods.make_sgn(**SGN[tag]) if tag in SGN else methods.BASELINES[tag]


if __name__ == "__main__":
    if "run" in sys.argv:
        for n, act, seeds in RUNS:
            tags = [t for t in TAGS if not (t == "gp" and act == "tanh")]  # the GP has no activation
            runner.run(f"03_baselines_n{n}_{act}", build, tags, [("toy", t, dict(n=n)) for t in TOYS], seeds=range(seeds), hp=dict(act=act))
    for n, act, _ in RUNS:
        if not (runner.RESULTS / f"03_baselines_n{n}_{act}.jsonl").exists():
            continue
        rows = runner.load(f"03_baselines_n{n}_{act}")
        print(f"\n================ n = {n}, {act}")
        print(runner.table(rows, ["crps", "w1", "ccov_err90", "cov90", "width90", "cal_err", "nll", "time"]))
