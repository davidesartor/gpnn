"""How should K samples through the stochastic graph be scored, and how many are needed?

Sweeps objective x K on synthetic problems where the true p(y|x) is known, so
the result is judged against the truth (W1 distance, conditional coverage),
not just against held-out points.
"""

import sys

from gpnn import methods, runner

OBJ = ["iw", "es", "es_rb"]
KS = [1, 2, 4, 8, 32]
TAGS = [f"{o}:K={k}" for o in OBJ for k in KS if not (o != "iw" and k == 1)] + ["gauss_homo"]
TOYS = ["hetero", "skew", "bimodal", "heavy", "inverse", "friedman"]


def build(tag):
    if tag in methods.BASELINES:
        return methods.BASELINES[tag]
    obj, k = tag.split(":K=")
    return methods.make_sgn(obj, K=int(k))


if __name__ == "__main__":
    if "run" in sys.argv:
        runner.run("01_objectives", build, TAGS, [("toy", t, {}) for t in TOYS], seeds=range(3))
    rows = runner.load("01_objectives")
    print(runner.table(rows, ["crps", "w1", "ccov_err90", "cov90", "width90", "cal_err", "nll", "info_sy"]))
