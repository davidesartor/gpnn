"""Where should the randomness live in the graph?

Fixes the objective and varies the structure: which layers are stochastic,
whether a node's noise scale depends on its parents, whether each path carries
its own output variance (het), and whether the last layer is integrated
analytically (collapse).
"""

import sys

from gpnn import methods, runner

OBJ, K = "es_rb", 8
STRUCT = {
    "all": dict(noise="all"),
    "first": dict(noise="first"),
    "last": dict(noise="last"),
    "input": dict(noise="input"),
    "all+cond": dict(noise="all", cond=True),
    "all+het": dict(noise="all", het=True),
    "all+collapse": dict(noise="all", collapse=True),
    "all+het+collapse": dict(noise="all", het=True, collapse=True),
    "all+cond+het": dict(noise="all", cond=True, het=True),
    "none+het": dict(noise="none", het=True),  # heteroscedastic Gaussian, CRPS-trained
}
TOYS = ["hetero", "skew", "bimodal", "heavy", "inverse", "friedman"]


TAGS = list(STRUCT) + ["sgn_analytic"]


def build(tag):
    if tag in methods.BASELINES:
        return methods.BASELINES[tag]
    return methods.make_sgn(OBJ, K=K, **STRUCT[tag])


if __name__ == "__main__":
    if "run" in sys.argv:
        runner.run("02_structure", build, TAGS, [("toy", t, {}) for t in TOYS], seeds=range(3))
    rows = runner.load("02_structure")
    print(runner.table(rows, ["crps", "w1", "ccov_err90", "cov90", "width90", "cal_err", "nll", "info_sy"]))
