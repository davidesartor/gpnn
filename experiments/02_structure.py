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
    "both": dict(noise="both"),
}
TOYS = ["hetero", "skew", "bimodal", "heavy", "inverse", "friedman"]


TAGS = list(STRUCT) + ["sgn_analytic"]
# tanh saturates, so with a linear readout the sampled means are bounded; unbounded
# activations let the network stretch Gaussian noise into heavy tails.
TAGS += [f"{s}@{a}" for a in ("relu", "silu") for s in ("all", "first", "input")]
# Is it the depth of processing after the noise that matters? (tag#depth)
TAGS += ["both", "both@silu", "all#3", "input#3", "both#3", "all@silu#3", "input@silu#3"]


def build(tag):
    if tag in methods.BASELINES:
        return methods.BASELINES[tag]
    rest, _, depth = tag.partition("#")
    struct, _, act = rest.partition("@")
    over = dict(act=act) if act else {}
    if depth:
        over["depth"] = int(depth)
    method = methods.make_sgn(OBJ, K=K, **STRUCT[struct])
    return lambda d, key, **hp: method(d, key, **{**hp, **over})


if __name__ == "__main__":
    if "run" in sys.argv:
        runner.run("02_structure", build, TAGS, [("toy", t, {}) for t in TOYS], seeds=range(3))
    rows = runner.load("02_structure")
    print(runner.table(rows, ["crps", "w1", "ccov_err90", "cov90", "width90", "cal_err", "nll", "info_sy"]))
