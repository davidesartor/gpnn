"""Regression problems: synthetic ones with a known p(y | x), and UCI benchmarks."""

from pathlib import Path

import numpy as np

UCI_DIR = Path(__file__).resolve().parent.parent / "data" / "uci"
UCI = {
    "boston": "bostonHousing",
    "concrete": "concrete",
    "energy": "energy",
    "kin8nm": "kin8nm",
    "naval": "naval-propulsion-plant",
    "power": "power-plant",
    "protein": "protein-tertiary-structure",
    "wine": "wine-quality-red",
    "yacht": "yacht",
}

# ------------------------------------------------------------ synthetic
# Each toy is (sample_x(rng, n) -> (n, d), sample_y(rng, x, m) -> (n, m)).
# sample_y draws m values from the true conditional at every row of x.


def _x1(lo=-1.0, hi=1.0):
    return lambda rng, n: rng.uniform(lo, hi, (n, 1))


def _hetero(rng, x, m):
    x = x[:, :1]
    sd = 0.05 + 0.4 / (1 + np.exp(-6 * x))
    return np.sin(3 * x) + sd * rng.standard_normal((len(x), m))


def _skew(rng, x, m):
    x = x[:, :1]
    scale = 0.1 + 0.4 * (x + 1) / 2
    return np.cos(2 * x) + scale * (rng.exponential(1.0, (len(x), m)) - 1)


def _bimodal(rng, x, m):
    """Pitchfork: one branch for x < 0 that splits into two for x > 0."""
    x = x[:, :1]
    sign = rng.choice([-1.0, 1.0], (len(x), m))
    return sign * np.sqrt(np.maximum(x, 0)) + 0.08 * rng.standard_normal((len(x), m))


def _heavy(rng, x, m):
    x = x[:, :1]
    return x**2 + 0.15 * (1 + np.abs(x)) * rng.standard_t(3, (len(x), m))


def _inverse(rng, x, m):
    """Bishop's inverse problem: y ~ U(0,1), x = y + 0.3 sin(2 pi y) + 0.05 eps."""
    grid = np.linspace(0, 1, 2001)
    fwd = grid + 0.3 * np.sin(2 * np.pi * grid)
    logp = -0.5 * ((x[:, :1] - fwd[None]) / 0.05) ** 2
    cdf = np.cumsum(np.exp(logp - logp.max(1, keepdims=True)), 1)
    cdf /= cdf[:, -1:]
    u = rng.uniform(size=(len(x), m))
    return np.stack([np.interp(ui, ci, grid) for ui, ci in zip(u, cdf)])


def _inverse_x(rng, n):
    y = rng.uniform(size=(n, 1))
    return y + 0.3 * np.sin(2 * np.pi * y) + 0.05 * rng.standard_normal((n, 1))


def _friedman_x(rng, n):
    return rng.uniform(size=(n, 10))


def _friedman(rng, x, m):
    """Friedman #1 in 10-d (5 irrelevant inputs) with skewed, input-dependent noise."""
    f = 10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2 + 10 * x[:, 3] + 5 * x[:, 4]
    scale = 0.5 + 3.0 * x[:, 0]
    mix = rng.uniform(size=(len(x), m)) < 0.3 * x[:, 1:2]
    noise = np.where(mix, rng.exponential(2.0, (len(x), m)) + 1.0, rng.standard_normal((len(x), m)))
    return f[:, None] + scale[:, None] * noise


def _gap_x(rng, n):
    """Training inputs avoid (-0.2, 0.4)."""
    left = rng.uniform(size=n) < 0.57
    return np.where(left, rng.uniform(-1, -0.2, n), rng.uniform(0.4, 1, n))[:, None]


TOYS = {
    "hetero_gap": (_gap_x, _hetero),
    "hetero": (_x1(), _hetero),
    "skew": (_x1(), _skew),
    "bimodal": (_x1(), _bimodal),
    "heavy": (_x1(), _heavy),
    "inverse": (_inverse_x, _inverse),
    "friedman": (_friedman_x, _friedman),
}


class Split:
    """Standardised train/val/test arrays; `ys` is the y scale in original units."""

    def __init__(self, xtr, ytr, xva, yva, xte, yte, truth=None):
        self.xm, self.xs = xtr.mean(0), xtr.std(0) + 1e-8
        self.ym, self.ys = float(ytr.mean()), float(ytr.std())
        nx = lambda a: ((a - self.xm) / self.xs).astype(np.float32)
        ny = lambda a: ((a - self.ym) / self.ys).astype(np.float32)
        self.xtr, self.ytr = nx(xtr), ny(ytr)
        self.xva, self.yva = nx(xva), ny(yva)
        self.xte, self.yte = nx(xte), ny(yte)
        self.truth = None if truth is None else ny(truth)  # (n_test, m) true draws


def toy(name, n=2000, seed=0, n_test=500, m_truth=2000, x_test=None):
    """x_test (raw units) overrides the random test inputs, e.g. a grid for plots."""
    sx, sy = TOYS[name]
    rng = np.random.default_rng(seed)
    n_val = max(100, n // 5)
    x = sx(rng, n + n_val)
    y = sy(rng, x, 1)[:, 0]
    xte = sx(np.random.default_rng(10_000), n_test) if x_test is None else x_test  # fixed across seeds
    rte = np.random.default_rng(10_001)
    truth = sy(rte, xte, m_truth)
    yte = sy(rte, xte, 1)[:, 0]
    return Split(x[:n], y[:n], x[n:], y[n:], xte, yte, truth)


def uci(name, seed=0, test_frac=0.1, val_frac=0.1, max_n=None):
    d = UCI_DIR / UCI[name]
    data = np.loadtxt(d / "data.txt")
    fi = np.loadtxt(d / "index_features.txt").astype(int)
    ti = int(np.loadtxt(d / "index_target.txt"))
    x, y = data[:, fi], data[:, ti]
    perm = np.random.default_rng(seed).permutation(len(x))
    if max_n:
        perm = perm[:max_n]
    n_te, n_va = int(len(perm) * test_frac), int(len(perm) * val_frac)
    te, va, tr = perm[:n_te], perm[n_te : n_te + n_va], perm[n_te + n_va :]
    return Split(x[tr], y[tr], x[va], y[va], x[te], y[te])
