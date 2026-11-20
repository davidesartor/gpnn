"""End-to-end checks of the front end on problems small enough to train in seconds."""

import numpy as np

from gpnn.api import ProbNet


def _hetero(rng, n):
    x = rng.uniform(-1, 1, (n, 1))
    return x, np.sin(3 * x[:, 0]) + (0.05 + 0.4 * (x[:, 0] > 0)) * rng.standard_normal(n)


def test_scalar_intervals_cover_and_adapt_to_the_noise_level():
    rng = np.random.default_rng(0)
    x, y = _hetero(rng, 1500)
    xt, yt = _hetero(rng, 2000)
    net = ProbNet(steps=1500).fit(x, y)
    lo, hi = net.interval(xt, 0.9)
    assert abs(((yt >= lo) & (yt <= hi)).mean() - 0.9) < 0.05
    width = hi - lo
    assert width[xt[:, 0] > 0.2].mean() > 3 * width[xt[:, 0] < -0.2].mean()  # noisy half is wider
    q = net.quantile(xt[:5], [0.1, 0.5, 0.9])
    assert q.shape == (5, 3) and np.all(np.diff(q, axis=1) > 0)
    assert np.isfinite(net.log_prob(xt, yt)).all()


def test_regions_exclude_the_empty_space_between_two_modes():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.5, 1, (2000, 1))
    y = rng.choice([-1.0, 1.0], 2000) * x[:, 0] + 0.05 * rng.standard_normal(2000)
    net = ProbNet(steps=3000).fit(x, y)
    parts = net.regions(np.array([[0.9]]), 0.9)[0]
    assert any(hi < 0 for _, hi in parts) and any(lo > 0 for lo, _ in parts)
    assert not any(lo <= 0 <= hi for lo, hi in parts)
    lo, hi = net.interval(np.array([[0.9]]), 0.9)
    assert sum(b - a for a, b in parts) < 0.7 * (hi - lo)[0]  # shorter than the central interval


def test_vector_targets_give_joint_samples_and_functional_intervals():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, (1500, 2))
    shared = rng.standard_normal(1500)  # one noise source moves both outputs together
    y = np.stack([x[:, 0] + shared, x[:, 1] + shared], 1)
    net = ProbNet(steps=1500).fit(x, y)
    S = net.sample(x[:200], 300)
    assert S.shape == (200, 300, 2)
    corr = np.mean([np.corrcoef(s.T)[0, 1] for s in S])
    assert corr > 0.8  # the dependence between outputs is learned, not just the marginals
    lo, hi = net.interval(x[:200], 0.9, fn=lambda v: v[..., 0] - v[..., 1])
    assert (hi - lo).mean() < 1.0  # the shared noise cancels in the difference
