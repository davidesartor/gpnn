"""Checks that the estimators compute what their docstrings say."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, stats

from gpnn import metrics, sgn

KEY = jax.random.key(0)


def gaussian_crps(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def test_abs_moment_matches_monte_carlo():
    x = np.random.default_rng(0).normal(0.7, 1.3, 2_000_000)
    assert abs(float(sgn.abs_moment(0.7, 1.3**2)) - np.abs(x).mean()) < 3e-3
    assert abs(float(sgn.abs_moment(-2.0, 0.0)) - 2.0) < 1e-5


def test_sample_crps_matches_gaussian_closed_form():
    rng = np.random.default_rng(1)
    S = rng.normal(0.3, 0.8, (50, 40_000))
    y = rng.normal(0.3, 0.8, 50)
    np.testing.assert_allclose(metrics.crps(S, y), gaussian_crps(0.3, 0.8, y), atol=0.02)


def test_energy_losses_agree_and_equal_crps_of_the_mixture():
    """K=2 plain and Rao-Blackwellised energy scores are both unbiased for the CRPS."""
    k1, k2, k3 = jax.random.split(KEY, 3)
    n = 400_000
    mu = jax.random.normal(k1, (2, n)) * 0.5 + 1.0  # paths: mu ~ N(1, .5^2)
    sd = jnp.full((2, n), 0.4)
    y = jnp.full(n, 0.2)
    target = gaussian_crps(1.0, np.sqrt(0.5**2 + 0.4**2), 0.2)  # marginal is N(1, .5^2 + .4^2)
    assert abs(float(sgn.loss_es(mu, sd, y, k2).mean()) - target) < 5e-3
    assert abs(float(sgn.loss_es_rb(mu, sd, y).mean()) - target) < 5e-3


def test_quadratic_score_is_unbiased_with_two_paths():
    k1 = jax.random.split(KEY)[0]
    n = 400_000
    mu = jax.random.normal(k1, (2, n)) * 0.5 + 1.0
    sd = jnp.full((2, n), 0.4)
    p = stats.norm(1.0, np.sqrt(0.5**2 + 0.4**2))
    target = -2 * p.pdf(0.2) + integrate.quad(lambda u: p.pdf(u) ** 2, -10, 10)[0]
    assert abs(float(sgn.loss_qs(mu, sd, jnp.full(n, 0.2)).mean()) - target) < 5e-3


def test_importance_weighted_bound_is_biased_low_and_tightens_with_k():
    """E[log mean_k p] <= log p, with equality as K grows (Jensen)."""
    n = 200_000
    p = stats.norm(1.0, np.sqrt(0.5**2 + 0.1**2))
    vals = []
    for K in (1, 4, 64, 4096):
        mu = jax.random.normal(jax.random.fold_in(KEY, K), (K, max(n // K, 500))) * 0.5 + 1.0
        vals.append(-float(sgn.loss_iw(mu, jnp.full(mu.shape, 0.1), jnp.full(mu.shape[1], 0.2)).mean()))
    assert vals[0] < vals[1] < vals[2] < vals[3] <= p.logpdf(0.2) + 0.02
    assert abs(vals[3] - p.logpdf(0.2)) < 0.02


def _moments(cfg, params, x, n=40_000):
    S = np.asarray(sgn.sample(cfg, params, KEY, x, n, chunk=4))
    return S.mean(1), S.std(1)


def test_analytic_propagation_is_exact_for_one_hidden_layer():
    cfg = sgn.SGN(width=32, depth=1, noise0=0.7)
    params = sgn.init(cfg, KEY, 3)
    x = jax.random.normal(KEY, (4, 3))
    mu, sd = sgn.forward_analytic(cfg, params, x)
    m, s = _moments(cfg, params, x)
    np.testing.assert_allclose(mu, m, atol=0.02)
    np.testing.assert_allclose(sd, s, rtol=0.03)


def test_collapsed_last_layer_preserves_the_first_two_moments():
    base = sgn.SGN(width=32, depth=2, noise0=0.7)
    params = sgn.init(base, KEY, 3)
    x = jax.random.normal(KEY, (4, 3))
    m0, s0 = _moments(base, params, x)
    m1, s1 = _moments(base._replace(collapse=True), params, x)
    np.testing.assert_allclose(m0, m1, atol=0.02)
    np.testing.assert_allclose(s0, s1, rtol=0.03)


def test_coverage_and_calibration_of_a_perfect_forecaster():
    rng = np.random.default_rng(2)
    S, y = rng.standard_normal((4000, 500)), rng.standard_normal(4000)
    out = metrics.summarize(S, y)
    assert abs(out["cov90"] - 0.9) < 0.02 and out["cal_err"] < 0.02
    truth = rng.standard_normal((4000, 500))
    assert metrics.against_truth(S, truth)["w1"] < 0.12
