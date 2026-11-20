"""Stochastic graph network (SGN).

An MLP in which every hidden node is a conditional distribution instead of a
value. Weights are point estimates; all randomness lives in the nodes and is
drawn fresh for every input, so the model describes p(y | x) and nothing else.

    z_l = W_l h_{l-1} + b_l + s_l * eps_l,     h_l = act(z_l)
    y   = w . h_L + b + s_y * eps_y

A forward pass is one ancestral sample from p(y | x). K forward passes give K
samples, and the losses below score those samples against the observed y.

Two optional refinements keep the same picture but integrate part of the graph
exactly instead of sampling it:

  het       the output noise s_y is itself a function of the sampled path
  collapse  the last hidden layer's noise is integrated out by quadrature, so
            each path yields a Gaussian over y rather than a point
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

ACTS = {"tanh": jnp.tanh, "relu": jax.nn.relu, "gelu": jax.nn.gelu, "silu": jax.nn.silu}


class SGN(NamedTuple):
    width: int = 64
    depth: int = 2
    act: str = "tanh"
    noise: str = "all"  # all | none | first | last | input
    cond: bool = False  # node noise scale depends on the parent layer
    input_noise: int = 8  # dims of noise appended to x when noise == "input"
    noise0: float = 0.5
    sy0: float = 0.3
    learn_sy: bool = True
    het: bool = False  # output noise depends on the sampled path
    collapse: bool = False  # integrate the last hidden layer's noise analytically
    n_quad: int = 12
    d_out: int = 1  # > 1: vector y; forward returns (B, d_out), see multi.py

    def noisy(self, l):
        return {
            "all": True,
            "none": False,
            "input": False,
            "first": l == 0,
            "last": l == self.depth - 1,
        }[self.noise]


def _inv_softplus(v):
    return jnp.log(jnp.expm1(v))


def init(cfg: SGN, key, d_in):
    d0 = d_in + (cfg.input_noise if cfg.noise == "input" else 0)
    sizes = [d0] + [cfg.width] * cfg.depth
    keys = jax.random.split(key, cfg.depth + 1)
    layers = []
    for k, m, n in zip(keys, sizes[:-1], sizes[1:]):
        layer = dict(W=jax.random.normal(k, (m, n)) / jnp.sqrt(m), b=jnp.zeros(n))
        if cfg.cond:
            layer["V"] = jnp.zeros((m, n))
            layer["c"] = jnp.full(n, _inv_softplus(cfg.noise0))
        else:
            layer["log_s"] = jnp.full(n, jnp.log(cfg.noise0))
        layers.append(layer)
    oshape = () if cfg.d_out == 1 else (cfg.d_out,)
    out = dict(w=jax.random.normal(keys[-1], (cfg.width, *oshape)) / jnp.sqrt(cfg.width), b=jnp.zeros(oshape))
    if cfg.het:
        out["w_s"] = jnp.zeros((sizes[-2] if cfg.collapse else cfg.width, *oshape))
    return dict(layers=layers, out=out, log_sy=jnp.log(cfg.sy0) * jnp.ones(oshape))


def noise_scale(cfg: SGN, layer, h):
    if cfg.cond:
        return jax.nn.softplus(h @ layer["V"] + layer["c"])
    return jnp.exp(layer["log_s"])


def _log_sy(cfg: SGN, params):
    return params["log_sy"] if cfg.learn_sy else jax.lax.stop_gradient(params["log_sy"])


def _collapsed_layer(cfg: SGN, params, layer, h):
    """Mean and variance of w . act(z) + b over the last layer's node noise.

    Given the parent state the nodes of one layer are independent, so these two
    moments are exact (up to quadrature); treating the sum as Gaussian is a CLT
    approximation over `width` independent terms.
    """
    t, wq = np.polynomial.hermite.hermgauss(cfg.n_quad)
    t, wq = jnp.asarray(np.sqrt(2) * t, h.dtype), jnp.asarray(wq / np.sqrt(np.pi), h.dtype)
    z = h @ layer["W"] + layer["b"]
    a = ACTS[cfg.act](z[..., None] + noise_scale(cfg, layer, h)[..., None] * t)
    m1, m2 = a @ wq, (a**2) @ wq
    w = params["out"]["w"]
    return m1 @ w + params["out"]["b"], jnp.maximum(m2 - m1**2, 0.0) @ w**2


def forward(cfg: SGN, params, key, x):
    """One ancestral sample per row of x.

    Returns (mu, sd), each (B,): given the sampled path, y ~ N(mu, sd^2).
    """
    act = ACTS[cfg.act]
    keys = jax.random.split(key, cfg.depth + 1)
    h = x
    if cfg.noise == "input":
        h = jnp.concatenate([h, jax.random.normal(keys[-1], (x.shape[0], cfg.input_noise))], -1)
    n_sampled = cfg.depth - 1 if cfg.collapse else cfg.depth
    for l, layer in enumerate(params["layers"][:n_sampled]):
        z = h @ layer["W"] + layer["b"]
        if cfg.noisy(l):
            z = z + noise_scale(cfg, layer, h) * jax.random.normal(keys[l], z.shape)
        h = act(z)
    log_sy = _log_sy(cfg, params) + (h @ params["out"]["w_s"] if cfg.het else 0.0)
    if cfg.collapse:
        mu, var = _collapsed_layer(cfg, params, params["layers"][-1], h)
        return mu, jnp.sqrt(var + jnp.exp(2 * log_sy))
    mu = h @ params["out"]["w"] + params["out"]["b"]
    return mu, jnp.broadcast_to(jnp.exp(log_sy), mu.shape)


def _gauss_act_moments(cfg: SGN, m, v):
    """Mean and variance of act(z) for z ~ N(m, v), by Gauss-Hermite quadrature."""
    t, wq = np.polynomial.hermite.hermgauss(cfg.n_quad)
    t, wq = jnp.asarray(np.sqrt(2) * t, m.dtype), jnp.asarray(wq / np.sqrt(np.pi), m.dtype)
    a = ACTS[cfg.act](m[..., None] + jnp.sqrt(v)[..., None] * t)
    m1, m2 = a @ wq, (a**2) @ wq
    return m1, jnp.maximum(m2 - m1**2, 0.0)


def forward_analytic(cfg: SGN, params, x):
    """No sampling: push a mean and a diagonal variance through every layer.

    Exact for one hidden layer; for deeper nets it ignores the correlations
    between nodes that share noisy parents (assumed density filtering). The
    result is a single Gaussian over y, (mu, sd), each (B,).
    """
    m, v = x, jnp.zeros_like(x)
    for l, layer in enumerate(params["layers"]):
        zm = m @ layer["W"] + layer["b"]
        zv = v @ layer["W"] ** 2
        if cfg.noisy(l):
            zv = zv + noise_scale(cfg, layer, m) ** 2
        m, v = _gauss_act_moments(cfg, zm, zv)
    w = params["out"]["w"]
    return m @ w + params["out"]["b"], jnp.sqrt(v @ w**2 + jnp.exp(2 * _log_sy(cfg, params)))


def forward_k(cfg: SGN, params, key, x, K):
    """K independent ancestral samples. Returns (mu, sd), each (K, B)."""
    return jax.vmap(lambda k: forward(cfg, params, k, x))(jax.random.split(key, K))


# ---------------------------------------------------------------- losses
# Each takes mu, sd (K, B) describing K sampled paths (y | path ~ N(mu, sd^2))
# and y (B,), and returns a per-example loss (B,).


def abs_moment(m, v):
    """E|N(m, v)|, the Rao-Blackwellised version of |a - b| under Gaussian noise."""
    sd = jnp.sqrt(v + 1e-12)
    r = m / sd
    return sd * (2 * norm.pdf(r) + r * (2 * norm.cdf(r) - 1))


def _offdiag_mean(d):
    """Mean of d[k, k'] over k != k'. d has shape (K, K, B)."""
    K = d.shape[0]
    return (d.sum((0, 1)) - jnp.trace(d)) / (K * (K - 1))


def loss_iw(mu, sd, y, key=None):
    """Importance-weighted likelihood bound: -log 1/K sum_k N(y | mu_k, s_y^2)."""
    logp = norm.logpdf(y[None], mu, sd)
    return -(logsumexp(logp, 0) - jnp.log(mu.shape[0]))


def loss_es(mu, sd, y, key):
    """Energy score (CRPS for scalar y) on plain samples. Needs K >= 2."""
    ys = mu + sd * jax.random.normal(key, mu.shape)
    fit = jnp.abs(ys - y[None]).mean(0)
    spread = _offdiag_mean(jnp.abs(ys[:, None] - ys[None, :]))
    return fit - 0.5 * spread


def loss_es_rb(mu, sd, y, key=None):
    """Energy score with each path's Gaussian integrated out analytically."""
    v = sd**2
    fit = abs_moment(y[None] - mu, v).mean(0)
    spread = _offdiag_mean(abs_moment(mu[:, None] - mu[None, :], v[:, None] + v[None, :]))
    return fit - 0.5 * spread


def loss_qs(mu, sd, y, key=None):
    """Quadratic score  -2 p(y) + int p(u)^2 du.

    Strictly proper like the log score, but linear in the density, so K >= 2
    paths give an unbiased estimate (the log of an average does not).
    """
    v = sd**2
    fit = norm.pdf(y[None], mu, sd).mean(0)
    overlap = _offdiag_mean(norm.pdf(mu[:, None], mu[None, :], jnp.sqrt(v[:, None] + v[None, :])))
    return -2 * fit + overlap


LOSSES = {"iw": loss_iw, "es": loss_es, "es_rb": loss_es_rb, "qs": loss_qs}


def make_loss(cfg: SGN, objective: str, K: int):
    """`objective` is a loss name, or several joined by '+' (summed)."""
    fns = [LOSSES[o] for o in objective.split("+")]
    fn = lambda mu, sd, y, key: sum(f(mu, sd, y, key) for f in fns)

    def loss(params, key, x, y):
        k1, k2 = jax.random.split(key)
        mu, sd = forward_k(cfg, params, k1, x, K)
        return fn(mu, sd, y, k2).mean()

    return loss


# ---------------------------------------------------------------- prediction


def _chunks(n, size):
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def sample(cfg: SGN, params, key, x, M=1000, chunk=512):
    """Draw M samples from p(y | x) for each row of x. Returns (N, M)."""

    @jax.jit
    def f(k, xb):
        k1, k2 = jax.random.split(k)
        mu, sd = forward_k(cfg, params, k1, xb, M)
        return (mu + sd * jax.random.normal(k2, mu.shape)).T

    spans = _chunks(x.shape[0], chunk)
    keys = jax.random.split(key, len(spans))
    return jnp.concatenate([f(k, x[a:b]) for k, (a, b) in zip(keys, spans)])


def log_prob(cfg: SGN, params, key, x, y, M=1000, chunk=512):
    """Monte Carlo estimate of log p(y | x), marginalising the nodes."""

    @jax.jit
    def f(k, xb, yb):
        return -loss_iw(*forward_k(cfg, params, k, xb, M), yb)

    spans = _chunks(x.shape[0], chunk)
    keys = jax.random.split(key, len(spans))
    return jnp.concatenate([f(k, x[a:b], y[a:b]) for k, (a, b) in zip(keys, spans)])
