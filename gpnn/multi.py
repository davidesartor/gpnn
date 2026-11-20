"""Vector-valued y: joint predictive distributions, and intervals for functionals.

A stochastic graph network samples whole vectors y, so any functional of y
(a sum, a maximum, a norm) gets an interval for free: apply it to the samples.
Per-output heads (quantiles, independent Gaussians) cannot do that without an
extra assumption about dependence.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from . import metrics, sgn
from .methods import _hp, _train_kw, init_mlp, mlp
from .train import fit

M = 1000


def _norm(d):
    return jnp.sqrt(jnp.sum(d**2, -1) + 1e-12)


def loss_es(mu, sd, y, key):
    """Multivariate energy score on K sampled vectors. mu, sd: (K, B, D); y: (B, D)."""
    ys = mu + sd * jax.random.normal(key, mu.shape)
    K = ys.shape[0]
    fit_ = _norm(ys - y[None]).mean(0)
    pair = _norm(ys[:, None] - ys[None, :])
    spread = (pair.sum((0, 1)) - jnp.trace(pair)) / (K * (K - 1))
    return fit_ - 0.5 * spread


def loss_iw(mu, sd, y, key=None):
    logp = norm.logpdf(y[None], mu, sd).sum(-1)
    return -(logsumexp(logp, 0) - jnp.log(mu.shape[0]))


LOSSES = {"es": loss_es, "iw": loss_iw}

# ---------------------------------------------------------------- methods
# f(d, key, **hp) -> dict(S=(N, M, D) joint samples, info)


def make_sgn(objective="es", K=8, K_val=32, **cfg_kw):
    def method(d, key, **hp):
        hp = _hp(hp)
        D = d.ytr.shape[1]
        cfg = sgn.SGN(width=hp["width"], depth=hp["depth"], act=hp["act"], d_out=D, **cfg_kw)

        def make(k_):
            def loss(p, k, x, y):
                k1, k2 = jax.random.split(k)
                return LOSSES[objective](*sgn.forward_k(cfg, p, k1, x, k_), y, k2).mean()

            return loss

        k0, k1, k2 = jax.random.split(key, 3)
        params = sgn.init(cfg, k0, d.xtr.shape[1])
        t = time.time()
        params, _ = fit(k1, params, make(K), d.xtr, d.ytr, val=(d.xva, d.yva), val_loss_fn=make(max(K, K_val)), **_train_kw(hp))

        @jax.jit
        def draw(k, xb):
            ka, kb = jax.random.split(k)
            mu, sd = sgn.forward_k(cfg, params, ka, xb, M)
            return jnp.swapaxes(mu + sd * jax.random.normal(kb, mu.shape), 0, 1)

        xs = jnp.asarray(d.xte)
        S = np.concatenate([np.asarray(draw(jax.random.fold_in(k2, i), xs[i : i + 256])) for i in range(0, len(xs), 256)])
        _, ess = sgn.log_prob(cfg, params, k2, xs, jnp.asarray(d.yte), 2000, chunk=128, return_ess=True)
        info = dict(train_s=time.time() - t, ess=float(jnp.median(ess)), sy=float(jnp.exp(params["log_sy"]).mean()))
        return dict(S=S, info=info)

    return method


def _fit(d, key, hp, n_out, nll):
    params = init_mlp(key, [d.xtr.shape[1]] + [hp["width"]] * hp["depth"] + [n_out])
    loss = lambda p, k, x, y: nll(mlp(p, x, hp["act"]), y).mean()
    t = time.time()
    params, _ = fit(key, params, loss, d.xtr, d.ytr, val=(d.xva, d.yva), **_train_kw(hp))
    return mlp(params, jnp.asarray(d.xte), hp["act"]), dict(train_s=time.time() - t)


def mv_gauss(d, key, **hp):
    """Full-covariance Gaussian head (mean + Cholesky factor), trained by NLL."""
    hp = _hp(hp)
    D = d.ytr.shape[1]
    tril = np.tril_indices(D)
    diag = tril[0] == tril[1]

    def chol(raw):
        vals = jnp.where(diag, jnp.exp(raw), raw)
        return jnp.zeros(raw.shape[:-1] + (D, D)).at[..., tril[0], tril[1]].set(vals)

    def nll(out, y):
        L = chol(out[:, D:])
        z = jax.vmap(lambda Li, ri: jax.scipy.linalg.solve_triangular(Li, ri, lower=True))(L, y - out[:, :D])
        return 0.5 * (z**2).sum(-1) + jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)).sum(-1)

    k0, k1 = jax.random.split(key)
    out, info = _fit(d, k0, hp, D + len(tril[0]), nll)
    eps = jax.random.normal(k1, (len(d.xte), M, D))
    S = out[:, None, :D] + jnp.einsum("nij,nmj->nmi", chol(out[:, D:]), eps)
    return dict(S=np.asarray(S), info=info)


def make_mdn(C=10):
    """Mixture of C diagonal Gaussians over the whole vector y."""

    def method(d, key, **hp):
        hp = _hp(hp)
        D = d.ytr.shape[1]

        def split(out):
            return out[:, :C], out[:, C : C + C * D].reshape(-1, C, D), out[:, C + C * D :].reshape(-1, C, D)

        def nll(out, y):
            lw, mu, ls = split(out)
            lp = norm.logpdf(y[:, None], mu, jnp.exp(ls)).sum(-1)
            return -logsumexp(jax.nn.log_softmax(lw) + lp, 1)

        k0, k1, k2 = jax.random.split(key, 3)
        out, info = _fit(d, k0, hp, C + 2 * C * D, nll)
        lw, mu, ls = split(out)
        comp = jax.random.categorical(k1, lw[:, None, :], shape=(len(d.xte), M))
        mu = jnp.take_along_axis(mu, comp[..., None], 1)
        sd = jnp.exp(jnp.take_along_axis(ls, comp[..., None], 1))
        return dict(S=np.asarray(mu + sd * jax.random.normal(k2, mu.shape)), info=info)

    return method


def make_quantile(coupling="indep", Q=50):
    """Per-output quantile regression. Marginals only, so a joint sample needs an
    assumed coupling: independent ranks, or one shared rank (comonotone)."""

    def method(d, key, **hp):
        hp = _hp(hp)
        D = d.ytr.shape[1]
        taus = (jnp.arange(Q) + 0.5) / Q

        def pinball(out, y):
            e = y[:, :, None] - out.reshape(-1, D, Q)
            return jnp.maximum(taus * e, (taus - 1) * e).mean((1, 2))

        k0, k1 = jax.random.split(key)
        out, info = _fit(d, k0, hp, D * Q, pinball)
        q = np.sort(np.asarray(out).reshape(-1, D, Q), -1)
        shape = (len(q), M, D) if coupling == "indep" else (len(q), M, 1)
        u = np.broadcast_to(np.asarray(jax.random.uniform(k1, shape)), (len(q), M, D))
        t = np.asarray(taus)
        S = np.stack([np.stack([np.interp(u[n, :, j], t, q[n, j]) for j in range(D)], -1) for n in range(len(q))])
        return dict(S=S, info=info)

    return method


METHODS = {
    "sgn_es": make_sgn("es", K=8),
    "sgn_iw": make_sgn("iw", K=32),
    "mv_gauss": mv_gauss,
    "mdn": make_mdn(10),
    "quantile_indep": make_quantile("indep"),
    "quantile_comono": make_quantile("comono"),
}

# ---------------------------------------------------------------- problems


def _ring_x(rng, n):
    return rng.uniform(-1, 1, (n, 1))


def _ring(rng, x, m):
    th = rng.uniform(0, 2 * np.pi, (len(x), m))
    r = 1.0 + 0.5 * x[:, :1]
    y = np.stack([r * np.cos(th), r * np.sin(th)], -1)
    return y + 0.05 * rng.standard_normal(y.shape)


def _well_x(rng, n):
    return np.stack([rng.uniform(-1.2, 1.2, n), rng.uniform(0.2, 0.8, n)], 1)


def _well(rng, x, m, H=12, dt=0.25, sub=5):
    """Double-well SDE dy = y(1 - y^2) dt + sigma dW from y0; x = (y0, sigma).

    Returns the next H observations. Paths starting near 0 commit to one well,
    so the joint law over the horizon is strongly non-Gaussian.
    """
    y = np.repeat(x[:, :1], m, 1)
    sig, h = x[:, 1:2], dt / sub
    out = []
    for _ in range(H):
        for _ in range(sub):
            y = y + y * (1 - y**2) * h + sig * np.sqrt(h) * rng.standard_normal(y.shape)
        out.append(y)
    return np.stack(out, -1)


TOYS = {"ring": (_ring_x, _ring), "well": (_well_x, _well)}


class Split:
    def __init__(self, name, n=4000, seed=0, n_test=300, m_truth=1000, dims=None):
        sx, sy_ = TOYS[name]
        sy = sy_ if dims is None else (lambda rng, x, m: sy_(rng, x, m)[..., :dims])
        rng = np.random.default_rng(seed)
        n_val = n // 5
        x = sx(rng, n + n_val)
        y = sy(rng, x, 1)[:, 0]
        xte = sx(np.random.default_rng(10_000), n_test)
        rte = np.random.default_rng(10_001)
        truth, yte = sy(rte, xte, m_truth), sy(rte, xte, 1)[:, 0]
        xm, xs = x[:n].mean(0), x[:n].std(0)
        self.ym, self.ys = y[:n].mean(), y[:n].std()  # one scale for all outputs
        f32 = lambda a: a.astype(np.float32)
        nx, ny = lambda a: f32((a - xm) / xs), lambda a: f32((a - self.ym) / self.ys)
        self.xtr, self.ytr, self.xva, self.yva = nx(x[:n]), ny(y[:n]), nx(x[n:]), ny(y[n:])
        self.xte, self.yte, self.truth = nx(xte), ny(yte), ny(truth)


# ---------------------------------------------------------------- metrics

FUNCTIONALS = {
    "sum": lambda y: y.sum(-1),
    "max": lambda y: y.max(-1),
    "last": lambda y: y[..., -1],
    "range": lambda y: y.max(-1) - y.min(-1),
}


def energy_score(S, y, m=200):
    """Sample estimate of the multivariate energy score. S: (N, M, D); y: (N, D)."""
    S = np.asarray(S, np.float64)[:, :m]
    fit_ = np.linalg.norm(S - y[:, None], axis=-1).mean(1)
    pair = np.linalg.norm(S[:, :, None] - S[:, None, :], axis=-1)
    return float((fit_ - 0.5 * pair.sum((1, 2)) / (m * (m - 1))).mean())


def evaluate(S, d):
    """Joint score, per-output marginals, and intervals for functionals of y."""
    y, T = np.asarray(d.yte, np.float64), np.asarray(d.truth, np.float64)
    out = dict(energy=energy_score(S, y))
    D = y.shape[1]
    marg = [metrics.against_truth(S[:, :, j], T[:, :, j])["w1"] for j in range(D)]
    out["marg_w1"] = float(np.mean(marg))
    for name, f in FUNCTIONALS.items():
        fs, ft, fy = f(S), f(T), f(y)
        r = metrics.summarize(fs, fy) | metrics.against_truth(fs, ft)
        out.update({f"{name}_{k}": r[k] for k in ("crps", "w1", "cov90", "width90", "ccov_err90")})
    return out
