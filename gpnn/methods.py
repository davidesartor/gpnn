"""Every method behind one interface, so experiments compare like with like.

A method is a function  f(d: Split, key, **hp) -> dict(S=(N_test, M) samples,
logp=(N_test,) or None, info=dict).  Everything works in standardised units.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from . import sgn
from .train import fit

M = 1000  # predictive samples per test point
HP = dict(width=64, depth=2, act="tanh", steps=6000, batch=128, lr=2e-3, wd=0.0)


def _hp(hp):
    out = dict(HP)
    out.update(hp)
    return out


def _train_kw(hp):
    return {k: hp[k] for k in ("steps", "batch", "lr", "wd")}


# ---------------------------------------------------------------- SGN


def _with_noise(params, delta):
    """Shift the log noise scales: of the nodes, of the input noise, of the output."""
    layers = [dict(l, log_s=l["log_s"] + dl) for l, dl in zip(params["layers"], delta["layers"])]
    return dict(params, layers=layers, log_eps=params.get("log_eps", 0.0) + delta["eps"], log_sy=params["log_sy"] + delta["out"])


def refit_noise(cfg, params, key, xv, yv, objective, mode="temp", steps=600, K=16):
    """Re-estimate the noise scales on held-out data, weights frozen.

    Training data cannot see the error a fitted network makes on new inputs;
    held-out data can.
      mode="temp"    one multiplier for the output noise and the node noise
      mode="temp2"   one multiplier for the noise inside the network (input and
                     nodes) and a separate one for the output noise
      mode="scales"  one multiplier per node
    """
    zero = jnp.zeros(())
    if mode == "temp":
        init, expand = dict(t=zero), lambda q: dict(layers=[q["t"]] * cfg.depth, eps=0.0, out=q["t"])
    elif mode == "temp2":
        init = dict(inner=zero, out=zero)
        expand = lambda q: dict(layers=[q["inner"]] * cfg.depth, eps=q["inner"], out=q["out"])
    else:
        init = dict(layers=[jnp.zeros(cfg.width)] * cfg.depth, eps=zero, out=zero)
        expand = lambda q: q
    inner = sgn.make_loss(cfg, objective, K)
    loss = lambda q, k, x, y: inner(_with_noise(params, expand(q)), k, x, y)
    q, _ = fit(key, init, loss, xv, yv, steps=steps, batch=256, lr=2e-2, eval_every=steps)
    return _with_noise(params, expand(q))


def pit_recalibrate(S, S_val, y_val):
    """Push predictive samples through the held-out PIT distribution.

    If G is the CDF of the validation PITs F(y), then G o F is calibrated on the
    validation split by construction; its quantile function is Q(G^-1(u)).
    It can only re-weight the range the samples already span, so it cannot
    repair a forecaster whose samples never reach the observed values.
    """
    pit = (np.asarray(S_val) < np.asarray(y_val)[:, None]).mean(1)
    u = (np.arange(S.shape[1]) + 0.5) / S.shape[1]
    return np.quantile(S, np.quantile(pit, u), axis=1).T


def make_sgn(objective="es_rb", K=8, K_val=32, refit=None, n_ens=1, **cfg_kw):
    """refit: None | "temp" | "scales" (see refit_noise) | "pit" (see pit_recalibrate).
    n_ens > 1 pools the samples of independently trained networks."""

    def one(d, key, hp):
        cfg = sgn.SGN(width=hp["width"], depth=hp["depth"], act=hp["act"], **cfg_kw)
        k0, k1, k2, k3, k4 = jax.random.split(key, 5)
        params = sgn.init(cfg, k0, d.xtr.shape[1])
        t = time.time()
        params, hist = fit(
            k1,
            params,
            sgn.make_loss(cfg, objective, K),
            d.xtr,
            d.ytr,
            val=(d.xva, d.yva),
            val_loss_fn=sgn.make_loss(cfg, objective, max(K, K_val)),
            **_train_kw(hp),
        )
        if refit in ("temp", "temp2", "scales"):
            params = refit_noise(cfg, params, k4, d.xva, d.yva, objective, refit)
        info = dict(train_s=time.time() - t, sy=float(jnp.exp(params["log_sy"])), hist=hist)
        if not cfg.cond:
            info["node_s"] = [float(jnp.exp(l["log_s"]).mean()) for l in params["layers"]]
        S = sgn.sample(cfg, params, k2, jnp.asarray(d.xte), M // n_ens)
        logp, ess = sgn.log_prob(cfg, params, k3, jnp.asarray(d.xte), jnp.asarray(d.yte), 2000, return_ess=True)
        info["ess"] = float(jnp.median(ess))
        S, logp = np.asarray(S), np.asarray(logp)
        if refit == "pit":
            S_val = sgn.sample(cfg, params, k4, jnp.asarray(d.xva), M)
            S, logp = pit_recalibrate(S, S_val, d.yva), None  # the density is no longer the model's
        return dict(S=S, logp=logp, info=info, params=params, cfg=cfg)

    def method(d, key, **hp):
        hp = _hp(hp)
        if n_ens == 1:
            return one(d, key, hp)
        rs = [one(d, k, hp) for k in jax.random.split(key, n_ens)]
        logp = np.asarray(logsumexp(jnp.asarray(np.stack([r["logp"] for r in rs])), 0) - np.log(n_ens))
        info = dict(train_s=sum(r["info"]["train_s"] for r in rs))
        return dict(S=np.concatenate([r["S"] for r in rs], 1), logp=logp, info=info)

    return method


def sgn_analytic(d, key, **hp):
    """SGN with the node noise propagated analytically; Gaussian output, one pass."""
    hp = _hp(hp)
    cfg = sgn.SGN(width=hp["width"], depth=hp["depth"], act=hp["act"])
    k0, k1, k2 = jax.random.split(key, 3)
    params = sgn.init(cfg, k0, d.xtr.shape[1])
    loss = lambda p, k, x, y: -norm.logpdf(y, *sgn.forward_analytic(cfg, p, x)).mean()
    t = time.time()
    params, hist = fit(k1, params, loss, d.xtr, d.ytr, val=(d.xva, d.yva), **_train_kw(hp))
    mu, sd = sgn.forward_analytic(cfg, params, jnp.asarray(d.xte))
    S = mu[:, None] + sd[:, None] * jax.random.normal(k2, (len(d.xte), M))
    # How far is the analytic Gaussian from what the same weights do when sampled?
    Smc = np.asarray(sgn.sample(cfg, params, k2, jnp.asarray(d.xte), M))
    info = dict(train_s=time.time() - t, mc_sd_ratio=float(np.mean(Smc.std(1) / np.asarray(sd))))
    return dict(S=np.asarray(S), logp=np.asarray(norm.logpdf(jnp.asarray(d.yte), mu, sd)), info=info)


# ---------------------------------------------------------------- MLP baselines


def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        dict(W=jax.random.normal(k, (m, n)) / jnp.sqrt(m), b=jnp.zeros(n))
        for k, m, n in zip(keys, sizes[:-1], sizes[1:])
    ]


def mlp(params, x, act):
    f = sgn.ACTS[act]
    for layer in params[:-1]:
        x = f(x @ layer["W"] + layer["b"])
    return x @ params[-1]["W"] + params[-1]["b"]


def _fit_mlp(d, key, hp, n_out, loss_of_out):
    params = init_mlp(key, [d.xtr.shape[1]] + [hp["width"]] * hp["depth"] + [n_out])
    loss = lambda p, k, x, y: loss_of_out(mlp(p, x, hp["act"]), y).mean()
    t = time.time()
    params, hist = fit(key, params, loss, d.xtr, d.ytr, val=(d.xva, d.yva), **_train_kw(hp))
    return params, dict(train_s=time.time() - t, hist=hist)


def _hetero_nll(out, y):
    return -norm.logpdf(y, out[:, 0], jnp.exp(out[:, 1]))


def gauss_hetero(d, key, **hp):
    """Mean and log-std heads trained by Gaussian NLL."""
    hp = _hp(hp)
    k0, k1 = jax.random.split(key)
    params, info = _fit_mlp(d, k0, hp, 2, _hetero_nll)
    out = mlp(params, jnp.asarray(d.xte), hp["act"])
    mu, sd = out[:, :1], jnp.exp(out[:, 1:])
    S = mu + sd * jax.random.normal(k1, (len(d.xte), M))
    return dict(S=np.asarray(S), logp=np.asarray(-_hetero_nll(out, jnp.asarray(d.yte))), info=info)


def ensemble(d, key, n=5, **hp):
    """Deep ensemble of heteroscedastic Gaussians; predictive is the mixture."""
    hp = _hp(hp)
    keys = jax.random.split(key, n + 1)
    S, lp, t = [], [], 0.0
    for k in keys[:n]:
        r = gauss_hetero(d, k, **hp)
        S.append(r["S"][:, : M // n])
        lp.append(r["logp"])
        t += r["info"]["train_s"]
    logp = np.asarray(logsumexp(jnp.asarray(np.stack(lp)), 0) - np.log(n))
    return dict(S=np.concatenate(S, 1), logp=logp, info=dict(train_s=t))


def make_mdn(C=5):
    def nll(out, y):
        logw = jax.nn.log_softmax(out[:, :C])
        lp = norm.logpdf(y[:, None], out[:, C : 2 * C], jnp.exp(out[:, 2 * C :]))
        return -logsumexp(logw + lp, 1)

    def method(d, key, **hp):
        hp = _hp(hp)
        k0, k1, k2 = jax.random.split(key, 3)
        params, info = _fit_mlp(d, k0, hp, 3 * C, nll)
        out = mlp(params, jnp.asarray(d.xte), hp["act"])
        comp = jax.random.categorical(k1, out[:, None, :C], shape=(len(d.xte), M))
        mu = jnp.take_along_axis(out[:, C : 2 * C], comp, 1)
        sd = jnp.take_along_axis(jnp.exp(out[:, 2 * C :]), comp, 1)
        S = mu + sd * jax.random.normal(k2, mu.shape)
        return dict(S=np.asarray(S), logp=np.asarray(-nll(out, jnp.asarray(d.yte))), info=info)

    return method


def quantile_reg(d, key, Q=100, **hp):
    """One output per quantile level, trained with the pinball loss."""
    hp = _hp(hp)
    taus = (jnp.arange(Q) + 0.5) / Q

    def pinball(out, y):
        e = y[:, None] - out
        return jnp.maximum(taus * e, (taus - 1) * e).mean(1)

    k0, k1 = jax.random.split(key)
    params, info = _fit_mlp(d, k0, hp, Q, pinball)
    q = np.sort(np.asarray(mlp(params, jnp.asarray(d.xte), hp["act"])), 1)  # fix crossings
    u = np.asarray(jax.random.uniform(k1, (len(d.xte), M)))
    S = np.stack([np.interp(ui, np.asarray(taus), qi) for ui, qi in zip(u, q)])
    return dict(S=S, logp=None, info=info)


def gauss_homo(d, key, **hp):
    """Deterministic net + one global noise level (plain regression)."""
    return make_sgn("iw", K=1, K_val=1, noise="none")(d, key, **hp)


def conformal(d, key, **hp):
    """Deterministic net; predictive = point prediction + held-out residuals.

    Its central intervals are split-conformal style: marginally valid, constant width.
    """
    r = gauss_homo(d, key, **hp)
    cfg, params = r["cfg"], r["params"]
    k0, k1 = jax.random.split(key)
    res = np.asarray(d.yva) - np.asarray(sgn.forward(cfg, params, k0, jnp.asarray(d.xva))[0])
    mu = np.asarray(sgn.forward(cfg, params, k0, jnp.asarray(d.xte))[0])
    idx = np.asarray(jax.random.randint(k1, (len(d.xte), M), 0, len(res)))
    return dict(S=mu[:, None] + res[idx], logp=None, info=r["info"])


def gp(d, key, max_n=1000, **hp):
    """Exact GP, ARD RBF + white noise, type-II maximum likelihood."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    x = np.concatenate([d.xtr, d.xva]).astype(np.float64)
    y = np.concatenate([d.ytr, d.yva]).astype(np.float64)
    if len(x) > max_n:
        idx = np.random.default_rng(0).choice(len(x), max_n, replace=False)
        x, y = x[idx], y[idx]
    kern = ConstantKernel(1.0) * RBF(np.ones(x.shape[1]), (1e-2, 1e3)) + WhiteKernel(0.1, (1e-6, 1e1))
    t = time.time()
    model = GaussianProcessRegressor(kern, n_restarts_optimizer=0, random_state=0).fit(x, y)
    mu, sd = model.predict(d.xte.astype(np.float64), return_std=True)
    eps = np.asarray(jax.random.normal(key, (len(mu), M)))
    logp = np.asarray(norm.logpdf(d.yte, mu, sd))
    return dict(S=mu[:, None] + sd[:, None] * eps, logp=logp, info=dict(train_s=time.time() - t))


# MC dropout: stochastic hidden units, trained with one sample per example on the
# Gaussian likelihood; the predictive distribution is read off test-time samples.
mc_dropout = make_sgn("iw", K=1, K_val=1, noise="dropout")

BASELINES = {
    "mc_dropout": mc_dropout,
    "gauss_homo": gauss_homo,
    "conformal": conformal,
    "gauss_hetero": gauss_hetero,
    "mdn": make_mdn(5),
    "quantile": quantile_reg,
    "ensemble": ensemble,
    "gp": gp,
    "sgn_analytic": sgn_analytic,
}
