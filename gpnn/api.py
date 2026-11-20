"""A small front end: fit a sampling network, ask it for intervals.

    net = ProbNet().fit(x, y)
    lo, hi = net.interval(x_new, level=0.9)
    draws = net.sample(x_new, n=1000)

The network is a sampler for p(y | x). Everything it reports (intervals,
quantiles, means, intervals for functions of a vector y) is read off its
samples; there is no separate uncertainty head and no posterior over weights.

GPProbNet puts a Gaussian process underneath: the GP supplies a mean and a
scale that grows away from the data, and the sampler models the distribution
of what the GP leaves over.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import metrics, multi, sgn
from .methods import _fit_gp, gp_loo, gp_parts, refit_noise
from .train import fit


def _loss(cfg, K):
    if cfg.d_out == 1:
        return sgn.make_loss(cfg, "es_rb", K)

    def loss(p, k, x, y):
        k1, k2 = jax.random.split(k)
        return multi.loss_es(*sgn.forward_k(cfg, p, k1, x, K), y, k2).mean()

    return loss


def _draw(cfg, params, key, xs, n):
    """n samples per row of standardised inputs xs: (N, n) or (N, n, D)."""

    @jax.jit
    def draw(k, xb):
        k1, k2 = jax.random.split(k)
        mu, sd = sgn.forward_k(cfg, params, k1, xb, n)
        return jnp.swapaxes(mu + sd * jax.random.normal(k2, mu.shape), 0, 1)

    out = [draw(jax.random.fold_in(key, i), xs[i : i + 256]) for i in range(0, len(xs), 256)]
    return np.concatenate([np.asarray(o) for o in out])


class ProbNet:
    def __init__(
        self,
        width=64,
        depth=2,
        act="relu",
        noise="input",
        K=8,
        steps=6000,
        batch=128,
        lr=2e-3,
        val_frac=0.15,
        calibrate=True,
        seed=0,
    ):
        """K: sampled paths per training example (>= 2).
        calibrate: after training, freeze the weights and fit two multipliers on
        the validation split, one for the noise fed into the network and one for
        the output noise, so the spread reflects error on unseen data."""
        self.hp = dict(width=width, depth=depth, act=act, noise=noise)
        self.K, self.steps, self.batch, self.lr = K, steps, batch, lr
        self.val_frac, self.calibrate, self.seed = val_frac, calibrate, seed

    # ------------------------------------------------------------ fitting

    def fit(self, x, y, val=None):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        x = x.reshape(len(x), -1)
        if val is None:
            perm = np.random.default_rng(self.seed).permutation(len(x))
            n_val = max(1, int(len(x) * self.val_frac))
            val = (x[perm[:n_val]], y[perm[:n_val]])
            x, y = x[perm[n_val:]], y[perm[n_val:]]
        self.vector = y.ndim == 2
        self.xm, self.xs = x.mean(0), x.std(0) + 1e-8
        self.ym, self.ys = y.mean(), y.std() + 1e-8
        self._fit_std(jax.random.key(self.seed), self._x(x), self._y(y), self._x(val[0]), self._y(val[1]))
        return self

    def _train(self, key, xt, yt, xv, yv):
        """Train one sampler on standardised arrays. Returns (cfg, params)."""
        cfg = sgn.SGN(d_out=yt.shape[1] if yt.ndim == 2 else 1, **self.hp)
        k0, k1, k2 = jax.random.split(key, 3)
        params = sgn.init(cfg, k0, xt.shape[1])
        params, _ = fit(k1, params, _loss(cfg, self.K), xt, yt, val=(xv, yv), val_loss_fn=_loss(cfg, max(self.K, 32)), steps=self.steps, batch=self.batch, lr=self.lr)
        if self.calibrate and cfg.d_out == 1:
            params = refit_noise(cfg, params, k2, xv, yv, "es_rb", "temp2")
        return cfg, params

    def _fit_std(self, key, xt, yt, xv, yv):
        self.cfg, self.params = self._train(key, xt, yt, xv, yv)

    def _x(self, x):
        return jnp.asarray((np.asarray(x, np.float64).reshape(len(x), -1) - self.xm) / self.xs, jnp.float32)

    def _y(self, y):
        return jnp.asarray((np.asarray(y, np.float64) - self.ym) / self.ys, jnp.float32)

    # ------------------------------------------------------------ prediction

    def _sample_std(self, xs, n, key):
        return _draw(self.cfg, self.params, key, xs, n)

    def _log_prob_std(self, xs, ys, n, key):
        return np.asarray(sgn.log_prob(self.cfg, self.params, key, xs, ys, n))

    def sample(self, x, n=1000, seed=0):
        """n draws from p(y | x) per row: (N, n) for scalar y, (N, n, D) for vector y."""
        return self._sample_std(self._x(x), n, jax.random.key(seed)) * self.ys + self.ym

    def _scalar_samples(self, x, n, fn, seed):
        S = self.sample(x, n, seed)
        if fn is not None:
            S = fn(S)
        if S.ndim != 2:
            raise ValueError("vector y needs fn=..., a function of the last axis, e.g. lambda y: y.sum(-1)")
        return S

    def quantile(self, x, q, n=1000, fn=None, seed=0):
        """Quantiles of y (or of fn(y) for vector y). Returns (N, len(q))."""
        return np.quantile(self._scalar_samples(x, n, fn, seed), np.atleast_1d(q), axis=1).T

    def interval(self, x, level=0.9, n=1000, fn=None, seed=0):
        """Central interval containing y with probability `level`. Returns (lo, hi)."""
        a = (1 - level) / 2
        lo, hi = self.quantile(x, [a, 1 - a], n, fn, seed).T
        return lo, hi

    def regions(self, x, level=0.9, n=2000, fn=None, seed=0):
        """Highest-density regions: for each row, a list of (lo, hi) intervals that
        together hold `level` of the predictive mass. More than one interval means
        the prediction is multimodal."""
        edges, mask = metrics.hdr(self._scalar_samples(x, n, fn, seed), level)
        out = []
        for e, m in zip(edges, mask):
            step = np.diff(np.concatenate([[0], m.astype(int), [0]]))
            out.append([(float(e[a]), float(e[b])) for a, b in zip(np.where(step == 1)[0], np.where(step == -1)[0])])
        return out

    def mean(self, x, n=1000, seed=0):
        return self.sample(x, n, seed).mean(1)

    def log_prob(self, x, y, n=2000, seed=0):
        """Monte Carlo estimate of log p(y | x) for scalar y, in the original units of y."""
        if self.vector:
            raise NotImplementedError("density estimates degrade quickly with the dimension of y")
        return self._log_prob_std(self._x(x), self._y(y), n, jax.random.key(seed)) - np.log(self.ys)


class GPProbNet(ProbNet):
    """A GP for the function, a sampling net for the shape of the noise.

    y = m(x) + s_f(x) xi + s_n z, with m the GP's predictive mean, s_f its
    uncertainty about the function (small near data, large away from it), s_n
    its noise level, xi ~ N(0, 1), and z drawn from one of three models chosen
    by validation CRPS: N(0, 1) (the GP itself), one learned distribution shared
    by all x, or a sampling net conditional on x. `self.pick` records the winner.

    Scalar y only. The GP is exact, so it is fitted on at most `gp_points`.
    """

    def __init__(self, gp_points=1000, **kw):
        super().__init__(**kw)
        self.gp_points = gp_points

    def _fit_std(self, key, xt, yt, xv, yv):
        if yt.ndim != 1:
            raise NotImplementedError("GPProbNet handles scalar targets")
        x64, y64 = np.asarray(xt, np.float64), np.asarray(yt, np.float64)
        idx = np.arange(len(x64))
        if len(idx) > self.gp_points:
            idx = np.random.default_rng(self.seed).choice(len(idx), self.gp_points, replace=False)
        self.gp = _fit_gp(x64[idx], y64[idx])

        def total(x):
            m, s_f, s_n = gp_parts(self.gp, x)
            return m, np.sqrt(s_f**2 + s_n**2)

        m, s = total(x64)
        m[idx], s[idx] = gp_loo(self.gp, y64[idx])  # leave-one-out where the GP saw the point
        mv, sv = total(xv)
        zt = jnp.asarray((y64 - m) / np.maximum(s, 1e-4), jnp.float32)
        zv = jnp.asarray((np.asarray(yv, np.float64) - mv) / np.maximum(sv, 1e-4), jnp.float32)

        k1, k2, k3 = jax.random.split(key, 3)
        blank = lambda a: jnp.zeros_like(a[:, :1])
        self.models = {"cond": self._train(k1, xt, zt, xv, zv), "shape": self._train(k2, blank(xt), zt, blank(xv), zv)}
        score = {"gauss": metrics.crps(np.asarray(jax.random.normal(k3, (len(zv), 400))), zv).mean()}
        score["cond"] = metrics.crps(_draw(*self.models["cond"], k3, xv, 400), zv).mean()
        score["shape"] = metrics.crps(_draw(*self.models["shape"], k3, blank(xv), 400), zv).mean()
        self.val_crps = {k: float(v) for k, v in score.items()}
        self.pick = min(score, key=score.get)

    def _z_inputs(self, xs):
        return jnp.zeros_like(xs[:, :1]) if self.pick == "shape" else xs

    def _sample_std(self, xs, n, key):
        k1, k2 = jax.random.split(key)
        m, s_f, s_n = gp_parts(self.gp, xs)
        if self.pick == "gauss":
            z = np.asarray(jax.random.normal(k1, (len(xs), n)))
        else:
            z = _draw(*self.models[self.pick], k1, self._z_inputs(xs), n)
        xi = np.asarray(jax.random.normal(k2, z.shape))
        return m[:, None] + s_f[:, None] * xi + s_n * z

    def _log_prob_std(self, xs, ys, n, key):
        from jax.scipy.special import logsumexp
        from jax.scipy.stats import norm

        m, s_f, s_n = gp_parts(self.gp, xs)
        ys = np.asarray(ys, np.float64)
        if self.pick == "gauss":
            return np.asarray(norm.logpdf(ys, m, np.sqrt(s_f**2 + s_n**2)))
        mu, sd = sgn.forward_k(*self.models[self.pick], key, self._z_inputs(xs), n)
        comp = norm.logpdf(ys, m + s_n * mu, jnp.sqrt((s_n * sd) ** 2 + s_f**2))
        return np.asarray(logsumexp(comp, 0) - np.log(n))
