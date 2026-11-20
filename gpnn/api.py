"""A small front end: fit a stochastic network, ask it for intervals.

    net = ProbNet().fit(x, y)
    lo, hi = net.interval(x_new, level=0.9)
    draws = net.sample(x_new, n=1000)

The network is a sampler for p(y | x). Everything it reports (intervals,
quantiles, means, intervals for functions of a vector y) is read off its
samples; there is no separate uncertainty head and no posterior over weights.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import multi, sgn
from .methods import refit_noise
from .train import fit


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
        calibrate: after training, rescale the noise on the validation split so
        the predictive spread reflects error on unseen data, not on the fit."""
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
        xt, yt = self._x(x), self._y(y)
        xv, yv = self._x(val[0]), self._y(val[1])

        self.cfg = sgn.SGN(d_out=y.shape[1] if self.vector else 1, **self.hp)
        k0, k1, k2 = jax.random.split(jax.random.key(self.seed), 3)
        params = sgn.init(self.cfg, k0, x.shape[1])
        loss, val_loss = self._loss(self.K), self._loss(max(self.K, 32))
        params, self.history = fit(k1, params, loss, xt, yt, val=(xv, yv), val_loss_fn=val_loss, steps=self.steps, batch=self.batch, lr=self.lr)
        if self.calibrate and not self.vector:
            params = refit_noise(self.cfg, params, k2, xv, yv, "es_rb", "temp")
        self.params = params
        return self

    def _loss(self, K):
        if not self.vector:
            return sgn.make_loss(self.cfg, "es_rb", K)

        def loss(p, k, x, y):
            k1, k2 = jax.random.split(k)
            return multi.loss_es(*sgn.forward_k(self.cfg, p, k1, x, K), y, k2).mean()

        return loss

    def _x(self, x):
        return jnp.asarray((np.asarray(x, np.float64).reshape(len(x), -1) - self.xm) / self.xs, jnp.float32)

    def _y(self, y):
        return jnp.asarray((np.asarray(y, np.float64) - self.ym) / self.ys, jnp.float32)

    # ------------------------------------------------------------ prediction

    def sample(self, x, n=1000, seed=0):
        """n draws from p(y | x) per row: (N, n) for scalar y, (N, n, D) for vector y."""
        key, xs = jax.random.key(seed), self._x(x)

        @jax.jit
        def draw(k, xb):
            k1, k2 = jax.random.split(k)
            mu, sd = sgn.forward_k(self.cfg, self.params, k1, xb, n)
            return jnp.swapaxes(mu + sd * jax.random.normal(k2, mu.shape), 0, 1)

        out = [draw(jax.random.fold_in(key, i), xs[i : i + 256]) for i in range(0, len(xs), 256)]
        return np.concatenate([np.asarray(o) for o in out]) * self.ys + self.ym

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
        from .metrics import hdr

        edges, mask = hdr(self._scalar_samples(x, n, fn, seed), level)
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
        lp = sgn.log_prob(self.cfg, self.params, jax.random.key(seed), self._x(x), self._y(y), n)
        return np.asarray(lp) - np.log(self.ys)
