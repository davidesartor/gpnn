"""Is a plain NN the 1-sample estimate of a probabilistic graph?

Model: every hidden node is a conditional distribution, not a value.

    h_l | h_{l-1} ~ N(W_l phi(h_{l-1}) + b_l, diag(s_l^2))      l = 1..L
    y   | h_L     ~ N(w . phi(h_L) + b, s_y^2)

Weights are point estimates (no epistemic uncertainty anywhere); the only
randomness is in the nodes. The likelihood marginalises them:

    log p(y|x) = log E_{h ~ p(h|x)} [ p(y | h_L) ]

Estimating the expectation with K ancestral samples gives the objective

    L_K = E [ log 1/K sum_k p(y | h_L^k) ]

K=1 is a noise-injected forward pass + backprop, i.e. ordinary training.
K>1 only changes how the K backprop gradients are weighted: by
w_k ∝ p(y | h_L^k), a self-normalised estimate of the posterior over nodes.

Task: Bishop's inverse problem, where p(y|x) is multimodal, so a model that
can only represent a mean has something to lose.
"""

import jax
import jax.numpy as jnp
import matplotlib
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH, DEPTH = 64, 2
STEPS, BATCH, LR = 8000, 256, 2e-3
K_EVAL = 2000


def make_data(key, n):
    k1, k2 = jax.random.split(key)
    y = jax.random.uniform(k1, (n,))
    x = y + 0.3 * jnp.sin(2 * jnp.pi * y) + 0.05 * jax.random.normal(k2, (n,))
    return x, y


def init(key, noise0):
    sizes = [1] + [WIDTH] * DEPTH
    keys = jax.random.split(key, DEPTH + 1)
    layers = []
    for k, m, n in zip(keys, sizes[:-1], sizes[1:]):
        layers.append(
            dict(
                W=jax.random.normal(k, (n, m)) / jnp.sqrt(m),
                b=jnp.zeros(n),
                log_s=jnp.full(n, jnp.log(noise0)),
            )
        )
    out = dict(w=jax.random.normal(keys[-1], (WIDTH,)) / jnp.sqrt(WIDTH), b=0.0)
    return dict(layers=layers, out=out, log_sy=jnp.log(0.3))


def sample_mean_y(params, key, x, stochastic):
    """One ancestral sample through the graph; returns E[y | h_L]."""
    h = jnp.atleast_1d(x)
    for layer, k in zip(params["layers"], jax.random.split(key, DEPTH)):
        h = layer["W"] @ h + layer["b"]
        if stochastic:
            h = h + jnp.exp(layer["log_s"]) * jax.random.normal(k, h.shape)
        h = jnp.tanh(h)
    return params["out"]["w"] @ h + params["out"]["b"]


def log_lik(params, key, x, y, K, stochastic=True):
    """log 1/K sum_k N(y | mu(h^k), s_y^2) for a single (x, y)."""
    mus = jax.vmap(lambda k: sample_mean_y(params, k, x, stochastic))(
        jax.random.split(key, K)
    )
    sy = jnp.exp(params["log_sy"])
    logp = -0.5 * ((y - mus) / sy) ** 2 - jnp.log(sy) - 0.5 * jnp.log(2 * jnp.pi)
    return jax.nn.logsumexp(logp) - jnp.log(K)


def batch_log_lik(params, key, x, y, K, stochastic=True):
    keys = jax.random.split(key, x.shape[0])
    return jax.vmap(lambda k, xi, yi: log_lik(params, k, xi, yi, K, stochastic))(
        keys, x, y
    )


def train(key, x, y, K, stochastic=True):
    k_init, k_loop = jax.random.split(key)
    params = init(k_init, noise0=1.0)
    opt = optax.adam(LR)
    state = opt.init(params)

    def loss(p, k, xb, yb):
        return -batch_log_lik(p, k, xb, yb, K, stochastic).mean()

    @jax.jit
    def step(carry, k):
        p, s = carry
        k_idx, k_noise = jax.random.split(k)
        idx = jax.random.choice(k_idx, x.shape[0], (BATCH,), replace=False)
        g = jax.grad(loss)(p, k_noise, x[idx], y[idx])
        u, s = opt.update(g, s)
        return (optax.apply_updates(p, u), s), None

    (params, _), _ = jax.lax.scan(step, (params, state), jax.random.split(k_loop, STEPS))
    return params


def main():
    key = jax.random.key(0)
    k_tr, k_te, k_fit, k_ev, k_plot = jax.random.split(key, 5)
    x_tr, y_tr = make_data(k_tr, 4000)
    x_te, y_te = make_data(k_te, 2000)

    runs = [
        ("deterministic", dict(K=1, stochastic=False)),
        ("K=1", dict(K=1)),
        ("K=5", dict(K=5)),
        ("K=50", dict(K=50)),
    ]
    xs = jnp.linspace(-0.1, 1.1, 300)
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.6), sharey=True)

    print(f"{'objective':>14} {'test log p(y|x)':>16} {'node noise':>11} {'s_y':>7}")
    for ax, (name, cfg) in zip(axes, runs):
        params = train(k_fit, x_tr, y_tr, **cfg)
        stoch = cfg.get("stochastic", True)
        # Same estimator for every model: K_EVAL-sample marginal likelihood.
        ll = batch_log_lik(params, k_ev, x_te, y_te, K_EVAL if stoch else 1, stoch).mean()
        s_node = (
            float(jnp.mean(jnp.exp(jnp.stack([l["log_s"] for l in params["layers"]]))))
            if stoch
            else 0.0
        )
        sy = float(jnp.exp(params["log_sy"]))
        print(f"{name:>14} {float(ll):16.3f} {s_node:11.3f} {sy:7.3f}")

        # Samples from the model's p(y|x), including observation noise.
        n_draw = 40
        k_h, k_y = jax.random.split(k_plot)
        mus = jax.vmap(
            lambda xi, k: jax.vmap(lambda kk: sample_mean_y(params, kk, xi, stoch))(
                jax.random.split(k, n_draw)
            )
        )(xs, jax.random.split(k_h, xs.shape[0]))
        ys = mus + sy * jax.random.normal(k_y, mus.shape)
        ax.scatter(x_te[:600], y_te[:600], s=4, c="0.75", label="data")
        ax.scatter(jnp.repeat(xs, n_draw), ys.ravel(), s=1, c="C3", alpha=0.35, label="model samples")
        ax.set_title(f"{name}   test ll = {float(ll):.2f}")
        ax.set_xlabel("x")
        ax.set_ylim(-0.3, 1.3)
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper left", markerscale=4, frameon=False)
    fig.tight_layout()
    fig.savefig("one_sample.png", dpi=150)


if __name__ == "__main__":
    main()
