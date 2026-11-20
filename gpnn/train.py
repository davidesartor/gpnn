"""Minibatch training with optional early stopping on a validation loss."""

import jax
import jax.numpy as jnp
import optax


def fit(
    key,
    params,
    loss_fn,
    x,
    y,
    *,
    steps=6000,
    batch=128,
    lr=2e-3,
    wd=0.0,
    val=None,
    val_loss_fn=None,
    eval_every=250,
    clip=10.0,
):
    """Minimise loss_fn(params, key, xb, yb). Returns (params, history).

    If val=(xv, yv) is given, the parameters with the best validation loss
    (checked every eval_every steps) are returned.
    """
    sched = optax.cosine_decay_schedule(lr, steps, alpha=0.05)
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adamw(sched, weight_decay=wd))
    state = opt.init(params)
    x, y = jnp.asarray(x), jnp.asarray(y)
    if val is not None:
        val = tuple(jnp.asarray(v) for v in val)
    n = x.shape[0]
    val_loss_fn = val_loss_fn or loss_fn

    def step(carry, k):
        p, s = carry
        k1, k2 = jax.random.split(k)
        idx = jax.random.randint(k1, (min(batch, n),), 0, n)
        loss, g = jax.value_and_grad(loss_fn)(p, k2, x[idx], y[idx])
        u, s = opt.update(g, s, p)
        return (optax.apply_updates(p, u), s), loss

    @jax.jit
    def run(p, s, k):
        (p, s), losses = jax.lax.scan(step, (p, s), jax.random.split(k, eval_every))
        return p, s, losses.mean()

    val_eval = jax.jit(val_loss_fn)
    best, best_val, hist = params, jnp.inf, []
    for i, k in enumerate(jax.random.split(key, max(1, steps // eval_every))):
        k1, k2 = jax.random.split(k)
        params, state, tr = run(params, state, k1)
        if val is not None:
            v = float(val_eval(params, k2, *val))
            if v < best_val:
                best, best_val = params, v
        else:
            v, best = float("nan"), params
        hist.append((int((i + 1) * eval_every), float(tr), v))
    return best, hist
