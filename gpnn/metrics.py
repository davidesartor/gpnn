"""Metrics for predictive distributions represented by samples.

S is always an (N, M) array: M draws from the predicted p(y | x_n).
"""

import numpy as np

LEVELS = (0.5, 0.8, 0.9, 0.95)


def crps(S, y):
    """Unbiased sample estimate of CRPS(F_n, y_n) = E|Y-y| - 0.5 E|Y-Y'|. Shape (N,)."""
    S = np.sort(np.asarray(S, np.float64), 1)
    M = S.shape[1]
    fit = np.abs(S - np.asarray(y)[:, None]).mean(1)
    i = np.arange(1, M + 1)
    spread = (2 * (2 * i - M - 1) * S).sum(1) / (M * (M - 1))
    return fit - 0.5 * spread


def interval(S, level):
    a = (1 - level) / 2
    return np.quantile(S, a, axis=1), np.quantile(S, 1 - a, axis=1)


def pit(S, y):
    return (np.asarray(S) < np.asarray(y)[:, None]).mean(1)


def calibration_error(S, y, grid=99):
    """Mean |P(PIT <= t) - t| over a grid of t. 0 for a marginally calibrated model."""
    u = pit(S, y)
    t = np.arange(1, grid + 1) / (grid + 1)
    return float(np.abs((u[:, None] <= t[None]).mean(0) - t).mean())


def summarize(S, y, scale=1.0):
    """Marginal metrics. `scale` converts standardised y back to original units."""
    S, y = np.asarray(S), np.asarray(y)
    out = dict(
        crps=float(crps(S, y).mean() * scale),
        rmse=float(np.sqrt(((S.mean(1) - y) ** 2).mean()) * scale),
        cal_err=calibration_error(S, y),
    )
    for lv in LEVELS:
        lo, hi = interval(S, lv)
        a = 1 - lv
        tag = f"{int(lv * 100)}"
        out[f"cov{tag}"] = float(((y >= lo) & (y <= hi)).mean())
        out[f"width{tag}"] = float((hi - lo).mean() * scale)
        score = (hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)
        out[f"iscore{tag}"] = float(score.mean() * scale)
    return out


def against_truth(S, T):
    """Conditional metrics when the true p(y | x_n) can be sampled.

    T is (N, M') draws from the truth at the same inputs.
    w1: mean Wasserstein-1 distance between predicted and true conditionals.
    ccov_err{L}: mean over x of |P_true(y in predicted L-interval | x) - L|.
    """
    S, T = np.asarray(S, np.float64), np.asarray(T, np.float64)
    q = (np.arange(200) + 0.5) / 200
    out = dict(w1=float(np.abs(np.quantile(S, q, axis=1) - np.quantile(T, q, axis=1)).mean()))
    for lv in LEVELS:
        lo, hi = interval(S, lv)
        cc = ((T >= lo[:, None]) & (T <= hi[:, None])).mean(1)
        out[f"ccov_err{int(lv * 100)}"] = float(np.abs(cc - lv).mean())
    return out
