"""Baseline allocation methods (real-Mbps space, multi-horizon compatible).

Each method returns an [n_test, P, H] allocation tensor that can be directly
compared to a model's P95 output. The H horizons sit on the last axis to match
Transformer output shape [B, P, H, Q].

Convention: target time = test_t[i] + h, where h is the horizon (in 15-min steps).
"""

import numpy as np

WEEK_STEPS = 7 * 96   # 672 steps per week at 15-min granularity


def static_peak(train_flows, n_test, horizons):
    """Per-pair training maximum. Horizon-independent: all horizons use the same value."""
    val = train_flows.max(axis=0)                                       # [P]
    P, H = train_flows.shape[1], len(horizons)
    return np.broadcast_to(val.reshape(1, P, 1), (n_test, P, H)).copy()


def static_p95(train_flows, n_test, horizons):
    """Per-pair training P95. Horizon-independent."""
    val = np.quantile(train_flows, 0.95, axis=0)                        # [P]
    P, H = train_flows.shape[1], len(horizons)
    return np.broadcast_to(val.reshape(1, P, 1), (n_test, P, H)).copy()


def naive_last_residual_p95(train_flows, full_flows, test_t, horizons):
    """Conformal-style: point predictor = actual(t); alloc(t+h) = actual(t) + r_p95(h).

    Residual depends on horizon: r(h) = actual(s+h) - actual(s) over training set.
    Shorter horizons have smaller residuals (traffic changes slowly);
    longer horizons have larger residuals.
    """
    n_test, P, H = len(test_t), train_flows.shape[1], len(horizons)
    out = np.zeros((n_test, P, H), dtype=np.float32)
    for h_idx, h in enumerate(horizons):
        resid = train_flows[h:] - train_flows[:-h]
        r_p95 = np.quantile(resid, 0.95, axis=0)                        # [P]
        out[:, :, h_idx] = np.maximum(full_flows[test_t] + r_p95[None, :], 0)
    return out


def seasonal_naive_residual_p95(train_flows, full_flows, test_t, horizons):
    """Conformal-style: point predictor = traffic at (t+h - WEEK); alloc = that + r_p95.

    Residual is horizon-independent: r = actual(s) - actual(s - WEEK) over training.
    Point predictor varies with horizon (each horizon looks at its own "same time last week"),
    but the residual P95 is shared across horizons.
    """
    n_test, P, H = len(test_t), train_flows.shape[1], len(horizons)
    resid = train_flows[WEEK_STEPS:] - train_flows[:-WEEK_STEPS]
    r_p95 = np.quantile(resid, 0.95, axis=0)                            # [P]
    out = np.zeros((n_test, P, H), dtype=np.float32)
    for h_idx, h in enumerate(horizons):
        # t+h-WEEK must be >= 0; in practice test_t starts well past WEEK_STEPS
        out[:, :, h_idx] = np.maximum(
            full_flows[test_t + h - WEEK_STEPS] + r_p95[None, :], 0
        )
    return out


if __name__ == "__main__":
    # Shape sanity test
    rng = np.random.default_rng(0)
    T, P = 11460, 462
    full = rng.uniform(0, 100, (T, P)).astype(np.float32)
    train_end = 9168
    train = full[:train_end]
    test_t = np.arange(10313, 10313 + 5)   # 5 test timesteps
    horizons = (1, 4, 16)

    for name, fn in [
        ("static_peak", lambda: static_peak(train, len(test_t), horizons)),
        ("static_p95",  lambda: static_p95(train, len(test_t), horizons)),
        ("naive_last",  lambda: naive_last_residual_p95(train, full, test_t, horizons)),
        ("seasonal",    lambda: seasonal_naive_residual_p95(train, full, test_t, horizons)),
    ]:
        a = fn()
        print(f"{name:14s}  shape={a.shape}  "
              f"per-horizon mean alloc = {a.mean(axis=(0,1)).round(1).tolist()}")
