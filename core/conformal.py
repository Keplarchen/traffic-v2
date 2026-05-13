"""Split Conformal Prediction for bandwidth allocation calibration.

References:
  Vovk, Gammerman & Shafer (2005), "Algorithmic Learning in a Random World"
  Romano, Patterson & Candes (2019), "Conformalized Quantile Regression" (CQR)

Approach:
  Compute conformity scores on a calibration set (we use the validation split),
  take the (1-alpha) quantile per pair, and add it to test-time predictions.
  Under exchangeability this gives marginal coverage >= 1-alpha.

Setup:
  pred and target are in real-Mbps space (already inverse_transformed).
  score = target - pred (positive value = violation magnitude).
  Per-pair, per-horizon correction (axis=0 is time).
"""

import numpy as np


def fit_correction(pred_cal: np.ndarray,
                   target_cal: np.ndarray,
                   alpha: float = 0.05) -> np.ndarray:
    """Fit conformal correction on a calibration set.

    Args:
        pred_cal:   [N_cal, ...]   real Mbps, model-predicted allocation
        target_cal: [N_cal, ...]   same shape, real Mbps, actual traffic
        alpha:                     target violation rate (default 0.05 = 5% SLA target)

    Returns:
        q_hat: shape = pred_cal.shape[1:]
               One correction per (pair, horizon, ...) element.
               q_hat > 0  -> model under-allocates, allocation needs to be raised
               q_hat < 0  -> model over-allocates, allocation can be lowered (saves buffer)
    """
    if pred_cal.shape != target_cal.shape:
        raise ValueError(f"shape mismatch: pred {pred_cal.shape} vs target {target_cal.shape}")
    scores = target_cal - pred_cal                # > 0 = violation magnitude
    n = scores.shape[0]
    # Finite-sample correction: (n+1)/n
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, q_level, axis=0)


def apply_correction(pred: np.ndarray, q_hat: np.ndarray) -> np.ndarray:
    """Apply the conformal correction. alloc = max(pred + q_hat, 0).

    Args:
        pred:   [N, ...]   real Mbps
        q_hat:  shape = pred.shape[1:]   from fit_correction()

    Returns:
        alloc:  [N, ...]   clipped to >= 0
    """
    return np.maximum(pred + q_hat[None], 0)


if __name__ == "__main__":
    # Unit test: verify conformal produces near-target marginal coverage
    np.random.seed(0)
    N_cal, N_test, P = 1000, 1000, 5

    # Simulated data: actual ~ N(100, 20), pred = actual - bias (per pair)
    actual_cal  = np.abs(np.random.normal(100, 20, size=(N_cal, P)))
    actual_test = np.abs(np.random.normal(100, 20, size=(N_test, P)))
    bias_per_pair = np.array([0, -5, +5, +10, -10])
    pred_cal  = actual_cal  + bias_per_pair[None, :] + np.random.normal(0, 5, size=(N_cal, P))
    pred_test = actual_test + bias_per_pair[None, :] + np.random.normal(0, 5, size=(N_test, P))

    # Without conformal: per-pair violation rate is biased by `bias_per_pair`
    sla_raw = (pred_test < actual_test).mean(axis=0)
    print(f"Raw SLA (target 5%): per-pair {sla_raw.round(3)}, mean {sla_raw.mean():.3f}")

    # With conformal: should converge to ~5% per pair
    q_hat = fit_correction(pred_cal, actual_cal, alpha=0.05)
    print(f"q_hat per pair: {q_hat.round(2)}  (larger bias -> larger q_hat)")

    alloc_calibrated = apply_correction(pred_test, q_hat)
    sla_cal = (alloc_calibrated < actual_test).mean(axis=0)
    print(f"Calibrated SLA: per-pair {sla_cal.round(3)}, mean {sla_cal.mean():.3f}")
    print(f"  expected ~0.05, actual mean = {sla_cal.mean():.4f}")
