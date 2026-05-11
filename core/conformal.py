"""Split Conformal Prediction for bandwidth allocation calibration.

参考:
  Vovk, Gammerman & Shafer 2005, "Algorithmic Learning in a Random World"
  Romano, Patterson & Candès 2019, "Conformalized Quantile Regression" (CQR)

核心思想:
  在 calibration set (我们用 val 集) 上算模型的 conformity score,
  再把 (1-α) 分位数加到 test 集的预测上, 数学保证 marginal coverage >= 1-α.

我们的设置:
  pred, target 都在 real Mbps 空间 (已经 inverse_transform).
  score = target - pred (> 0 时为违约量).
  per-pair, per-horizon 独立计算 (axis=0 是时间维).
"""

import numpy as np


def fit_correction(pred_cal: np.ndarray,
                   target_cal: np.ndarray,
                   alpha: float = 0.05) -> np.ndarray:
    """在 calibration set 上拟合 conformal 修正量.

    Args:
        pred_cal:   [N_cal, ...]   real Mbps 空间, 模型预测的 alloc
        target_cal: [N_cal, ...]   same shape, real Mbps 空间, 真实流量
        alpha:                     目标违约率 (默认 0.05 = 5% SLA target)

    Returns:
        q_hat: shape = pred_cal.shape[1:]   每个 (pair, horizon, ...) 一个修正量
               q_hat > 0 → 模型偏低估, 需要抬高 alloc
               q_hat < 0 → 模型偏高估, 可以压低 alloc (省 buffer)
    """
    if pred_cal.shape != target_cal.shape:
        raise ValueError(f"shape mismatch: pred {pred_cal.shape} vs target {target_cal.shape}")
    scores = target_cal - pred_cal                # > 0 = 违约量
    n = scores.shape[0]
    # finite-sample 修正: (n+1)/n
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, q_level, axis=0)


def apply_correction(pred: np.ndarray, q_hat: np.ndarray) -> np.ndarray:
    """应用 conformal 修正. alloc = max(pred + q_hat, 0).

    Args:
        pred:   [N, ...]   real Mbps 空间
        q_hat:  shape = pred.shape[1:]   来自 fit_correction

    Returns:
        alloc:  [N, ...]   clip 到 >= 0
    """
    return np.maximum(pred + q_hat[None], 0)


if __name__ == "__main__":
    # 单元测试: 验证 conformal 给出严格的 marginal coverage
    np.random.seed(0)
    N_cal, N_test, P = 1000, 1000, 5

    # 模拟数据: actual ~ N(100, 20), pred = actual - bias (per pair)
    actual_cal  = np.abs(np.random.normal(100, 20, size=(N_cal, P)))
    actual_test = np.abs(np.random.normal(100, 20, size=(N_test, P)))
    bias_per_pair = np.array([0, -5, +5, +10, -10])
    pred_cal  = actual_cal  + bias_per_pair[None, :] + np.random.normal(0, 5, size=(N_cal, P))
    pred_test = actual_test + bias_per_pair[None, :] + np.random.normal(0, 5, size=(N_test, P))

    # 不加 conformal: 各 pair 的违约率应该有偏 (bias 影响)
    sla_raw = (pred_test < actual_test).mean(axis=0)
    print(f"Raw SLA (target 5%): per-pair {sla_raw.round(3)}, mean {sla_raw.mean():.3f}")

    # 加 conformal: 应该 ≈ 5% per pair
    q_hat = fit_correction(pred_cal, actual_cal, alpha=0.05)
    print(f"q_hat per pair: {q_hat.round(2)}  (bias 越大 q_hat 越正)")

    alloc_calibrated = apply_correction(pred_test, q_hat)
    sla_cal = (alloc_calibrated < actual_test).mean(axis=0)
    print(f"Calibrated SLA: per-pair {sla_cal.round(3)}, mean {sla_cal.mean():.3f}")
    print(f"  → 期望 ≈ 0.05, 实际 mean = {sla_cal.mean():.4f}")
