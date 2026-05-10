"""带宽分配的 baseline 方法 (real Mbps 空间, 多 horizon 兼容).

每个方法返回 [n_test, P, H] 的 alloc 矩阵, 与模型输出 (取 P95 后) 可直接比较.
H 个 horizon 在最后一维, 跟 Transformer 输出 [B, P, H, Q] 对齐.

约定: target 时刻 = test_t[i] + h, 其中 h 是 horizon (单位 = 15 min).
"""

import numpy as np

WEEK_STEPS = 7 * 96   # = 672, 一周的 15-min step


def static_peak(train_flows, n_test, horizons):
    """每 SD 对取 train 集最大值. horizon 无关——所有 horizon 用同一个值."""
    val = train_flows.max(axis=0)                                       # [P]
    P, H = train_flows.shape[1], len(horizons)
    return np.broadcast_to(val.reshape(1, P, 1), (n_test, P, H)).copy()


def static_p95(train_flows, n_test, horizons):
    """每 SD 对取 train 集 P95. horizon 无关."""
    val = np.quantile(train_flows, 0.95, axis=0)                        # [P]
    P, H = train_flows.shape[1], len(horizons)
    return np.broadcast_to(val.reshape(1, P, 1), (n_test, P, H)).copy()


def naive_last_residual_p95(train_flows, full_flows, test_t, horizons):
    """conformal 风格: 点预测 = 当前值 actual(t); alloc(t+h) = actual(t) + r_p95(h).

    残差跟 horizon 有关: r(h) = actual(s+h) - actual(s) over training.
    短 horizon 残差小 (变化慢), 长 horizon 残差大.
    """
    n_test, P, H = len(test_t), train_flows.shape[1], len(horizons)
    out = np.zeros((n_test, P, H), dtype=np.float32)
    for h_idx, h in enumerate(horizons):
        resid = train_flows[h:] - train_flows[:-h]
        r_p95 = np.quantile(resid, 0.95, axis=0)                        # [P]
        out[:, :, h_idx] = np.maximum(full_flows[test_t] + r_p95[None, :], 0)
    return out


def seasonal_naive_residual_p95(train_flows, full_flows, test_t, horizons):
    """conformal 风格: 点预测 = (t+h-WEEK) 时刻流量; alloc = 该值 + r_p95.

    残差跟 horizon 无关: r = actual(s) - actual(s - WEEK) over training.
    点预测随 horizon 变 (各 horizon 看不同的 "上周同时刻"), 但 r_p95 共用.
    """
    n_test, P, H = len(test_t), train_flows.shape[1], len(horizons)
    resid = train_flows[WEEK_STEPS:] - train_flows[:-WEEK_STEPS]
    r_p95 = np.quantile(resid, 0.95, axis=0)                            # [P]
    out = np.zeros((n_test, P, H), dtype=np.float32)
    for h_idx, h in enumerate(horizons):
        # 注意可能下溢: t+h-WEEK 必须 >= 0, 但 test_t 起点 (10313) 加 h - WEEK 远 > 0
        out[:, :, h_idx] = np.maximum(
            full_flows[test_t + h - WEEK_STEPS] + r_p95[None, :], 0
        )
    return out


if __name__ == "__main__":
    # 形状 sanity test
    rng = np.random.default_rng(0)
    T, P = 11460, 462
    full = rng.uniform(0, 100, (T, P)).astype(np.float32)
    train_end = 9168
    train = full[:train_end]
    test_t = np.arange(10313, 10313 + 5)   # 5 个测试时刻
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
