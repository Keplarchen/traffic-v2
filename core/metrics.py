"""带宽分配的工程指标 (在真实 Mbps 空间计算).

标准用法:
    metrics = compute_metrics(pred_z, target_z, mean, std, q_idx=2)
        pred_z   [N, 462, 3]   模型输出 (log1p + z-score 空间)
        target_z [N, 462]      真实流量 (log1p + z-score 空间)
        mean,std [462]         scaler 参数 (来自 all.pt)
        q_idx                  用哪个分位数作为带宽配额; 0=P50, 1=P90, 2=P95
"""

import torch


def inverse_transform(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """z-score(log1p) -> 真实 Mbps, clamp 到 >= 0.

    约定: SD 对维度位于 axis -1 (2D 时) 或 axis 1 (3D+ 时, 即 batch 之后).
        2D [T, P] 或 [B, P]:        默认广播即可
        3D+ [B, P, ...]:           需把 mean/std 显式 reshape 成 [1, P, 1, ...]
                                    再广播 (避免对齐到错误的最后一维)
    """
    if z.ndim >= 3:
        shape = [1] * z.ndim
        shape[1] = -1
        mean = mean.view(shape)
        std = std.view(shape)
    return torch.expm1(z * std + mean).clamp(min=0)


def compute_metrics(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    q_idx: int = 2,
) -> dict:
    pred_p50 = inverse_transform(pred_z[..., 0], mean, std)
    alloc = inverse_transform(pred_z[..., q_idx], mean, std)
    actual = inverse_transform(target_z, mean, std)

    sum_actual = actual.sum().item()
    sum_alloc = alloc.sum().item()
    return {
        "mae_p50": (pred_p50 - actual).abs().mean().item(),
        "sla_violation_rate": (alloc < actual).float().mean().item(),
        "utilization": sum_actual / max(sum_alloc, 1e-6),
        "avg_wasted_mbps": (alloc - actual).clamp(min=0).mean().item(),
        "avg_actual_mbps": actual.mean().item(),
        "avg_alloc_mbps": alloc.mean().item(),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    N, P = 100, 462
    # 模拟 mean / std (典型 log1p 空间)
    mean = torch.rand(P) * 5
    std = torch.rand(P) * 1.5 + 0.5

    target_real = torch.rand(N, P) * 100      # 0..100 Mbps
    target_z = (torch.log1p(target_real) - mean) / std

    # 模拟模型输出: P50 接近真实, P95 偏高
    noise = torch.randn(N, P) * 0.3
    pred_z = torch.stack([target_z + noise,
                          target_z + noise + 0.5,
                          target_z + noise + 1.0], dim=-1)

    m = compute_metrics(pred_z, target_z, mean, std, q_idx=2)
    print("Single-horizon (legacy):")
    for k, v in m.items():
        print(f"  {k:24s} {v:>10.4f}")

    # 多 horizon 测试
    H = 3
    target_real_mh = torch.rand(N, P, H) * 100
    target_z_mh = (torch.log1p(target_real_mh) - mean.view(1, -1, 1)) / std.view(1, -1, 1)
    noise_mh = torch.randn(N, P, H) * 0.3
    pred_z_mh = torch.stack([target_z_mh + noise_mh,
                             target_z_mh + noise_mh + 0.5,
                             target_z_mh + noise_mh + 1.0], dim=-1)
    print(f"\nMulti-horizon test: pred {tuple(pred_z_mh.shape)}, target {tuple(target_z_mh.shape)}")
    m_mh = compute_metrics(pred_z_mh, target_z_mh, mean, std, q_idx=2)
    for k, v in m_mh.items():
        print(f"  {k:24s} {v:>10.4f}")
