"""Bandwidth allocation evaluation metrics (computed in real-Mbps space).

Standard usage:
    metrics = compute_metrics(pred_z, target_z, mean, std, q_idx=2)
        pred_z   [N, 462, 3]   model output (log1p + z-score space)
        target_z [N, 462]      actual traffic (log1p + z-score space)
        mean,std [462]         scaler parameters (from all.pt)
        q_idx                  which quantile is used as the allocation; 0=P50, 1=P90, 2=P95
"""

import torch


def inverse_transform(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """z-score(log1p) -> real Mbps, clamped to >= 0.

    Convention: SD-pair dim is axis -1 for 2D inputs, axis 1 for 3D+ inputs.
        2D [T, P] or [B, P]:        default broadcast works
        3D+ [B, P, ...]:            reshape mean/std to [1, P, 1, ...] to broadcast correctly
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


def compute_alloc_metrics(
    alloc_z: torch.Tensor,
    target_z: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict:
    """Single-quantile setting: model output is the allocation directly,
    no P50/P90, no MAE_P50.

    alloc_z, target_z: same shape, both in z-score space.
    """
    alloc = inverse_transform(alloc_z, mean, std)
    actual = inverse_transform(target_z, mean, std)

    sum_actual = actual.sum().item()
    sum_alloc = alloc.sum().item()
    diff = actual - alloc                       # > 0 indicates a violation magnitude
    violations = diff > 0
    n_viols = int(violations.sum().item())

    avg_violation_size = (
        diff[violations].mean().item() if n_viols > 0 else 0.0
    )

    return {
        "sla_violation_rate": violations.float().mean().item(),
        "utilization": sum_actual / max(sum_alloc, 1e-6),
        "avg_wasted_mbps": (alloc - actual).clamp(min=0).mean().item(),
        "avg_actual_mbps": actual.mean().item(),
        "avg_alloc_mbps": alloc.mean().item(),
        "avg_violation_size_mbps": avg_violation_size,
        "total_unmet_mbps": diff.clamp(min=0).sum().item(),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    N, P = 100, 462
    # Simulated mean / std (typical log1p space)
    mean = torch.rand(P) * 5
    std = torch.rand(P) * 1.5 + 0.5

    target_real = torch.rand(N, P) * 100      # 0..100 Mbps
    target_z = (torch.log1p(target_real) - mean) / std

    # Simulated model output: P50 close to truth, P95 shifted upward
    noise = torch.randn(N, P) * 0.3
    pred_z = torch.stack([target_z + noise,
                          target_z + noise + 0.5,
                          target_z + noise + 1.0], dim=-1)

    m = compute_metrics(pred_z, target_z, mean, std, q_idx=2)
    print("Single-horizon (legacy):")
    for k, v in m.items():
        print(f"  {k:24s} {v:>10.4f}")

    # Multi-horizon test
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
