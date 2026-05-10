"""把 flows.npy 中超过阈值的异常值用线性插值替换.

读取:  data/flows.npy        [T, 462]
写出:  data/flows_clean.npy  [T, 462]

阈值之上的值被视为 SNMP 计数器溢出/路由器重启造成的伪信号,
按 SD 对沿时间轴做线性插值替换.
"""

import argparse
from pathlib import Path

import numpy as np


def interpolate_nan_1d(arr: np.ndarray) -> np.ndarray:
    """线性填充 1D 数组里的 NaN; 边界 NaN 用最近的有效值."""
    isnan = np.isnan(arr)
    if not isnan.any():
        return arr
    valid_idx = np.where(~isnan)[0]
    return np.interp(np.arange(arr.size), valid_idx, arr[valid_idx])


def main():
    parser = argparse.ArgumentParser()
    data_dir = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--in", dest="inp", type=Path, default=data_dir / "flows.npy")
    parser.add_argument("--out", type=Path, default=data_dir / "flows_clean.npy")
    parser.add_argument(
        "--threshold", type=float, default=10_000.0,
        help="Mbps; 超过此值的 cell 被视为异常 (默认 10 Gbps)",
    )
    args = parser.parse_args()

    flows = np.load(args.inp).astype(np.float64)
    mask = flows > args.threshold
    print(f"输入 {args.inp.name}: shape={flows.shape}")
    print(f"阈值 {args.threshold:.0f} Mbps -> 异常 cell {mask.sum()} "
          f"({100*mask.mean():.4f}%)")

    flows[mask] = np.nan
    for j in range(flows.shape[1]):
        flows[:, j] = interpolate_nan_1d(flows[:, j])

    assert not np.isnan(flows).any(), "插值后仍有 NaN, 检查是否有全异常列"

    flows = flows.astype(np.float32)
    np.save(args.out, flows)
    print()
    print(f"输出 {args.out.name}: shape={flows.shape}")
    print(f"  min={flows.min():.3f}  max={flows.max():.2f}  "
          f"mean={flows.mean():.2f}  median={np.median(flows):.3f}")


if __name__ == "__main__":
    main()
