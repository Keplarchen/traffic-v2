"""Replace outlier values in flows.npy with linear interpolation along the time axis.

Reads:   data/flows.npy        [T, 462]
Writes:  data/flows_clean.npy  [T, 462]

Values above the threshold are treated as artifacts (likely SNMP counter
overflow or router restarts) and replaced by per-SD-pair linear interpolation.
"""

import argparse
from pathlib import Path

import numpy as np


def interpolate_nan_1d(arr: np.ndarray) -> np.ndarray:
    """Linearly fill NaN entries in a 1D array; boundary NaNs use the nearest valid value."""
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
        help="Mbps; cells above this threshold are treated as outliers (default 10 Gbps)",
    )
    args = parser.parse_args()

    flows = np.load(args.inp).astype(np.float64)
    mask = flows > args.threshold
    print(f"Input {args.inp.name}: shape={flows.shape}")
    print(f"Threshold {args.threshold:.0f} Mbps -> {mask.sum()} outlier cells "
          f"({100*mask.mean():.4f}%)")

    flows[mask] = np.nan
    for j in range(flows.shape[1]):
        flows[:, j] = interpolate_nan_1d(flows[:, j])

    assert not np.isnan(flows).any(), "NaN remains after interpolation; check for all-outlier columns"

    flows = flows.astype(np.float32)
    np.save(args.out, flows)
    print()
    print(f"Output {args.out.name}: shape={flows.shape}")
    print(f"  min={flows.min():.3f}  max={flows.max():.2f}  "
          f"mean={flows.mean():.2f}  median={np.median(flows):.3f}")


if __name__ == "__main__":
    main()
