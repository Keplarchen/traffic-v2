"""Data preparation: time features + log1p + StandardScaler (fit on train only) + split.

Reads:   data/flows_clean.npy, data/timestamps.npy
Writes:  data/processed/all.pt  (single file containing all transformed data and split indices)
"""

from pathlib import Path

import numpy as np
import torch


def compute_time_features(ts: np.ndarray) -> np.ndarray:
    """[T] datetime64[m] -> [T, 4] (sin/cos hour, sin/cos dow), float32."""
    ts_int = ts.astype(np.int64)            # minutes since epoch
    hour = (ts_int % (24 * 60)) / 60.0      # [0, 24)
    # Epoch (1970-01-01) is Thursday = weekday 3, so add 3 before mod 7
    dow = ((ts_int // (24 * 60)) + 3) % 7   # 0=Mon..6=Sun
    feats = np.empty((len(ts), 4), dtype=np.float32)
    feats[:, 0] = np.sin(2 * np.pi * hour / 24)
    feats[:, 1] = np.cos(2 * np.pi * hour / 24)
    feats[:, 2] = np.sin(2 * np.pi * dow / 7)
    feats[:, 3] = np.cos(2 * np.pi * dow / 7)
    return feats


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir = data_dir / "processed"
    out_dir.mkdir(exist_ok=True)

    flows = np.load(data_dir / "flows_clean.npy")     # [T, 462] Mbps
    ts = np.load(data_dir / "timestamps.npy")         # [T]
    T, P = flows.shape
    print(f"Loaded flows_clean: shape={flows.shape}")

    time_feats = compute_time_features(ts)
    print(f"Time features: shape={time_feats.shape}, "
          f"range=[{time_feats.min():.3f}, {time_feats.max():.3f}]")

    # 80 / 10 / 10 chronological split
    n_train = int(T * 0.8)
    n_val = int(T * 0.1)
    train_idx = (0, n_train)
    val_idx = (n_train, n_train + n_val)
    test_idx = (n_train + n_val, T)
    print(f"Split: train [{train_idx[0]:>5}, {train_idx[1]:>5})  n={train_idx[1]-train_idx[0]}")
    print(f"       val   [{val_idx[0]:>5}, {val_idx[1]:>5})  n={val_idx[1]-val_idx[0]}")
    print(f"       test  [{test_idx[0]:>5}, {test_idx[1]:>5})  n={test_idx[1]-test_idx[0]}")
    print(f"       time window: train {ts[train_idx[0]]} .. {ts[train_idx[1]-1]}")
    print(f"                    test  {ts[test_idx[0]]} .. {ts[test_idx[1]-1]}")

    # log1p tames the heavy-tailed flow distribution
    flows_log = np.log1p(flows)

    # Fit scaler on train only to avoid leakage
    train_log = flows_log[train_idx[0]:train_idx[1]]
    mean = train_log.mean(axis=0).astype(np.float32)
    std = train_log.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0   # guard against division by zero for constant columns

    # Transform full series with train-fitted scaler
    flows_z = ((flows_log - mean) / std).astype(np.float32)
    print(f"\nScaler (log1p space): mean range [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"                      std  range [{std.min():.3f}, {std.max():.3f}]")
    print(f"flows_z: train mean={flows_z[train_idx[0]:train_idx[1]].mean():.3f}, "
          f"std={flows_z[train_idx[0]:train_idx[1]].std():.3f}")

    out = {
        "flows_z":    torch.from_numpy(flows_z),
        "time_feats": torch.from_numpy(time_feats),
        "mean":       torch.from_numpy(mean),
        "std":        torch.from_numpy(std),
        "train_idx":  train_idx,
        "val_idx":    val_idx,
        "test_idx":   test_idx,
    }
    save_path = out_dir / "all.pt"
    torch.save(out, save_path)
    print(f"\nSaved: {save_path}  (~{save_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
