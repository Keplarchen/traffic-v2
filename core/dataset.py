"""WindowDataset: builds (x, y) training samples from the all.pt continuous series.

Each sample:
  x  [seq_len, 466]  continuous past window; 462 flows + 4 time features
  y  [462, H]        target flow per SD pair for each of H horizons (z-score space)
                     Default horizons=(1, 4, 16) -> 15min, 1h, 4h
                     Dimension order (P, H) aligns with model output (P, H, Q).
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    SEQ_LEN = 96   # past 24 hours, one token per 15 minutes

    @classmethod
    def min_t(cls) -> int:
        # Earliest valid t requires SEQ_LEN history, i.e. t - SEQ_LEN + 1 >= 0
        return cls.SEQ_LEN - 1

    def __init__(self, all_pt_path, split: str, horizons=(1, 4, 16)):
        d = torch.load(all_pt_path, map_location="cpu", weights_only=False)
        self.flows_z = d["flows_z"]              # [T, 462]
        self.time_feats = d["time_feats"]        # [T, 4]
        self.horizons = torch.tensor(horizons, dtype=torch.long)
        max_h = int(self.horizons.max())

        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split: {split}")
        start, end = d[f"{split}_idx"]
        # Target y[h] = flows_z[t+h]; every h must fall within the split.
        # => t in [start-1, end-1-max_h], and globally t >= min_t.
        t_lo = max(start - 1, self.min_t())
        t_hi = end - 1 - max_h
        if t_lo > t_hi:
            raise ValueError(f"split {split} too short, no valid samples")
        self.t_starts = torch.arange(t_lo, t_hi + 1)

        # Continuous window: t-SEQ_LEN+1 .. t (inclusive)
        self.rel_offsets = torch.arange(-self.SEQ_LEN + 1, 1)   # [SEQ_LEN]

    def __len__(self):
        return len(self.t_starts)

    def __getitem__(self, i):
        t = int(self.t_starts[i])
        idx = self.rel_offsets + t                   # [SEQ_LEN]
        flows = self.flows_z[idx]                    # [SEQ_LEN, 462]
        times = self.time_feats[idx]                 # [SEQ_LEN, 4]
        x = torch.cat([flows, times], dim=-1)        # [SEQ_LEN, 466]
        # Take H future steps and transpose to (P, H) to align with model output
        y = self.flows_z[t + self.horizons].transpose(0, 1).contiguous()  # [462, H]
        return x, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    pt_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "all.pt"

    for split in ("train", "val", "test"):
        ds = WindowDataset(pt_path, split, horizons=(1, 4, 16))
        x, y = ds[0]
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        bx, by = next(iter(loader))
        print(f"{split:5s}: n={len(ds):>5}, "
              f"t in [{int(ds.t_starts[0])}, {int(ds.t_starts[-1])}], "
              f"horizons={ds.horizons.tolist()}")
        print(f"        sample: x={tuple(x.shape)}, y={tuple(y.shape)}")
        print(f"        batch:  x={tuple(bx.shape)}, y={tuple(by.shape)}")
        print(f"        sanity: x finite? {torch.isfinite(bx).all().item()}, "
              f"y finite? {torch.isfinite(by).all().item()}")
