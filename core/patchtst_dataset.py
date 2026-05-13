"""PatchTSTDataset: pure continuous seq_len window for channel-independent input.

Differs from WindowDataset (multi-resolution tokens) in dataset.py:
  x  [seq_len, 462]    continuous seq_len-step traffic, no time features
  y  [462, H]          target traffic per pair per horizon

PatchTST uses RevIN internally for normalization, so no extra time features are needed.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class PatchTSTDataset(Dataset):
    def __init__(self, all_pt_path, split: str,
                 seq_len: int = 96, horizons=(1,)):
        d = torch.load(all_pt_path, map_location="cpu", weights_only=False)
        self.flows_z = d["flows_z"]                       # [T, n_channels]
        self.seq_len = seq_len
        self.horizons = torch.tensor(horizons, dtype=torch.long)
        max_h = int(self.horizons.max())

        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split: {split}")
        start, end = d[f"{split}_idx"]
        # t = last timestep in the input window ("now")
        # Constraints:
        #   - input window fits: t - seq_len + 1 >= 0  -> t >= seq_len - 1
        #   - target stays in split: t + max_h < end   -> t <= end - 1 - max_h
        #   - target not earlier than split start: start <= t + min_h -> t >= start - 1
        t_lo = max(start - 1, seq_len - 1)
        t_hi = end - 1 - max_h
        if t_lo > t_hi:
            raise ValueError(f"split {split} too short, no valid samples "
                             f"(t_lo={t_lo}, t_hi={t_hi})")
        self.t_starts = torch.arange(t_lo, t_hi + 1)

    def __len__(self):
        return len(self.t_starts)

    def __getitem__(self, i):
        t = int(self.t_starts[i])
        x = self.flows_z[t - self.seq_len + 1 : t + 1]   # [seq_len, n_channels]
        # y: H future steps, transposed to [n_channels, H] to align with model output
        y = self.flows_z[t + self.horizons].transpose(0, 1).contiguous()
        return x, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    pt_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "all.pt"

    for split in ("train", "val", "test"):
        ds = PatchTSTDataset(pt_path, split, seq_len=96, horizons=(1,))
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
