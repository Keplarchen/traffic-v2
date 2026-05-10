"""PatchTSTDataset: 纯连续 seq_len 步窗口, channel-independent 友好.

跟 dataset.py 里的 multi-resolution WindowDataset 不同:
  x  [seq_len, 462]    连续 seq_len 步流量, 不混合时间特征
  y  [462, H]          H 个 horizon 各通道的目标

PatchTST 自带 RevIN 处理归一化, 不需要外部时间特征.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class PatchTSTDataset(Dataset):
    def __init__(self, all_pt_path, split: str,
                 seq_len: int = 672, horizons=(16,)):
        d = torch.load(all_pt_path, map_location="cpu", weights_only=False)
        self.flows_z = d["flows_z"]                       # [T, n_channels]
        self.seq_len = seq_len
        self.horizons = torch.tensor(horizons, dtype=torch.long)
        max_h = int(self.horizons.max())

        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split: {split}")
        start, end = d[f"{split}_idx"]
        # t = 输入窗口最后一个时刻 (即"现在")
        # 约束:
        #   - 输入窗口完整: t - seq_len + 1 >= 0  → t >= seq_len - 1
        #   - target 落在 split 内: t + max_h < end  → t <= end - 1 - max_h
        #   - target 不能更早于 split 起点: start <= t + min_h  → t >= start - 1
        t_lo = max(start - 1, seq_len - 1)
        t_hi = end - 1 - max_h
        if t_lo > t_hi:
            raise ValueError(f"split {split} 太短, 无有效样本 "
                             f"(t_lo={t_lo}, t_hi={t_hi})")
        self.t_starts = torch.arange(t_lo, t_hi + 1)

    def __len__(self):
        return len(self.t_starts)

    def __getitem__(self, i):
        t = int(self.t_starts[i])
        x = self.flows_z[t - self.seq_len + 1 : t + 1]   # [seq_len, n_channels]
        # y: 取 H 个未来时刻, transpose 成 [n_channels, H] 跟模型输出对齐
        y = self.flows_z[t + self.horizons].transpose(0, 1).contiguous()
        return x, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    pt_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "all.pt"

    for split in ("train", "val", "test"):
        ds = PatchTSTDataset(pt_path, split, seq_len=672, horizons=(16,))
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
