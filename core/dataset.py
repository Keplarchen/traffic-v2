"""WindowDataset: 把 all.pt 里的连续序列切成 (x, y) 训练样本.

每条样本:
  x  [107, 466]      recent(96) + daily(7) + weekly(4) tokens; 462 流量 + 4 时间特征
  y  [462, H]        462 个 SD 对各 H 个 horizon 的目标流量 (z-score 空间)
                     默认 horizons=(1, 4, 16) → 15min, 1h, 4h
                     维度顺序 (P, H) 跟模型输出 (P, H, Q) 对齐, 损失计算更清晰
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    SEQ_RECENT = 96
    SEQ_DAILY = 7
    SEQ_WEEKLY = 4
    DAY_STEPS = 96
    WEEK_STEPS = 7 * 96   # 672

    @classmethod
    def min_t(cls) -> int:
        # 最早可预测的 t: 需要 4 周前 (t+1 - 4*672) >= 0
        return cls.SEQ_WEEKLY * cls.WEEK_STEPS - 1   # 2687

    def __init__(self, all_pt_path, split: str, horizons=(1, 4, 16)):
        d = torch.load(all_pt_path, map_location="cpu", weights_only=False)
        self.flows_z = d["flows_z"]              # [T, 462]
        self.time_feats = d["time_feats"]        # [T, 4]
        self.horizons = torch.tensor(horizons, dtype=torch.long)
        max_h = int(self.horizons.max())

        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split: {split}")
        start, end = d[f"{split}_idx"]
        # target y[h] = flows_z[t+h]; 所有 h 必须落在 split 内
        # → t ∈ [start-1, end-1-max_h], 同时全局 t >= min_t
        t_lo = max(start - 1, self.min_t())
        t_hi = end - 1 - max_h
        if t_lo > t_hi:
            raise ValueError(f"split {split} 太短, 无有效样本")
        self.t_starts = torch.arange(t_lo, t_hi + 1)

        # 预计算 107 个 token 相对 t 的偏移
        # recent: t-95 .. t  (含 t)
        rel_recent = torch.arange(-self.SEQ_RECENT + 1, 1)
        # daily k=1..7: 位置 = t+1 - k*96  → 偏移 = 1 - k*96
        rel_daily = torch.tensor(
            [1 - k * self.DAY_STEPS for k in range(1, self.SEQ_DAILY + 1)]
        )
        # weekly k=1..4: 位置 = t+1 - k*672 → 偏移 = 1 - k*672
        rel_weekly = torch.tensor(
            [1 - k * self.WEEK_STEPS for k in range(1, self.SEQ_WEEKLY + 1)]
        )
        self.rel_offsets = torch.cat([rel_recent, rel_daily, rel_weekly])   # [107]

    def __len__(self):
        return len(self.t_starts)

    def __getitem__(self, i):
        t = int(self.t_starts[i])
        idx = self.rel_offsets + t                   # [107]
        flows = self.flows_z[idx]                    # [107, 462]
        times = self.time_feats[idx]                 # [107, 4]
        x = torch.cat([flows, times], dim=-1)        # [107, 466]
        # 取 H 个未来时刻, 然后把维度从 (H, P) 转成 (P, H) 跟模型输出对齐
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
