"""PatchTST wrapper: 调用官方 PatchTST_backbone, 适配我们的 (B, T, C) 输入和量化输出.

官方源码: https://github.com/yuqinie98/PatchTST  (ICLR 2023)
   下载到 traffic-v2/models/patchtst_official/ (PatchTST_backbone.py + PatchTST_layers.py + RevIN.py)

本文件做的事 (薄薄一层):
  1. 把 [B, seq_len, n_channels] 转成官方期望的 [B, n_channels, seq_len]
  2. target_window = n_horizons * n_quantiles
  3. 输出 reshape 成 [B, n_channels, H, Q] 跟 pinball_loss / metrics 对齐

所有架构细节 (BatchNorm, post-norm, res_attention, learnable PE, RevIN-affine)
全部用官方默认值 → 跟论文实现完全一致.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .patchtst_official.PatchTST_backbone import PatchTST_backbone


@dataclass
class PatchTSTConfig:
    seq_len: int = 672          # 7 天历史 (15-min step)
    n_channels: int = 462       # SD 对数
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    num_layers: int = 3         # 论文默认
    nhead: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.2
    n_quantiles: int = 3
    horizons: tuple = (16,)     # 单 horizon: 4h ahead

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)


class PatchTST(nn.Module):
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        target_window = cfg.n_horizons * cfg.n_quantiles
        # 全部用官方默认: revin=True, affine=True, norm='BatchNorm',
        #   pre_norm=False, res_attention=True, pe='zeros', learn_pe=True,
        #   head_type='flatten', individual=False, act='gelu'
        self.backbone = PatchTST_backbone(
            c_in=cfg.n_channels,
            context_window=cfg.seq_len,
            target_window=target_window,
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            n_layers=cfg.num_layers,
            d_model=cfg.d_model,
            n_heads=cfg.nhead,
            d_ff=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, n_channels]   PatchTSTDataset 输出格式
        Returns:
            [B, n_channels, n_horizons, n_quantiles]
        """
        B = x.size(0)
        # 官方 backbone 期望 [B, n_channels, seq_len]
        z = x.permute(0, 2, 1)
        # backbone 输出 [B, n_channels, target_window=H*Q]
        out = self.backbone(z)
        # reshape 成 [B, n_channels, H, Q]
        return out.view(B, self.cfg.n_channels,
                        self.cfg.n_horizons, self.cfg.n_quantiles)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = PatchTSTConfig()
    model = PatchTST(cfg)

    x = torch.randn(4, cfg.seq_len, cfg.n_channels)
    out = model(x)
    print(f"Input:    {tuple(x.shape)}    [B, seq_len, n_channels]")
    print(f"Horizons: {cfg.horizons}    n_horizons={cfg.n_horizons}, "
          f"n_quantiles={cfg.n_quantiles}")
    print(f"Output:   {tuple(out.shape)}    [B, n_channels, H, Q]")
    assert out.shape == (4, cfg.n_channels, cfg.n_horizons, cfg.n_quantiles)

    out.mean().backward()
    print("Backward OK")

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {n_total:,}")
