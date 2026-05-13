"""PatchTST wrapper: calls the official PatchTST_backbone and adapts our (B, T, C) input
and quantile output.

Official source: https://github.com/yuqinie98/PatchTST  (ICLR 2023)
   Downloaded into traffic-v2/models/patchtst_official/
   (PatchTST_backbone.py + PatchTST_layers.py + RevIN.py)

This wrapper does:
  1. Transpose [B, seq_len, n_channels] to the backbone's expected [B, n_channels, seq_len].
  2. Set target_window = n_horizons * n_quantiles.
  3. Reshape output to [B, n_channels, H, Q] to align with pinball_loss / metrics.

All architectural details (BatchNorm, post-norm, res_attention, learnable PE, RevIN-affine)
use the official defaults, matching the paper's implementation.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .patchtst_official.PatchTST_backbone import PatchTST_backbone


@dataclass
class PatchTSTConfig:
    seq_len: int = 96           # past 24 hours at 15-min granularity
    n_channels: int = 462       # number of SD pairs
    patch_len: int = 8          # each patch covers 2h (8 * 15 min)
    stride: int = 4             # half-overlap stride
    d_model: int = 128
    num_layers: int = 3         # paper default
    nhead: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.2
    n_quantiles: int = 1        # P95 only (the allocation itself)
    horizons: tuple = (1,)      # single horizon: 15 min ahead

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)


class PatchTST(nn.Module):
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        target_window = cfg.n_horizons * cfg.n_quantiles
        # Use official defaults: revin=True, affine=True, norm='BatchNorm',
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
            x: [B, seq_len, n_channels]   PatchTSTDataset output format
        Returns:
            [B, n_channels, n_horizons, n_quantiles]
        """
        B = x.size(0)
        # Official backbone expects [B, n_channels, seq_len]
        z = x.permute(0, 2, 1)
        # Backbone outputs [B, n_channels, target_window = H*Q]
        out = self.backbone(z)
        # Reshape to [B, n_channels, H, Q]
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
