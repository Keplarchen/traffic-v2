"""TrafficTransformer: 24h continuous context + multi-horizon, single P95 output.

Input:  [B, 96, 466]      96 continuous steps (= 24h at 15 min/step); 462 flows + 4 time features
Output: [B, 462, H, 1]    H-horizon P95 allocations in z-score space, used directly as bandwidth.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    seq_len: int = 96            # past 24 hours at 15-min granularity
    n_sd_pairs: int = 462
    n_time_feats: int = 4
    d_model: int = 64
    num_layers: int = 2
    nhead: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.2
    n_quantiles: int = 1            # P95 only (the allocation itself)
    # Forecast horizons in 15-min steps. 1=15min, 4=1h, 16=4h, 96=24h
    horizons: tuple = (1,)

    @property
    def input_dim(self) -> int:
        return self.n_sd_pairs + self.n_time_feats

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)


class PositionalEncoding(nn.Module):
    """Classic sin/cos positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TrafficTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection: 466 -> d_model
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)

        # Positional encoding + dropout
        self.pos_encoding = PositionalEncoding(cfg.d_model, max_len=cfg.seq_len)
        self.input_dropout = nn.Dropout(cfg.dropout)

        # Transformer encoder: pre-norm for training stability, plus a final LayerNorm
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, cfg.num_layers, norm=nn.LayerNorm(cfg.d_model),
        )

        # We predict from the "current" token (last step of the window)
        self.current_idx = cfg.seq_len - 1

        # Output projection: d_model -> 462 * H * Q
        self.decoder = nn.Linear(
            cfg.d_model,
            cfg.n_sd_pairs * cfg.n_horizons * cfg.n_quantiles,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h = self.input_proj(x)
        h = self.input_dropout(self.pos_encoding(h))
        h = self.encoder(h)
        delta = self.decoder(h[:, self.current_idx, :])     # [B, P*H*Q]
        delta = delta.view(B, self.cfg.n_sd_pairs,
                           self.cfg.n_horizons, self.cfg.n_quantiles)
        # End-to-end residual: every horizon's baseline is the current z-score flow.
        # At short horizons delta is small -> output ~ Naive Last;
        # at long horizons delta dominates.
        current_z = x[:, self.current_idx, :self.cfg.n_sd_pairs]   # [B, 462]
        return current_z[:, :, None, None] + delta                 # [B, P, H, Q]


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = TransformerConfig()
    model = TrafficTransformer(cfg)

    x = torch.randn(4, cfg.seq_len, cfg.input_dim)
    out = model(x)
    print(f"Input:    {tuple(x.shape)}")
    print(f"Horizons: {cfg.horizons}  ({cfg.n_horizons} horizons, step = 15 min)")
    print(f"Output:   {tuple(out.shape)}    [B, n_sd_pairs, n_horizons, n_quantiles]")
    assert out.shape == (4, cfg.n_sd_pairs, cfg.n_horizons, cfg.n_quantiles)

    out.mean().backward()
    print("Backward OK")

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {n_total:,}")
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        if n:
            print(f"  {name:18s} {n:>9,}")
