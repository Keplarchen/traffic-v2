"""TrafficTransformer: 24h 连续上下文 + 多 horizon, 单 P95 分位数输出.

输入  [B, 96, 466]       96 步连续 (= 24h, 每 15min); 462 流量 + 4 时间特征
输出  [B, 462, H, 1]     z-score 空间下 H 个 horizon 的 P95 alloc (直接当带宽分配)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    seq_len: int = 96            # 过去 24 小时, 15-min step
    n_sd_pairs: int = 462
    n_time_feats: int = 4
    d_model: int = 64
    num_layers: int = 2
    nhead: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.2
    n_quantiles: int = 1            # 只输出 P95 (alloc 本身)
    # 预测的未来步数, 单位 = 15 min. 默认 1=15min, 4=1h, 16=4h, 96=24h
    horizons: tuple = (1,)

    @property
    def input_dim(self) -> int:
        return self.n_sd_pairs + self.n_time_feats

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)


class PositionalEncoding(nn.Module):
    """经典 sin/cos 位置编码 (Vaswani 2017)."""

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

        # 1. 把 466 维输入映射到 d_model
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)

        # 2. 位置编码 + Dropout
        self.pos_encoding = PositionalEncoding(cfg.d_model, max_len=cfg.seq_len)
        self.input_dropout = nn.Dropout(cfg.dropout)

        # 3. Transformer 编码器: pre-norm, 训练更稳定; 末尾再加一层 LayerNorm
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, cfg.num_layers, norm=nn.LayerNorm(cfg.d_model),
        )

        # 4. 取"当下"那个 token 做预测: 窗口最后一个 token
        self.current_idx = cfg.seq_len - 1

        # 5. 单层 Linear 解码到 462 SD * H horizon * Q 分位数
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
        # 端到端残差: 所有 horizon 的 baseline 都是当前时刻 z-score 流量
        # 短 horizon 时 delta 很小 ≈ Naive Last; 长 horizon 时 delta 主导
        current_z = x[:, self.current_idx, :self.cfg.n_sd_pairs]   # [B, 462]
        return current_z[:, :, None, None] + delta                 # [B, P, H, Q]


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = TransformerConfig()
    model = TrafficTransformer(cfg)

    x = torch.randn(4, cfg.seq_len, cfg.input_dim)
    out = model(x)
    print(f"Input:    {tuple(x.shape)}")
    print(f"Horizons: {cfg.horizons}  ({cfg.n_horizons} 个, 单位 = 15 min)")
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
