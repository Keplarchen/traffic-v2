"""DLinear: linear time-series forecaster with trend/seasonal decomposition.

Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?"
AAAI 2023. https://arxiv.org/abs/2205.13504

We add a small time-feature projection branch on top of the original seasonal
and trend linear heads, since our WindowDataset input includes 4 sin/cos time
features alongside the 462 flow channels.
"""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving-average filter via AvgPool1d, used for trend extraction."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, P]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """Decompose a series into trend (moving average) and residual (seasonal)."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend


class DLinear(nn.Module):
    """DLinear with trend/seasonal decomposition plus a time-feature branch.

    Input  x: [B, L, 462 + 4]    flows on the first 462 channels, 4 time features after
    Output:   [B, 462, pred_len]
    """

    def __init__(self, seq_len, pred_len, channels=462):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.decomp = SeriesDecomp(kernel_size=25)

        # Channel-wise linear via grouped conv (one filter per channel)
        self.Linear_Seasonal = nn.Conv1d(channels, channels * pred_len,
                                         kernel_size=seq_len, groups=channels)
        self.Linear_Trend = nn.Conv1d(channels, channels * pred_len,
                                      kernel_size=seq_len, groups=channels)
        # Time-feature branch: project flattened sin/cos features to per-pair bias
        self.time_projection = nn.Linear(4 * seq_len, channels * pred_len)

    def forward(self, x):
        batch_size = x.shape[0]
        flows = x[:, :, :self.channels]                                  # [B, L, 462]
        times = x[:, :, self.channels:].reshape(batch_size, -1)          # [B, 4*L]

        seasonal_init, trend_init = self.decomp(flows)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init).view(
            batch_size, self.channels, self.pred_len)
        trend_output = self.Linear_Trend(trend_init).view(
            batch_size, self.channels, self.pred_len)
        time_bias = self.time_projection(times).view(
            batch_size, self.channels, self.pred_len)

        return seasonal_output + trend_output + time_bias
