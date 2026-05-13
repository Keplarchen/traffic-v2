"""ResidualLSTM: LSTM-based residual predictor with per-flow alpha calibration.

Architecture:
  - LSTM encoder reads the past `seq_len` timesteps of flows + 4 time features.
  - A 2-layer MLP head predicts a residual to be added to the current flow.
  - A per-flow alpha buffer (alpha in [0, 1]) shrinks the residual per pair:
        pred = current + alpha * residual
    alpha = 0 falls back to Naive Last; alpha = 1 trusts the LSTM fully.

The alpha buffer is initialized to 1 (full LSTM) and is fit via grid search on
the validation set (per-pair MAE) by calling `calibrate_alpha()` after training.
Once calibrated, alpha is part of the model state and is applied automatically
inside `forward`, making the calibrated predictor end-to-end.

This is a regression model (point prediction). Conformal calibration is applied
post-hoc at evaluation time to convert the point prediction into a bandwidth
allocation that meets the target SLA violation rate.
"""

import torch
import torch.nn as nn


class ResidualLSTM(nn.Module):
    """LSTM with explicit residual head plus per-flow alpha shrinkage."""

    def __init__(self, input_size, n_flows, hidden_size=192, num_layers=2, dropout=0.2):
        super().__init__()
        self.n_flows = n_flows
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_flows),
        )
        # Per-flow shrinkage factor on the LSTM residual. Default = 1.0
        # (full LSTM). Set by calibrate_alpha() after training.
        self.register_buffer("alpha", torch.ones(n_flows))

    def _lstm_residual(self, x):
        """Run the LSTM stack and return (current_z, raw_residual_z) without alpha."""
        current = x[:, -1, :self.n_flows]      # [B, n_flows]
        out, _ = self.lstm(x)
        residual = self.head(out[:, -1])       # [B, n_flows]
        return current, residual

    def forward(self, x):
        """
        x: [B, L, input_size]   first n_flows channels are flows, rest are time features
        Output: [B, n_flows, 1]   alpha-calibrated point prediction at t+1
        """
        current, residual = self._lstm_residual(x)
        # Per-flow shrinkage: pred = current + alpha * residual
        calibrated = current + self.alpha.unsqueeze(0) * residual
        return calibrated.unsqueeze(-1)

    @torch.no_grad()
    def calibrate_alpha(self, val_loader, device, grid_size=21):
        """Fit per-flow alpha on the validation set by grid search (z-space MAE).

        For each SD pair independently, search alpha in [0, 1] for the value
        that minimizes val-set MAE of (current + alpha * residual) against the
        target. Updates `self.alpha` in place.

        Returns:
            best_alpha: numpy array [n_flows], the chosen alpha per flow
            best_mae:   numpy array [n_flows], the corresponding val MAE
        """
        self.eval()
        all_current, all_residual, all_target = [], [], []
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            current, residual = self._lstm_residual(x)
            all_current.append(current.cpu())
            all_residual.append(residual.cpu())
            all_target.append(y[:, :, 0].cpu())                # [B, n_flows]

        current_z = torch.cat(all_current, dim=0)              # [N, n_flows]
        residual_z = torch.cat(all_residual, dim=0)
        target_z = torch.cat(all_target, dim=0)

        alphas = torch.linspace(0.0, 1.0, grid_size)
        best_alpha = torch.ones(self.n_flows)
        best_mae = torch.full((self.n_flows,), float("inf"))

        for a in alphas:
            pred_z = current_z + a * residual_z
            mae = (pred_z - target_z).abs().mean(dim=0)        # [n_flows]
            better = mae < best_mae
            best_mae[better] = mae[better]
            best_alpha[better] = a

        self.alpha.copy_(best_alpha.to(self.alpha.device))
        return best_alpha.numpy(), best_mae.numpy()
