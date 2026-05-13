"""Train DLinear.

Usage: python train_dlinear.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dataset import WindowDataset
from core.losses import pinball_loss
from core.metrics import inverse_transform
from models.dlinear import DLinear


def calc_metrics(y_true, y_pred):
    """Simple Mbps-space metrics used for the end-of-training print."""
    y_pred = np.maximum(0, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    sla_drop = np.sum(np.maximum(0, y_true - y_pred)) / (np.sum(y_true) + 1e-6) * 100
    utilization = np.sum(y_true) / (np.sum(y_pred) + 1e-6) * 100
    return mae, sla_drop, utilization


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_pt_path = ROOT / "data" / "processed" / "all.pt"

    meta = torch.load(all_pt_path, map_location="cpu", weights_only=False)
    mean, std = meta["mean"].to(device), meta["std"].to(device)

    horizons = (1,)
    train_ds = WindowDataset(all_pt_path, "train", horizons=horizons)
    test_ds = WindowDataset(all_pt_path, "test", horizons=horizons)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = DLinear(seq_len=96, pred_len=1, channels=462).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)                                        # [B, 462, H]
            # core.losses.pinball_loss expects [..., Q]; unsqueeze for single-quantile
            loss = pinball_loss(out.unsqueeze(-1), by, taus=(0.95,))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f}")

    # Test evaluation in real-Mbps space
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred_z = model(bx)
            all_preds.append(inverse_transform(pred_z, mean, std).cpu().numpy())
            all_trues.append(inverse_transform(by, mean, std).cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)

    mae, sla, util = calc_metrics(y_true, y_pred)
    print()
    print("=" * 60)
    print(f"{'Method':<20} | {'MAE':<10} | {'SLA Drop %':<12} | {'Util %':<10}")
    print("-" * 60)
    print(f"{'DLinear (q=0.95)':<20} | {mae:<10.2f} | {sla:<12.2f} | {util:<10.2f}")

    # Save checkpoint
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "dlinear.pt")


if __name__ == "__main__":
    main()
