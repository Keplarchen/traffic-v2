"""Train ResidualLSTM with SmoothL1 (Huber) loss, then calibrate per-flow alpha.

Usage: python train_lstm.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dataset import WindowDataset
from models.lstm import ResidualLSTM

N_FLOWS = 462
N_TIME_FEATS = 4
HIDDEN_SIZE = 192
NUM_LAYERS = 2
DROPOUT = 0.2

LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 12
GRAD_CLIP = 1.0
SEED = 42
CALIBRATION_GRID_SIZE = 21


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)                         # [B, 462, 1]
        loss = loss_fn(pred, y)
        total += loss.item() * len(x)
        n += len(x)
    return total / max(n, 1)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_pt_path = ROOT / "data" / "processed" / "all.pt"
    if not all_pt_path.exists():
        sys.exit(f"Missing {all_pt_path}; please run prep.py first")

    horizons = (1,)
    train_ds = WindowDataset(all_pt_path, "train", horizons=horizons)
    val_ds = WindowDataset(all_pt_path, "val", horizons=horizons)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    model = ResidualLSTM(
        input_size=N_FLOWS + N_TIME_FEATS,
        n_flows=N_FLOWS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()

    print(f"Device: {device}")
    print(f"Horizons: {horizons}  (1 step = 15 min)")
    print(f"Params: {n_params:,}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print()

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "lstm.pt"

    best_val = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        train_sum, n_seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_sum += loss.item() * len(x)
            n_seen += len(x)

        train_loss = train_sum / max(n_seen, 1)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        dt = time.time() - t0

        print(f"E{epoch:3d}/{EPOCHS} | train {train_loss:.6f} | val {val_loss:.6f} | {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (val has not improved for {PATIENCE} epochs)")
                break

    # Restore best model state for calibration
    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibrate per-flow alpha on validation set
    print("\nCalibrating per-flow alpha on validation set...")
    best_alpha, best_alpha_mae = model.calibrate_alpha(val_loader, device,
                                                      grid_size=CALIBRATION_GRID_SIZE)
    print(f"  alpha mean = {best_alpha.mean():.4f}")
    print(f"  flows fully trusting LSTM (alpha = 1): "
          f"{int((best_alpha == 1.0).sum())}/{len(best_alpha)}")
    print(f"  flows fully discarding LSTM (alpha = 0): "
          f"{int((best_alpha == 0.0).sum())}/{len(best_alpha)}")

    # Save calibrated model
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val,
        "config": {
            "input_size": N_FLOWS + N_TIME_FEATS,
            "n_flows": N_FLOWS,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
    }, ckpt_path)
    print(f"\nBest val loss: {best_val:.6f}, checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
