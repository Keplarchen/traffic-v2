"""训练 PatchTST.

用法: python train_patchtst.py [--epochs 200] [--batch-size 16] ...
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.patchtst_dataset import PatchTSTDataset
from core.losses import pinball_loss
from core.metrics import compute_metrics
from models.patchtst import PatchTST, PatchTSTConfig


def get_lr(step, total_steps, warmup_steps, base_lr, min_ratio=0.01):
    """linear warmup + cosine decay 到 base_lr * min_ratio."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1 - min_ratio) * cosine)


@torch.no_grad()
def evaluate(model, loader, mean, std, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds.append(model(x).cpu())
        targets.append(y.cpu())
    pred_z = torch.cat(preds)
    target_z = torch.cat(targets)
    val_loss = pinball_loss(pred_z, target_z).item()
    metrics = compute_metrics(pred_z, target_z, mean, std, q_idx=2)
    return val_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parent
    pt_path = project_root / "data" / "processed" / "all.pt"
    if not pt_path.exists():
        sys.exit(f"未找到 {pt_path}, 请先跑 prep.py")

    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    mean, std = raw["mean"], raw["std"]

    cfg = PatchTSTConfig()
    train_ds = PatchTSTDataset(pt_path, "train",
                               seq_len=cfg.seq_len, horizons=cfg.horizons)
    val_ds = PatchTSTDataset(pt_path, "val",
                             seq_len=cfg.seq_len, horizons=cfg.horizons)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    model = PatchTST(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Device: {device}")
    print(f"Horizons: {cfg.horizons}  (1 step = 15 min)")
    print(f"seq_len={cfg.seq_len}, patch_len={cfg.patch_len}, stride={cfg.stride}")
    print(f"Params: {n_params:,}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)

    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "patchtst_best.pt"

    best_val = float("inf")
    patience = 0
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = get_lr(step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            pred = model(x)
            loss = pinball_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss_sum += loss.item()
            step += 1

        train_loss = train_loss_sum / len(train_loader)
        val_loss, vm = evaluate(model, val_loader, mean, std, device)
        dt = time.time() - t0

        print(f"E{epoch:3d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| sla {vm['sla_violation_rate']:.4f} | util {vm['utilization']:.3f} "
              f"| mae {vm['mae_p50']:.1f} Mbps | lr {lr:.2e} | {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_metrics": vm,
                "config": cfg,
            }, ckpt_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (val 已 {args.patience} epoch 没改善)")
                break

    print(f"\n最佳 val loss: {best_val:.4f}, checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
