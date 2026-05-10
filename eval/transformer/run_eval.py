"""在 test 集上评估 multi-horizon Transformer + 4 个 baseline.

每个 horizon 单独计算指标. 输出:
    results.csv   per-horizon × per-method 数值表
    preds.npz     中间结果 (含 horizons 数组, 各方法 alloc, P50/P90/P95 等)
    控制台          per-horizon 对比表
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.baselines import (
    static_peak, static_p95,
    naive_last_residual_p95, seasonal_naive_residual_p95,
)
from core.dataset import WindowDataset
from core.metrics import inverse_transform
from models.transformer import TrafficTransformer

PT_PATH = ROOT / "data" / "processed" / "all.pt"
CKPT_PATH = ROOT / "checkpoints" / "transformer_best.pt"
HERE = Path(__file__).resolve().parent

DISPLAY = {
    "static_peak":     "Static Peak",
    "static_p95":      "Static P95",
    "naive_last":      "Naive Last (resid)",
    "seasonal_naive":  "Seasonal Naive (resid)",
    "transformer":     "Transformer (P95)",
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    raw = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    flows_z = raw["flows_z"]
    mean, std = raw["mean"], raw["std"]
    n_train_end = raw["train_idx"][1]
    flows_real = inverse_transform(flows_z, mean, std).numpy()         # [T, P]

    # 2. 加载 checkpoint, 拿 horizons
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    horizons = tuple(cfg.horizons)
    H = len(horizons)
    print(f"Loaded ckpt: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print(f"Horizons: {horizons}  (1 step = 15 min)")

    model = TrafficTransformer(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    # 3. Inference
    test_ds = WindowDataset(PT_PATH, "test", horizons=horizons)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                        pin_memory=(device.type == "cuda"))
    pz, tz = [], []
    with torch.no_grad():
        for x, y in loader:
            pz.append(model(x.to(device, non_blocking=True)).cpu())
            tz.append(y)
    pred_z = torch.cat(pz)             # [N, P, H, Q]
    target_z = torch.cat(tz)           # [N, P, H]

    pred_p50 = inverse_transform(pred_z[..., 0], mean, std).numpy()    # [N, P, H]
    pred_p90 = inverse_transform(pred_z[..., 1], mean, std).numpy()
    pred_p95 = inverse_transform(pred_z[..., 2], mean, std).numpy()
    actual   = inverse_transform(target_z,    mean, std).numpy()
    test_t = test_ds.t_starts.numpy()

    # 4. Baselines (每个返回 [N, P, H])
    train_real = flows_real[:n_train_end]
    n_test = len(test_t)
    alloc = {
        "static_peak":     static_peak(train_real, n_test, horizons),
        "static_p95":      static_p95(train_real, n_test, horizons),
        "naive_last":      naive_last_residual_p95(train_real, flows_real, test_t, horizons),
        "seasonal_naive":  seasonal_naive_residual_p95(train_real, flows_real, test_t, horizons),
        "transformer":     pred_p95,
    }

    # 5. Per-horizon 指标 + 打印 + 收集 CSV 行
    rows = []
    for h_idx, h in enumerate(horizons):
        print(f"\n=== Horizon h={h}  ({h * 15} min ahead) ===")
        print(f"{'Method':<25} {'SLA %':>7} {'Util %':>7} {'AvgAlloc':>10} {'OverProv':>10}")
        print("-" * 62)
        actual_h = actual[:, :, h_idx]
        for k, a_full in alloc.items():
            a = a_full[:, :, h_idx]
            sla = 100 * (a < actual_h).mean()
            util = 100 * actual_h.sum() / max(a.sum(), 1e-9)
            m = {
                "sla_pct": sla,
                "util_pct": util,
                "avg_alloc": a.mean(),
                "avg_overprov": np.maximum(a - actual_h, 0).mean(),
            }
            print(f"{DISPLAY[k]:<25} {m['sla_pct']:>7.2f} {m['util_pct']:>7.2f} "
                  f"{m['avg_alloc']:>10.1f} {m['avg_overprov']:>10.1f}")
            rows.append([h, DISPLAY[k], m['sla_pct'], m['util_pct'],
                         m['avg_alloc'], m['avg_overprov']])
        mae = float(np.abs(pred_p50[:, :, h_idx] - actual_h).mean())
        print(f"  Transformer MAE_P50 = {mae:.2f} Mbps    "
              f"actual mean = {actual_h.mean():.2f}")

    # 6. CSV
    csv_path = HERE / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Horizon_steps", "Horizon_min", "Method",
                    "SLA_pct", "Util_pct", "Avg_alloc_Mbps", "Avg_overprov_Mbps"])
        for row in rows:
            w.writerow([row[0], row[0]*15, row[1],
                        f"{row[2]:.2f}", f"{row[3]:.2f}",
                        f"{row[4]:.2f}", f"{row[5]:.2f}"])
    print(f"\nSaved {csv_path}")

    # 7. 中间结果给 plots.py
    npz_path = HERE / "preds.npz"
    np.savez(
        npz_path,
        actual=actual, pred_p50=pred_p50, pred_p90=pred_p90, pred_p95=pred_p95,
        test_t=test_t,
        horizons=np.array(horizons),
        **{f"alloc_{k}": v for k, v in alloc.items()},
    )
    print(f"Saved {npz_path}")


if __name__ == "__main__":
    main()
