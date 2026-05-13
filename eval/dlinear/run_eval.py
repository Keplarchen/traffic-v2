"""Evaluate DLinear (P95 quantile + conformal calibration) vs baselines."""

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
from core.conformal import fit_correction, apply_correction
from core.dataset import WindowDataset
from core.metrics import inverse_transform
from models.dlinear import DLinear

PT_PATH = ROOT / "data" / "processed" / "all.pt"
CKPT_PATH = ROOT / "checkpoints" / "dlinear.pt"
HERE = Path(__file__).resolve().parent

DISPLAY = {
    "static_peak":            "Static Peak",
    "static_p95":             "Static P95",
    "naive_last":             "Naive Last (resid)",
    "seasonal_naive":         "Seasonal Naive (resid)",
    "dlinear":                "DLinear (P95)",
    "dlinear_conformal":      "DLinear + Conformal",
}


def calculate_comprehensive_metrics(actual_h, alloc_h):
    diff = actual_h - alloc_h
    violations = np.maximum(diff, 0)
    over_prov = np.maximum(-diff, 0)

    sla_pct = 100 * (alloc_h < actual_h).mean()
    util_pct = 100 * actual_h.sum() / max(alloc_h.sum(), 1e-9)
    avg_alloc = alloc_h.mean()
    avg_overprov = over_prov.mean()
    total_unmet = float(violations.sum())

    n_violation_events = (violations > 0).sum()
    avg_violation_size = total_unmet / n_violation_events if n_violation_events > 0 else 0

    pair_sla = 100 * (alloc_h < actual_h).mean(axis=0)  # [462]
    pct_pairs_above_5 = 100 * (pair_sla > 5.0).mean()
    worst_pair_sla = pair_sla.max()

    return [
        sla_pct, util_pct, avg_alloc, avg_overprov,
        pct_pairs_above_5, worst_pair_sla,
        avg_violation_size, total_unmet,
    ]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not PT_PATH.exists():
        return
    raw = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    mean, std = raw["mean"], raw["std"]
    n_train_end = raw["train_idx"][1]
    flows_real = inverse_transform(raw["flows_z"], mean, std).numpy()

    horizons = (1,)
    model = DLinear(seq_len=96, pred_len=len(horizons), channels=462).to(device)
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"Loaded: {CKPT_PATH}")

    model.eval()

    def run_inference(split):
        ds = WindowDataset(PT_PATH, split, horizons=horizons)
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        pz, tz = [], []
        with torch.no_grad():
            for x, y in loader:
                pz.append(model(x.to(device)).cpu())
                tz.append(y)
        return torch.cat(pz), torch.cat(tz), ds

    pred_val_z, target_val_z, _ = run_inference("val")
    pred_z, target_z, test_ds = run_inference("test")

    pred_val_p95 = inverse_transform(pred_val_z, mean, std).numpy()
    actual_val   = inverse_transform(target_val_z, mean, std).numpy()
    pred_p95     = inverse_transform(pred_z, mean, std).numpy()
    actual       = inverse_transform(target_z, mean, std).numpy()
    test_t = test_ds.t_starts.numpy()

    q_hat = fit_correction(pred_val_p95, actual_val, alpha=0.05)
    pred_p95_conformal = apply_correction(pred_p95, q_hat)

    train_real = flows_real[:n_train_end]
    alloc = {
        "static_peak":            static_peak(train_real, len(test_t), horizons),
        "static_p95":             static_p95(train_real, len(test_t), horizons),
        "naive_last":             naive_last_residual_p95(train_real, flows_real, test_t, horizons),
        "seasonal_naive":         seasonal_naive_residual_p95(train_real, flows_real, test_t, horizons),
        "dlinear":                pred_p95,
        "dlinear_conformal":      pred_p95_conformal,
    }

    rows = []
    for h_idx, h in enumerate(horizons):
        print(f"\n=== Horizon h={h} ({h*15} min) Detailed Comparison ===")
        print(f"{'Method':<25} | {'SLA%':>6} | {'Util%':>6} | {'Unmet(M)':>8}")
        print("-" * 60)
        actual_h = actual[:, :, h_idx]
        for k, a_full in alloc.items():
            a_h = a_full[:, :, h_idx]
            metrics = calculate_comprehensive_metrics(actual_h, a_h)

            print(f"{DISPLAY[k]:<25} | {metrics[0]:>6.2f} | {metrics[1]:>6.2f} | {metrics[7]/1e6:>8.2f}M")

            rows.append([h, DISPLAY[k]] + metrics)

    csv_path = HERE / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Horizon_steps", "Horizon_min", "Method",
            "SLA_pct", "Util_pct", "Avg_alloc_Mbps", "Avg_overprov_Mbps",
            "Pct_pairs_SLA_above_5pct", "Worst_pair_SLA_pct",
            "Avg_violation_size_Mbps", "Total_unmet_Mbps",
        ])
        for r in rows:
            w.writerow([
                r[0], r[0]*15, r[1],
                f"{r[2]:.2f}", f"{r[3]:.2f}",
                f"{r[4]:.2f}", f"{r[5]:.2f}",
                f"{r[6]:.2f}", f"{r[7]:.2f}",
                f"{r[8]:.2f}", f"{r[9]:.2f}",
            ])
    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    main()
