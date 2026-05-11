"""在 test 集上评估 single-quantile (P95) PatchTST + 4 个 baseline.

每个 horizon 单独计算指标. 输出:
    results.csv   per-horizon × per-method 数值表
    preds.npz     中间结果 (含 horizons 数组, 各方法 alloc, actual, test_t)
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
from core.conformal import fit_correction, apply_correction
from core.patchtst_dataset import PatchTSTDataset
from core.metrics import inverse_transform
from models.patchtst import PatchTST

PT_PATH = ROOT / "data" / "processed" / "all.pt"
CKPT_PATH = ROOT / "checkpoints" / "patchtst_best.pt"
HERE = Path(__file__).resolve().parent

DISPLAY = {
    "static_peak":         "Static Peak",
    "static_p95":          "Static P95",
    "naive_last":          "Naive Last (resid)",
    "seasonal_naive":      "Seasonal Naive (resid)",
    "patchtst":            "PatchTST (P95)",
    "patchtst_conformal":  "PatchTST + Conformal",
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
    print(f"seq_len={cfg.seq_len}, patch_len={cfg.patch_len}, stride={cfg.stride}")

    model = PatchTST(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    # 3. Inference (bf16 autocast, 跟训练一致). 对 val 和 test 都跑一次:
    #    val 用于拟合 conformal 修正, test 用于评估.
    def run_inference(split):
        ds = PatchTSTDataset(PT_PATH, split,
                             seq_len=cfg.seq_len, horizons=horizons)
        loader = DataLoader(ds, batch_size=64, shuffle=False,
                            pin_memory=(device.type == "cuda"))
        pz, tz = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    p = model(x)
                pz.append(p.float().cpu())
                tz.append(y)
        return torch.cat(pz), torch.cat(tz), ds

    pred_val_z, target_val_z, _ = run_inference("val")
    pred_z, target_z, test_ds = run_inference("test")

    # 反变换到 real Mbps 空间
    pred_val_p95 = inverse_transform(pred_val_z[..., 0], mean, std).numpy()
    actual_val   = inverse_transform(target_val_z,    mean, std).numpy()
    pred_p95     = inverse_transform(pred_z[..., 0], mean, std).numpy()
    actual       = inverse_transform(target_z,    mean, std).numpy()
    test_t = test_ds.t_starts.numpy()

    # 3b. Fit conformal correction on val set (per-pair, per-horizon)
    q_hat = fit_correction(pred_val_p95, actual_val, alpha=0.05)   # [P, H]
    pred_p95_conformal = apply_correction(pred_p95, q_hat)         # [N_test, P, H]
    print(f"\nConformal correction (q_hat):")
    print(f"  shape          = {q_hat.shape}")
    print(f"  mean           = {q_hat.mean():+.2f} Mbps")
    print(f"  range          = [{q_hat.min():+.2f}, {q_hat.max():+.2f}]")
    print(f"  positive pairs = {int((q_hat > 0).sum())}/{q_hat.size} "
          f"(模型偏低估, 需抬高 alloc)")
    print(f"  negative pairs = {int((q_hat < 0).sum())}/{q_hat.size} "
          f"(模型偏高估, 可压低 alloc)")

    # 4. Baselines (每个返回 [N, P, H])
    train_real = flows_real[:n_train_end]
    n_test = len(test_t)
    alloc = {
        "static_peak":         static_peak(train_real, n_test, horizons),
        "static_p95":          static_p95(train_real, n_test, horizons),
        "naive_last":          naive_last_residual_p95(train_real, flows_real, test_t, horizons),
        "seasonal_naive":      seasonal_naive_residual_p95(train_real, flows_real, test_t, horizons),
        "patchtst":            pred_p95,
        "patchtst_conformal":  pred_p95_conformal,
    }

    # 5. Per-horizon 指标 + 打印 + 收集 CSV 行
    rows = []
    target_sla = 0.05
    for h_idx, h in enumerate(horizons):
        print(f"\n=== Horizon h={h}  ({h * 15} min ahead) ===")
        print(f"{'Method':<27} {'SLA%':>6} {'Util%':>6} {'AvgAlloc':>9} "
              f"{'OverProv':>9} {'%Pair>5%':>9} {'WorstPair%':>11} "
              f"{'ViolSize':>9} {'TotUnmet':>10}")
        print("-" * 107)
        actual_h = actual[:, :, h_idx]
        for k, a_full in alloc.items():
            a = a_full[:, :, h_idx]

            # 聚合指标
            sla = 100 * (a < actual_h).mean()
            util = 100 * actual_h.sum() / max(a.sum(), 1e-9)
            avg_alloc = a.mean()
            avg_overprov = np.maximum(a - actual_h, 0).mean()

            # Per-pair SLA 分布: 每对在时间维上的违约率
            per_pair_sla = (a < actual_h).mean(axis=0)             # [P]
            pct_pairs_above_target = 100 * (per_pair_sla > target_sla).mean()
            worst_pair_sla = 100 * per_pair_sla.max()

            # 违约严重度
            diff = actual_h - a                                     # > 0 时为违约
            viol_mask = diff > 0
            avg_viol_size = float(diff[viol_mask].mean()) if viol_mask.any() else 0.0
            total_unmet = float(np.maximum(diff, 0).sum())          # Mbps total

            print(f"{DISPLAY[k]:<27} {sla:>6.2f} {util:>6.2f} "
                  f"{avg_alloc:>9.1f} {avg_overprov:>9.1f} "
                  f"{pct_pairs_above_target:>9.1f} {worst_pair_sla:>11.1f} "
                  f"{avg_viol_size:>9.1f} {total_unmet:>10.0f}")

            rows.append([h, DISPLAY[k], sla, util, avg_alloc, avg_overprov,
                         pct_pairs_above_target, worst_pair_sla,
                         avg_viol_size, total_unmet])
        print(f"  actual mean = {actual_h.mean():.2f} Mbps    "
              f"n_test_steps = {actual_h.shape[0]}    n_pairs = {actual_h.shape[1]}")

    # 6. CSV
    csv_path = HERE / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Horizon_steps", "Horizon_min", "Method",
            "SLA_pct", "Util_pct", "Avg_alloc_Mbps", "Avg_overprov_Mbps",
            "Pct_pairs_SLA_above_5pct", "Worst_pair_SLA_pct",
            "Avg_violation_size_Mbps", "Total_unmet_Mbps",
        ])
        for row in rows:
            w.writerow([
                row[0], row[0]*15, row[1],
                f"{row[2]:.2f}", f"{row[3]:.2f}",
                f"{row[4]:.2f}", f"{row[5]:.2f}",
                f"{row[6]:.2f}", f"{row[7]:.2f}",
                f"{row[8]:.2f}", f"{row[9]:.2f}",
            ])
    print(f"\nSaved {csv_path}")

    # 7. 中间结果给 plots.py
    npz_path = HERE / "preds.npz"
    np.savez(
        npz_path,
        actual=actual,
        test_t=test_t,
        horizons=np.array(horizons),
        **{f"alloc_{k}": v for k, v in alloc.items()},
    )
    print(f"Saved {npz_path}")


if __name__ == "__main__":
    main()
