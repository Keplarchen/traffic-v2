"""6 张评估图. 读 preds.npz, 输出到 figures/.

每个 plot_xxx 独立可调用. 直接 python plots.py 跑全部.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NPZ_PATH = HERE / "preds.npz"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

DISPLAY = {
    "static_peak":     "Static Peak",
    "static_p95":      "Static P95",
    "naive_last":      "Naive Last (resid)",
    "seasonal_naive":  "Seasonal Naive (resid)",
    "transformer":     "Transformer (P95)",
}
ALLOC_KEYS = list(DISPLAY.keys())


def _colors():
    return plt.cm.tab10(np.linspace(0, 1, len(ALLOC_KEYS)))


def _per_horizon_metric(d, h_idx, fn):
    """对每个方法在 horizon h_idx 上算 fn(alloc, actual)."""
    actual_h = d["actual"][:, :, h_idx]
    return {k: fn(d[f"alloc_{k}"][:, :, h_idx], actual_h) for k in ALLOC_KEYS}


def plot_horizon_metrics(d):
    """X=horizon (min), Y=metric, 5 条线 (一个方法一条). 主结果图."""
    horizons = d["horizons"]
    H = len(horizons)
    horizon_min = horizons * 15

    sla = {k: np.zeros(H) for k in ALLOC_KEYS}
    util = {k: np.zeros(H) for k in ALLOC_KEYS}
    for h_idx in range(H):
        sla_h = _per_horizon_metric(d, h_idx,
                                    lambda a, y: 100 * (a < y).mean())
        util_h = _per_horizon_metric(d, h_idx,
                                     lambda a, y: 100 * y.sum() / max(a.sum(), 1e-9))
        for k in ALLOC_KEYS:
            sla[k][h_idx] = sla_h[k]
            util[k][h_idx] = util_h[k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for k, c in zip(ALLOC_KEYS, _colors()):
        ax1.plot(horizon_min, sla[k], "o-", color=c, label=DISPLAY[k], lw=2, ms=8)
        ax2.plot(horizon_min, util[k], "o-", color=c, label=DISPLAY[k], lw=2, ms=8)
    ax1.axhline(5, color="gray", ls=":", alpha=0.7, label="Target = 5%")
    ax1.set(xlabel="Forecast horizon (min)", ylabel="SLA violation rate (%)",
            title="SLA vs Horizon", xticks=horizon_min)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set(xlabel="Forecast horizon (min)", ylabel="Utilization (%)",
            title="Utilization vs Horizon", xticks=horizon_min)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "horizon_metrics.png", dpi=130)
    plt.close(fig)


def plot_pareto_per_horizon(d):
    """3 panel pareto, 每个 horizon 一个 panel."""
    horizons = d["horizons"]
    H = len(horizons)
    fig, axes = plt.subplots(1, H, figsize=(5*H, 5), sharey=True)
    if H == 1:
        axes = [axes]
    for h_idx, h in enumerate(horizons):
        ax = axes[h_idx]
        sla_d = _per_horizon_metric(d, h_idx, lambda a, y: 100*(a < y).mean())
        util_d = _per_horizon_metric(d, h_idx,
                                     lambda a, y: 100*y.sum()/max(a.sum(), 1e-9))
        for k, c in zip(ALLOC_KEYS, _colors()):
            ax.scatter(sla_d[k], util_d[k], s=140, color=c, zorder=3)
            ax.annotate(DISPLAY[k], (sla_d[k], util_d[k]),
                        textcoords="offset points", xytext=(8, 5), fontsize=8)
        ax.axvline(5, color="gray", ls=":", alpha=0.7)
        ax.set(xlabel="SLA violation rate (%)", title=f"h={h} ({h*15} min)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Utilization (%)")
    fig.suptitle("Pareto: SLA vs Utilization, per horizon", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pareto_per_horizon.png", dpi=130)
    plt.close(fig)


def plot_reliability(d):
    """3 panel, P50/P90/P95 校准 vs 对角线."""
    horizons = d["horizons"]
    H = len(horizons)
    taus = np.array([0.5, 0.9, 0.95])
    fig, axes = plt.subplots(1, H, figsize=(5*H, 5), sharey=True)
    if H == 1:
        axes = [axes]
    for h_idx, h in enumerate(horizons):
        ax = axes[h_idx]
        actual_h = d["actual"][:, :, h_idx]
        cov = np.array([
            (d[f"pred_p{int(t*100)}"][:, :, h_idx] >= actual_h).mean() for t in taus
        ])
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.scatter(taus, cov, s=140, c="C0", zorder=3)
        for t, c in zip(taus, cov):
            ax.annotate(f"{c:.3f}", (t, c),
                        textcoords="offset points", xytext=(8, -8), fontsize=9)
        ax.set(xlim=(0, 1), ylim=(0, 1),
               xlabel="Predicted quantile τ", title=f"h={h} ({h*15} min)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Empirical coverage")
    fig.suptitle("Reliability diagram per horizon (Transformer)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliability.png", dpi=130)
    plt.close(fig)


def plot_per_sd_hist(d):
    """3 panel, per SD-pair 违约率分布."""
    horizons = d["horizons"]
    H = len(horizons)
    bins = np.linspace(0, 0.5, 41)
    fig, axes = plt.subplots(1, H, figsize=(6*H, 4.5), sharey=True)
    if H == 1:
        axes = [axes]
    for h_idx, h in enumerate(horizons):
        ax = axes[h_idx]
        actual_h = d["actual"][:, :, h_idx]
        for k, c in zip(ALLOC_KEYS, _colors()):
            per_sd = (d[f"alloc_{k}"][:, :, h_idx] < actual_h).mean(axis=0)
            ax.hist(per_sd, bins=bins, alpha=0.4, label=DISPLAY[k], color=c)
        ax.axvline(0.05, color="gray", ls=":", alpha=0.7)
        ax.set(xlabel="Per-SD-pair SLA violation rate", title=f"h={h} ({h*15} min)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Number of SD pairs")
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-SD-pair SLA distribution per horizon", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_sd_sla_hist.png", dpi=130)
    plt.close(fig)


def plot_sd_timeseries(d):
    """3 SD pair × 3 horizon = 9 panel, 时序 actual + P50-P95 band + violations."""
    horizons = d["horizons"]
    H = len(horizons)
    sd_names = np.load(ROOT / "data" / "sd_pair_names.npy")
    means = d["actual"][:, :, 0].mean(axis=0)
    nz = np.where(means > 1)[0]
    sorted_nz = nz[np.argsort(means[nz])]
    selected = [sorted_nz[-1], sorted_nz[len(sorted_nz)//2], sorted_nz[0]]

    fig, axes = plt.subplots(len(selected), H,
                             figsize=(5*H, 2.8*len(selected)), sharex=True)
    x = np.arange(d["actual"].shape[0])
    for r, sd in enumerate(selected):
        for h_idx in range(H):
            ax = axes[r, h_idx]
            actual_sd = d["actual"][:, sd, h_idx]
            p50_sd = d["pred_p50"][:, sd, h_idx]
            p95_sd = d["pred_p95"][:, sd, h_idx]
            ax.fill_between(x, p50_sd, p95_sd, alpha=0.25, color="C0")
            ax.plot(x, actual_sd, color="black", lw=0.8, alpha=0.7)
            ax.plot(x, p95_sd, color="C0", lw=0.9)
            viol = p95_sd < actual_sd
            if viol.any():
                ax.scatter(x[viol], actual_sd[viol], s=8, color="red", zorder=5)
            if h_idx == 0:
                ax.set_ylabel(f"{sd_names[sd]}\n(mean {means[sd]:.0f} Mbps)",
                              fontsize=8)
            if r == 0:
                ax.set_title(f"h={horizons[h_idx]} ({horizons[h_idx]*15} min)",
                             fontsize=10)
            ax.grid(alpha=0.2); ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Test step", fontsize=8)
    fig.suptitle("Per (SD pair, horizon): actual / P50-P95 band / violations",
                 y=0.999, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sd_pair_timeseries.png", dpi=130)
    plt.close(fig)


def plot_daily_cycle(d):
    """3 panel daily cycle. 横轴 hour-of-target, 纵轴 mean Mbps."""
    horizons = d["horizons"]
    H = len(horizons)
    timestamps = np.load(ROOT / "data" / "timestamps.npy")
    hour = ((timestamps.astype(np.int64) % (24*60)) // 60).astype(int)

    fig, axes = plt.subplots(1, H, figsize=(5*H, 4.5), sharey=True)
    if H == 1:
        axes = [axes]
    for h_idx, h in enumerate(horizons):
        ax = axes[h_idx]
        target_hour = hour[d["test_t"] + h]
        a_h = np.zeros(24); p50_h = np.zeros(24); p95_h = np.zeros(24)
        for hr in range(24):
            mask = target_hour == hr
            if mask.any():
                a_h[hr] = d["actual"][mask, :, h_idx].mean()
                p50_h[hr] = d["pred_p50"][mask, :, h_idx].mean()
                p95_h[hr] = d["pred_p95"][mask, :, h_idx].mean()
        hh = np.arange(24)
        ax.fill_between(hh, p50_h, p95_h, alpha=0.25, color="C0", label="P50–P95")
        ax.plot(hh, a_h, "k-", lw=1.5, label="Actual")
        ax.plot(hh, p50_h, color="C1", lw=1.2, label="P50")
        ax.plot(hh, p95_h, color="C0", lw=1.2, label="P95")
        ax.set(xticks=np.arange(0, 24, 6), xlabel="Hour of target (UTC)",
               title=f"h={h} ({h*15} min)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean Mbps over 462 SD pairs")
    axes[0].legend(fontsize=8)
    fig.suptitle("Daily cycle per horizon", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "daily_cycle.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    if not NPZ_PATH.exists():
        sys.exit(f"未找到 {NPZ_PATH}, 请先跑 run_eval.py")
    d = np.load(NPZ_PATH)
    plot_horizon_metrics(d)
    plot_pareto_per_horizon(d)
    plot_reliability(d)
    plot_per_sd_hist(d)
    plot_sd_timeseries(d)
    plot_daily_cycle(d)
    print(f"All figures saved to {FIG_DIR}")
