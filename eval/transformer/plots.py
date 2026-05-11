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
    """主结果图. 单 horizon → 柱状图 (5 个方法); 多 horizon → 折线图."""
    horizons = d["horizons"]
    H = len(horizons)

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

    if H == 1:
        h = int(horizons[0])
        x_pos = np.arange(len(ALLOC_KEYS))
        labels = [DISPLAY[k] for k in ALLOC_KEYS]
        colors = list(_colors())
        sla_vals = [sla[k][0] for k in ALLOC_KEYS]
        util_vals = [util[k][0] for k in ALLOC_KEYS]

        bars1 = ax1.bar(x_pos, sla_vals, color=colors)
        ax1.axhline(5, color="gray", ls=":", alpha=0.7, label="Target = 5%")
        ax1.set_xticks(x_pos); ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        for b, v in zip(bars1, sla_vals):
            ax1.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)
        ax1.set(ylabel="SLA violation rate (%)",
                title=f"SLA at h={h} ({h*15} min ahead)")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")

        bars2 = ax2.bar(x_pos, util_vals, color=colors)
        ax2.set_xticks(x_pos); ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        for b, v in zip(bars2, util_vals):
            ax2.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=8)
        ax2.set(ylabel="Utilization (%)",
                title=f"Util at h={h} ({h*15} min ahead)")
        ax2.grid(alpha=0.3, axis="y")
    else:
        horizon_min = horizons * 15
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
    """每 horizon 一个 panel: SLA vs Util 散点."""
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


def plot_violation_severity_cdf(d):
    """每 horizon 一个 panel: 违约严重度 CDF (log x). 突出 ViolSize 优势.

    每个方法收集所有 (actual - alloc) > 0 的违约量, 画 CDF.
    Transformer 的曲线越早爬到 1.0, 说明严重违约越少.
    """
    horizons = d["horizons"]
    H = len(horizons)
    fig, axes = plt.subplots(1, H, figsize=(6*H, 5), sharey=True)
    if H == 1:
        axes = [axes]
    for h_idx, h in enumerate(horizons):
        ax = axes[h_idx]
        actual_h = d["actual"][:, :, h_idx]
        for k, c in zip(ALLOC_KEYS, _colors()):
            alloc_h = d[f"alloc_{k}"][:, :, h_idx]
            diff = actual_h - alloc_h
            viols = diff[diff > 0]
            if len(viols) == 0:
                continue
            sorted_viols = np.sort(viols)
            cdf = np.arange(1, len(sorted_viols) + 1) / len(sorted_viols)
            ax.plot(sorted_viols, cdf, color=c, lw=2,
                    label=f"{DISPLAY[k]} (n={len(viols):,})")
        ax.set_xscale("log")
        ax.set_xlim(left=0.1)
        ax.set_ylim(0, 1.02)
        ax.set(xlabel="Violation size (Mbps, log scale)",
               title=f"h={h} ({h*15} min)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel(r"CDF: $P(\text{violation size} \leq x)$")
    fig.suptitle("Violation severity CDF per horizon "
                 "(curves rising left = smaller violations)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "violation_severity_cdf.png", dpi=130)
    plt.close(fig)


def plot_per_sd_sla_sorted(d):
    """每 horizon 一个 panel: 462 个 pair 按 per-pair SLA 降序画曲线.

    一图同时展示 %Pair>5% (5% 横线之上面积) 和 WorstPair% (最左点).
    Transformer 的曲线在右侧应该最低 = 长尾控制最好.
    """
    horizons = d["horizons"]
    H = len(horizons)
    fig, axes = plt.subplots(1, H, figsize=(6*H, 5), sharey=True)
    if H == 1:
        axes = [axes]
    for h_idx, h in enumerate(horizons):
        ax = axes[h_idx]
        actual_h = d["actual"][:, :, h_idx]
        for k, c in zip(ALLOC_KEYS, _colors()):
            alloc_h = d[f"alloc_{k}"][:, :, h_idx]
            per_pair_sla = 100 * (alloc_h < actual_h).mean(axis=0)
            sorted_sla = np.sort(per_pair_sla)[::-1]   # 降序
            x = np.arange(1, len(sorted_sla) + 1)
            ax.plot(x, sorted_sla, color=c, lw=1.8, label=DISPLAY[k])
        ax.axhline(5, color="gray", ls=":", alpha=0.7, label="Target = 5%")
        ax.set(xlabel="SD pair rank (sorted by per-pair SLA, desc)",
               title=f"h={h} ({h*15} min)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Per-pair SLA violation rate (%)")
    fig.suptitle("Per-pair SLA sorted curve "
                 "(lower-right = better long-tail control)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_sd_sla_sorted.png", dpi=130)
    plt.close(fig)


def plot_cumulative_unmet(d):
    """每 horizon 一个 panel: 测试集上累积未满足需求随时间增长.

    Y 轴 log scale 让小值方法和大值方法都看得见.
    Transformer 曲线斜率应该最缓 (除 Static Peak 外).
    """
    horizons = d["horizons"]
    H = len(horizons)
    fig, axes = plt.subplots(1, H, figsize=(6*H, 5), sharey=True)
    if H == 1:
        axes = [axes]
    for h_idx, h in enumerate(horizons):
        ax = axes[h_idx]
        actual_h = d["actual"][:, :, h_idx]
        for k, c in zip(ALLOC_KEYS, _colors()):
            alloc_h = d[f"alloc_{k}"][:, :, h_idx]
            unmet_per_step = np.maximum(actual_h - alloc_h, 0).sum(axis=1)   # [N]
            cumsum = np.cumsum(unmet_per_step)
            total = cumsum[-1]
            ax.plot(np.arange(len(cumsum)), cumsum, color=c, lw=1.8,
                    label=f"{DISPLAY[k]} (total={total:,.0f})")
        ax.set_yscale("log")
        ax.set(xlabel="Test step", title=f"h={h} ({h*15} min)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("Cumulative unmet demand (Mbps, log scale)")
    fig.suptitle("Cumulative unmet demand over test set "
                 "(flatter slope = less total bandwidth lost)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cumulative_unmet.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    if not NPZ_PATH.exists():
        sys.exit(f"未找到 {NPZ_PATH}, 请先跑 run_eval.py")
    d = np.load(NPZ_PATH)
    plot_horizon_metrics(d)
    plot_pareto_per_horizon(d)
    plot_violation_severity_cdf(d)
    plot_per_sd_sla_sorted(d)
    plot_cumulative_unmet(d)
    print(f"All figures saved to {FIG_DIR}")
