"""Render results.csv as a table image.

Usage: python eval/transformer/results_table.py
Output: figures/results_table.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results.csv"
OUT_PATH = HERE / "figures" / "results_table.png"

# Map CSV column names to compact display names
HEADER_MAP = {
    "Horizon_steps":            "h_step",
    "Horizon_min":              "h_min",
    "Method":                   "Method",
    "SLA_pct":                  "SLA%",
    "Util_pct":                 "Util%",
    "Avg_alloc_Mbps":           "AvgAlloc",
    "Avg_overprov_Mbps":        "OverProv",
    "Pct_pairs_SLA_above_5pct": "%Pair>5%",
    "Worst_pair_SLA_pct":       "WorstPair%",
    "Avg_violation_size_Mbps":  "ViolSize",
    "Total_unmet_Mbps":         "TotUnmet",
}


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}; please run run_eval.py first")

    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    display_header = [HEADER_MAP.get(h, h) for h in header]

    fig, ax = plt.subplots(figsize=(15, 0.6 + 0.45 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=display_header,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    # Bold header with a light gray background
    for j in range(len(display_header)):
        cell = table[0, j]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#e8e8e8")

    # Left-align and widen the Method column
    n_rows = len(rows)
    method_col_idx = header.index("Method")
    for i in range(n_rows + 1):
        cell = table[i, method_col_idx]
        cell.set_width(0.18)
        if i > 0:
            cell.set_text_props(ha="left")

    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
