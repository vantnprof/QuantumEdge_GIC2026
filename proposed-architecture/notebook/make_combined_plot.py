"""
Combined QRC + classical baseline comparison plot.
All models come from the same fair_comparison.csv — identical data, split, and test rows.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("notebook/qrc_financial_results")
OUT = RESULTS_DIR / "quantumedge_vs_classical.png"

df = pd.read_csv(RESULTS_DIR / "fair_comparison.csv")

NAME_MAP = {
    "QRC-dual_ent_regime_qlike": "QRC Dual+Ent+Regime ★",
    "QRC-dual_ent_regime":       "QRC Dual+Ent+Regime",
    "QRC-dual_ent":              "QRC Dual+Ent",
    "QRC-single_long_ent":       "QRC Single+Ent",
    "HAR-RV":                    "HAR-RV",
    "HAR-RV-log":                "HAR-RV (log)",
    "ESN-500-log":               "ESN-500 (log)",
    "GARCH(1,1)":                "GARCH(1,1)",
    "ARIMA-log":                 "ARIMA (log)",
    "XGBoost-log":               "XGBoost (log)",
    "LSTM-log":                  "LSTM (log)",
    "Persistence":               "Persistence",
}

df["display"] = df["model"].map(NAME_MAP)
df = df[df["display"].notna()].copy()

def model_color(n):
    if "★" in n:          return "#2DA04A"
    if "QRC" in n:
        return "#1B5299" if "Regime" in n else "#4C8AC0"
    if n == "HAR-RV":     return "#555555"
    if "HAR-RV (log)" in n: return "#888888"
    if "LSTM" in n:       return "#9B59B6"
    if "ESN" in n:        return "#E07B39"
    if "XGBoost" in n:    return "#E74C3C"
    if "ARIMA" in n:      return "#F39C12"
    if "GARCH" in n:      return "#BDC3C7"
    return "#CCCCCC"

df["color"] = df["display"].map(model_color)

har_rmse  = float(df[df["display"] == "HAR-RV"]["rmse"].values[0])
har_qlike = float(df[df["display"] == "HAR-RV"]["qlike"].values[0])

fig, axes = plt.subplots(1, 2, figsize=(15, max(5.5, len(df) * 0.52 + 1.5)))
fig.patch.set_facecolor("white")
fig.suptitle(
    "QuantumEdge vs Classical Baselines  |  GIC 2026  |  Identical data · same train/test split",
    fontsize=12, fontweight="bold", y=1.02,
)

def hbar(ax, dfsorted, col, xlabel, har_val, fmt, title):
    y = np.arange(len(dfsorted))
    bars = ax.barh(y, dfsorted[col].values, color=dfsorted["color"].values, height=0.6, zorder=3)

    for i, row in enumerate(dfsorted.itertuples()):
        if "★" in row.display:
            bars[i].set_edgecolor("#1a7a35"); bars[i].set_linewidth(2.5)

    ax.axvline(har_val, color="#555555", linewidth=1.2, linestyle="--", zorder=4)

    ann_x = dfsorted[col].max() * 1.06
    xmax  = dfsorted[col].max() * 1.32
    for i, (v, disp) in enumerate(zip(dfsorted[col].values, dfsorted["display"].values)):
        is_star = "★" in disp
        ax.text(ann_x, i, fmt.format(v), va="center", ha="left", fontsize=8,
                fontweight="bold" if is_star else "normal",
                color="#1a7a35" if is_star else "#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(dfsorted["display"].values, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(dfsorted) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e0e0e0", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_title(title, fontsize=10, pad=6)

hbar(axes[0], df.sort_values("rmse"),  "rmse",  "RMSE",  har_rmse,  "{:.2e}", "RMSE  ↓  (lower is better)")
hbar(axes[1], df.sort_values("qlike"), "qlike", "QLIKE", har_qlike, "{:.3f}", "QLIKE  ↓  (lower is better)")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2DA04A", edgecolor="#1a7a35", lw=2, label="QRC — QLIKE readout ★"),
    Patch(facecolor="#1B5299", label="QRC — Ridge + regime"),
    Patch(facecolor="#4C8AC0", label="QRC — Ridge"),
    Patch(facecolor="#555555", label="HAR-RV (reference)"),
    Patch(facecolor="#888888", label="HAR-RV (log-space)"),
    Patch(facecolor="#9B59B6", label="LSTM (log-space)"),
    Patch(facecolor="#E07B39", label="ESN-500 (log-space)"),
    Patch(facecolor="#E74C3C", label="XGBoost (log-space)"),
    Patch(facecolor="#F39C12", label="ARIMA (log-space)"),
    Patch(facecolor="#BDC3C7", label="GARCH(1,1)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.07), framealpha=0.9)

fig.text(0.5, -0.12,
    "All models evaluated on identical rows: same df, same n_tr, same n_te, same y_true_rv.  "
    "Log-space models train on log(RV) and output exp(pred) for positive predictions and valid QLIKE.  "
    "GARCH uses walk-forward refitting on log-returns.",
    ha="center", fontsize=7.5, color="#555555")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved: {OUT}")
