"""All result visualisations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PLOTS = Path(__file__).parent.parent / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "GARCH": "#E07B54",
    "HAR-RV": "#5B8DB8",
    "ESN": "#6ABF69",
    "LSTM": "#B067A3",
    "XGBoost": "#F0C040",
    "ARIMA": "#78909C",
}
MODEL_ORDER = ["GARCH", "HAR-RV", "ARIMA", "ESN", "LSTM", "XGBoost"]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# 1. Forecast vs Actual — financial / VIX (test period)
# ---------------------------------------------------------------------------

def plot_financial_forecasts(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: dict,
    dataset_label: str,
    ylabel: str = "Realized Variance",
) -> Path:
    n_models = len(predictions)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 3 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.plot(dates, y_true, color="#333333", lw=1.2, label="Actual", zorder=3)
        ax.plot(
            dates, y_pred,
            color=PALETTE.get(name, "tab:blue"),
            lw=1.2, alpha=0.85, label=f"{name} forecast",
        )
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"{dataset_label} — {name}", fontsize=10)

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"One-step-ahead forecasts: {dataset_label} (test split)", fontsize=12)
    fig.tight_layout()
    path = PLOTS / f"forecasts_{dataset_label.lower().replace(' ', '_')}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 2. Forecast vs Actual — chaotic systems (aligned predictions)
# ---------------------------------------------------------------------------

def plot_chaotic_forecasts(
    y_true: np.ndarray,
    predictions: dict,
    dataset_label: str,
    n_show: int = 300,
) -> Path:
    """All predictions and y_true must already be aligned to the same start index."""
    n_show = min(n_show, len(y_true))
    fig, axes = plt.subplots(len(predictions), 1, figsize=(12, 3 * len(predictions)), sharex=True)
    if len(predictions) == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.plot(y_true[:n_show], color="#333333", lw=1.2, label="Actual", zorder=3)
        ax.plot(
            y_pred[:n_show],
            color=PALETTE.get(name, "tab:blue"),
            lw=1.2, alpha=0.85, label=name,
        )
        ax.set_ylabel("Value (normalised)")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"{dataset_label} — {name}", fontsize=10)

    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"One-step-ahead forecasts: {dataset_label} (first {n_show} test steps)", fontsize=12)
    fig.tight_layout()
    path = PLOTS / f"forecasts_{dataset_label.lower().replace('-', '_').replace(' ', '_')}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 3. Metrics heatmaps — RMSE and QLIKE (capped for readability)
# ---------------------------------------------------------------------------

def plot_metrics_heatmap(metrics_df: pd.DataFrame, qlike_cap: float = 5.0) -> Path:
    """Dual heatmap. QLIKE values above qlike_cap are capped and annotated with '>'.

    Only models with valid QLIKE (< qlike_cap) are included on the QLIKE heatmap.
    """
    pivot_rmse = metrics_df.pivot(index="model", columns="dataset", values="rmse")
    pivot_rmse = pivot_rmse.reindex([m for m in MODEL_ORDER if m in pivot_rmse.index])

    has_qlike = "qlike" in metrics_df.columns and metrics_df["qlike"].notna().any()
    if has_qlike:
        qlike_data = metrics_df[metrics_df["qlike"].notna()].copy()
        # Drop rows where QLIKE is unreliable (> cap)
        qlike_reliable = qlike_data[qlike_data["qlike"] <= qlike_cap]
        if not qlike_reliable.empty:
            pivot_qlike = qlike_reliable.pivot(index="model", columns="dataset", values="qlike")
            pivot_qlike = pivot_qlike.reindex(
                [m for m in MODEL_ORDER if m in pivot_qlike.index]
            )
        else:
            has_qlike = False

    if has_qlike:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    def _heatmap(data, ax, title, fmt=".5f", cmap="YlOrRd"):
        mask = data.isna()
        sns.heatmap(
            data.astype(float),
            annot=True, fmt=fmt, cmap=cmap,
            mask=mask, ax=ax, linewidths=0.5,
            annot_kws={"size": 8},
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Dataset", fontsize=9)
        ax.set_ylabel("Model", fontsize=9)

    _heatmap(pivot_rmse, ax1, "RMSE (lower = better)")
    if has_qlike:
        _heatmap(
            pivot_qlike, ax2,
            f"QLIKE (lower = better; ESN/LSTM excluded — values >1e4)",
            fmt=".3f", cmap="YlGnBu",
        )

    fig.suptitle("Classical Baseline Metrics — Model × Dataset", fontsize=13)
    fig.tight_layout()
    path = PLOTS / "metrics_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 4. LSTM training loss curves
# ---------------------------------------------------------------------------

def plot_lstm_losses(loss_histories: dict) -> Path:
    n = len(loss_histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, history) in zip(axes, loss_histories.items()):
        ax.plot(history, color=PALETTE["LSTM"], lw=1.5)
        ax.set_title(f"LSTM — {label}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
    fig.suptitle("LSTM Training Loss Curves", fontsize=12)
    fig.tight_layout()
    path = PLOTS / "lstm_training_loss.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 5. XGBoost feature importance
# ---------------------------------------------------------------------------

def plot_xgboost_importance(models: dict, feature_names: dict) -> Path:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, model) in zip(axes, models.items()):
        scores = model.feature_importances_
        names = feature_names.get(label, [f"f{i}" for i in range(len(scores))])
        idx = np.argsort(scores)
        ax.barh([names[i] for i in idx], scores[idx], color=PALETTE["XGBoost"])
        ax.set_title(f"XGBoost Importance — {label}", fontsize=10)
        ax.set_xlabel("Importance")
    fig.suptitle("XGBoost Feature Importances", fontsize=12)
    fig.tight_layout()
    path = PLOTS / "xgboost_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 6. RMSE bar chart per dataset
# ---------------------------------------------------------------------------

def plot_rmse_bars(metrics_df: pd.DataFrame) -> Path:
    datasets = metrics_df["dataset"].unique()
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = metrics_df[metrics_df["dataset"] == ds].copy().sort_values("rmse")
        colors = [PALETTE.get(m, "grey") for m in sub["model"]]
        bars = ax.bar(sub["model"], sub["rmse"], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(ds, fontsize=10)
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, sub["rmse"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.5f}", ha="center", va="bottom", fontsize=7.5,
            )
    fig.suptitle("RMSE by Dataset and Model", fontsize=12)
    fig.tight_layout()
    path = PLOTS / "rmse_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 7. Lorenz attractor
# ---------------------------------------------------------------------------

def plot_lorenz_attractor(lorenz_df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        lorenz_df["x"].values,
        lorenz_df["y"].values,
        lorenz_df["z"].values,
        lw=0.4, alpha=0.7, color="#5B8DB8",
    )
    ax.set_title("Lorenz Attractor (full series)", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    path = PLOTS / "lorenz_attractor.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 8. Radar chart — all models × all datasets (RMSE + QLIKE)
# ---------------------------------------------------------------------------

def _log_normalize(values: np.ndarray) -> np.ndarray:
    """Map RMSE values to [0, 1] performance score (1 = best) using log scale."""
    vals = np.array(values, dtype=float)
    log_vals = np.log(vals + 1e-30)
    log_min = np.nanmin(log_vals)
    log_max = np.nanmax(log_vals)
    if log_max == log_min:
        return np.ones_like(vals)
    # Invert: lower RMSE → higher score
    return 1.0 - (log_vals - log_min) / (log_max - log_min)


def plot_radar_chart(metrics_df: pd.DataFrame) -> Path:
    """Spider/radar chart showing normalised RMSE and QLIKE performance.

    Left panel:  RMSE radar — all 5 models × 4 dataset spokes.
                 Non-applicable models (GARCH/HAR-RV on chaotic) get score=0.
    Right panel: QLIKE radar — GARCH, HAR-RV, XGBoost only (ESN/LSTM QLIKE
                 is pathological due to unconstrained output; excluded).
                 Spokes: sp500_rv, oxford_man_rv, vix.
    Scores are log-normalised: 1.0 = best model on that spoke, 0.0 = worst.
    """
    # ---- RMSE panel -------------------------------------------------------
    # Datasets to show: sp500_rv, vix (level only), mackey_glass, lorenz
    # For vix: GARCH uses a different target (variance vs level) → score=0 on this spoke
    rmse_spoke_datasets = ["sp500_rv", "vix", "mackey_glass", "lorenz"]
    vix_level_models = {"ESN", "LSTM", "XGBoost", "ARIMA"}  # GARCH VIX is variance target

    rmse_df = metrics_df.pivot(index="model", columns="dataset", values="rmse")

    rmse_scores = {}
    for model in MODEL_ORDER:
        scores = []
        for ds in rmse_spoke_datasets:
            if model not in rmse_df.index or ds not in rmse_df.columns:
                scores.append(np.nan)
            elif np.isnan(rmse_df.loc[model, ds]):
                scores.append(np.nan)
            elif ds == "vix" and model not in vix_level_models:
                scores.append(np.nan)
            else:
                scores.append(rmse_df.loc[model, ds])
        rmse_scores[model] = scores

    # Normalise per spoke
    rmse_norm = {}
    for model in MODEL_ORDER:
        rmse_norm[model] = []
    for i, ds in enumerate(rmse_spoke_datasets):
        vals = np.array([rmse_scores[m][i] for m in MODEL_ORDER])
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < 2:
            normed = np.where(valid_mask, 1.0, 0.0)
        else:
            normed_valid = _log_normalize(vals[valid_mask])
            normed = np.zeros(len(MODEL_ORDER))
            normed[valid_mask] = normed_valid
        for j, model in enumerate(MODEL_ORDER):
            rmse_norm[model].append(normed[j])

    # ---- QLIKE panel ------------------------------------------------------
    qlike_models = ["GARCH", "HAR-RV", "ARIMA", "XGBoost"]
    qlike_spoke_datasets = ["sp500_rv", "oxford_man_rv", "vix"]
    qlike_df = metrics_df.pivot(index="model", columns="dataset", values="qlike")

    qlike_scores = {}
    for model in qlike_models:
        scores = []
        for ds in qlike_spoke_datasets:
            if model not in qlike_df.index or ds not in qlike_df.columns:
                scores.append(np.nan)
            elif np.isnan(qlike_df.loc[model, ds]):
                scores.append(np.nan)
            else:
                scores.append(qlike_df.loc[model, ds])
        qlike_scores[model] = scores

    qlike_norm = {m: [] for m in qlike_models}
    for i, ds in enumerate(qlike_spoke_datasets):
        vals = np.array([qlike_scores[m][i] for m in qlike_models])
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < 2:
            normed = np.where(valid_mask, 1.0, 0.0)
        else:
            normed_valid = _log_normalize(vals[valid_mask])
            normed = np.zeros(len(qlike_models))
            normed[valid_mask] = normed_valid
        for j, model in enumerate(qlike_models):
            qlike_norm[model].append(normed[j])

    # ---- Draw -------------------------------------------------------------
    fig, (ax_rmse, ax_qlike) = plt.subplots(
        1, 2, figsize=(14, 6),
        subplot_kw={"projection": "polar"},
    )

    def _draw_radar(ax, spoke_labels, model_scores, title, models_to_plot):
        n = len(spoke_labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(spoke_labels, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="grey")
        ax.yaxis.grid(True, color="grey", alpha=0.3)
        ax.xaxis.grid(True, color="grey", alpha=0.4)

        for model in models_to_plot:
            scores = model_scores[model] + model_scores[model][:1]
            color = PALETTE.get(model, "grey")
            ax.plot(angles, scores, color=color, lw=2, label=model)
            ax.fill(angles, scores, color=color, alpha=0.12)

        ax.set_title(title, fontsize=11, pad=15)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
            fontsize=8,
            framealpha=0.7,
        )

    spoke_labels_rmse = ["SP500 RV", "VIX Level\n(ESN/LSTM/XGB)", "Mackey-Glass", "Lorenz"]
    _draw_radar(ax_rmse, spoke_labels_rmse, rmse_norm, "RMSE Performance\n(1 = best)", MODEL_ORDER)

    spoke_labels_qlike = ["SP500 RV", "Oxford-Man RV", "VIX\n(variance)"]
    _draw_radar(ax_qlike, spoke_labels_qlike, qlike_norm,
                "QLIKE Performance\n(1 = best; ESN/LSTM excluded)", qlike_models)

    fig.suptitle(
        "Classical Baselines — Normalised Performance Radar\n"
        "(log-normalised per spoke; 1.0 = best model, 0.0 = worst)",
        fontsize=12,
    )
    fig.tight_layout()
    path = PLOTS / "radar_chart.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 9. Regime overlay — RV time series coloured by HMM volatility regime
# ---------------------------------------------------------------------------

def plot_regime_overlay(
    dates: pd.DatetimeIndex,
    rv_true: np.ndarray,
    regimes: np.ndarray,
    predictions: dict,
    dataset_label: str = "SP500 RV",
) -> Path:
    """Two-panel figure designed for readability.

    Top panel   : Actual RV with strong regime background shading only —
                  shows clearly when the market is in each volatility state.
    Bottom panel: All model forecasts overlaid on actual RV in a single axes —
                  allows direct model comparison with regime context.
    """
    import matplotlib.patches as mpatches
    from models.hmm_regimes import REGIME_COLORS, REGIME_NAMES

    def _shade(ax, alpha):
        """Fill contiguous regime blocks with background colour."""
        prev_r, start_i = int(regimes[0]), 0
        for i in range(1, len(regimes) + 1):
            cur_r = int(regimes[i]) if i < len(regimes) else -1
            if cur_r != prev_r:
                ax.axvspan(
                    dates[start_i], dates[min(i, len(dates) - 1)],
                    alpha=alpha, color=REGIME_COLORS[prev_r], lw=0, zorder=0,
                )
                prev_r, start_i = cur_r, i

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4]},
    )

    # ---- Top panel: regime map ----
    _shade(ax1, alpha=0.40)
    ax1.plot(dates, rv_true, color="#111111", lw=1.4, label="Realized Variance", zorder=3)
    ax1.set_ylabel("Realized Variance", fontsize=10)
    ax1.set_title(
        f"S&P 500 Realized Variance — HMM Volatility Regime Map (test period)",
        fontsize=11,
    )
    regime_patches = [
        mpatches.Patch(color=REGIME_COLORS[i], alpha=0.7, label=f"{REGIME_NAMES[i]} volatility")
        for i in range(3)
    ]
    ax1.legend(
        handles=regime_patches + [plt.Line2D([0], [0], color="#111111", lw=1.4, label="Actual RV")],
        fontsize=9, loc="upper right", framealpha=0.85,
    )

    # ---- Bottom panel: all forecasts ----
    _shade(ax2, alpha=0.15)
    ax2.plot(dates, rv_true, color="#111111", lw=2.0, label="Actual", zorder=5)
    for name, y_pred in predictions.items():
        n = min(len(dates), len(rv_true), len(y_pred))
        ax2.plot(
            dates[:n], y_pred[:n],
            color=PALETTE.get(name, "grey"),
            lw=1.1, alpha=0.85, label=name, zorder=4,
        )
    ax2.set_ylabel("Realized Variance", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_title("Model Forecasts vs Actual RV (regime background for context)", fontsize=11)
    ax2.legend(fontsize=9, ncol=4, loc="upper right", framealpha=0.85)

    fig.suptitle(
        "HMM Volatility Regimes & One-Step-Ahead Forecasts — S&P 500 RV\n"
        "Shading: green = Low vol  |  amber = Medium vol  |  red = High vol",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    path = PLOTS / "regime_overlay.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 9b. Regime dashboard — log-scale RV + regime strip + per-regime RMSE
# ---------------------------------------------------------------------------

def plot_regime_dashboard(
    dates: pd.DatetimeIndex,
    rv_true: np.ndarray,
    regimes: np.ndarray,
    predictions: dict,
) -> Path:
    """Publication-quality 3-row dashboard.

    Row 1 (tall)  : Log-scale RV + regime background + all model forecasts.
    Row 2 (strip) : Discrete regime colour bar — exact transition dates at a glance.
    Row 3 (medium): Per-regime RMSE — which models degrade in high-vol periods?
    """
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    from models.hmm_regimes import REGIME_COLORS, REGIME_NAMES

    # ---- Layout: explicit top/bottom for each axes so gaps are controlled ----
    fig = plt.figure(figsize=(16, 15))

    # Define axes positions manually: [left, bottom, width, height]
    # top panel: occupies y 0.52 → 0.93
    ax_ts    = fig.add_axes([0.08, 0.52, 0.88, 0.40])
    # strip: sits just below top panel with a small gap, y 0.46 → 0.51
    ax_strip = fig.add_axes([0.08, 0.46, 0.88, 0.04], sharex=ax_ts)
    # bar chart: y 0.06 → 0.40, large gap from strip
    ax_bar   = fig.add_axes([0.08, 0.06, 0.88, 0.33])

    # ------------------------------------------------------------------ #
    # Row 1: log-scale RV + regime shading + model lines
    # ------------------------------------------------------------------ #
    def _shade(ax, alpha):
        prev_r, start_i = int(regimes[0]), 0
        for i in range(1, len(regimes) + 1):
            cur_r = int(regimes[i]) if i < len(regimes) else -1
            if cur_r != prev_r:
                ax.axvspan(
                    dates[start_i], dates[min(i, len(dates) - 1)],
                    alpha=alpha, color=REGIME_COLORS[prev_r], lw=0, zorder=0,
                )
                prev_r, start_i = cur_r, i

    _shade(ax_ts, alpha=0.22)
    for name, y_pred in predictions.items():
        n = min(len(dates), len(rv_true), len(y_pred))
        ax_ts.plot(
            dates[:n], np.maximum(y_pred[:n], 1e-10),
            color=PALETTE.get(name, "grey"), lw=1.0, alpha=0.75, label=name, zorder=3,
        )
    ax_ts.plot(dates, rv_true, color="#111111", lw=2.0, label="Actual", zorder=5)

    ax_ts.set_yscale("log")
    ax_ts.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0e}"
    ))
    ax_ts.set_ylabel("Realized Variance\n(log scale)", fontsize=10, labelpad=8)
    ax_ts.set_title(
        "S&P 500 Realized Variance — Log Scale Reveals Structure Across All Regimes",
        fontsize=12, pad=10,
    )

    # Annotate COVID peak
    peak_i = int(np.argmax(rv_true))
    ax_ts.annotate(
        f"Peak RV  {dates[peak_i].strftime('%b %Y')}",
        xy=(dates[peak_i], rv_true[peak_i]),
        xytext=(-80, -45), textcoords="offset points",
        fontsize=8.5, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0),
    )

    regime_patches = [
        mpatches.Patch(color=REGIME_COLORS[i], alpha=0.75, label=f"{REGIME_NAMES[i]} vol")
        for i in range(3)
    ]
    model_lines = [
        plt.Line2D([0], [0], color="#111111", lw=2.0, label="Actual")
    ] + [
        plt.Line2D([0], [0], color=PALETTE.get(n, "grey"), lw=1.3, label=n)
        for n in predictions
    ]
    ax_ts.legend(
        handles=model_lines + regime_patches,
        fontsize=9, ncol=3, loc="upper left",
        framealpha=0.9, borderpad=0.7, labelspacing=0.4,
    )
    plt.setp(ax_ts.get_xticklabels(), visible=False)
    ax_ts.tick_params(axis="y", labelsize=9)

    # ------------------------------------------------------------------ #
    # Row 2: regime colour strip
    # ------------------------------------------------------------------ #
    prev_r, start_i = int(regimes[0]), 0
    for i in range(1, len(regimes) + 1):
        cur_r = int(regimes[i]) if i < len(regimes) else -1
        if cur_r != prev_r:
            ax_strip.axvspan(
                dates[start_i], dates[min(i, len(dates) - 1)],
                alpha=0.88, color=REGIME_COLORS[prev_r], lw=0,
            )
            prev_r, start_i = cur_r, i

    ax_strip.set_yticks([])
    ax_strip.set_ylabel("Regime\nstate", fontsize=8, rotation=0,
                        labelpad=38, va="center", ha="right")
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    # Place regime labels inside the longest contiguous block of each regime
    for r_idx in range(3):
        positions = np.where(regimes == r_idx)[0]
        if len(positions) == 0:
            continue
        # Find the longest contiguous run for this regime
        runs, run_start = [], positions[0]
        for k in range(1, len(positions) + 1):
            if k == len(positions) or positions[k] != positions[k - 1] + 1:
                runs.append((run_start, positions[k - 1]))
                if k < len(positions):
                    run_start = positions[k]
        longest = max(runs, key=lambda r: r[1] - r[0])
        mid_idx = (longest[0] + longest[1]) // 2
        ax_strip.text(
            dates[mid_idx], 0.5, REGIME_NAMES[r_idx],
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white",  # white readable on all three regime colours
            transform=ax_strip.get_xaxis_transform(),
        )

    # Compact year-only tick labels so they don't crowd the bar chart title below
    import matplotlib.dates as mdates
    ax_strip.xaxis.set_major_locator(mdates.YearLocator())
    ax_strip.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_strip.tick_params(axis="x", labelsize=8, pad=3)

    # ------------------------------------------------------------------ #
    # Row 3: per-regime RMSE grouped bar chart
    # ------------------------------------------------------------------ #
    model_names = list(predictions.keys())
    n_models = len(model_names)
    group_w = 0.75
    bar_w = group_w / n_models
    x_base = np.arange(3)

    # Pre-compute all per-regime RMSE values so max is known before drawing
    all_rmse = {}
    for name in model_names:
        y_pred = predictions[name]
        n = min(len(rv_true), len(y_pred))
        vals = []
        for r_idx in range(3):
            mask = regimes[:n] == r_idx
            val = float(np.sqrt(np.mean((rv_true[:n][mask] - y_pred[:n][mask]) ** 2))) \
                  if mask.sum() > 0 else 0.0
            vals.append(val)
        all_rmse[name] = vals
    max_rmse = max(v for vals in all_rmse.values() for v in vals)

    for j, name in enumerate(model_names):
        rmse_per_regime = all_rmse[name]
        x_pos = x_base + (j - n_models / 2 + 0.5) * bar_w
        bars = ax_bar.bar(
            x_pos, rmse_per_regime,
            width=bar_w * 0.85,
            color=PALETTE.get(name, "grey"),
            label=name, edgecolor="white", linewidth=0.5,
        )
        # Value labels above each bar, horizontal, consistent offset
        for bar, val in zip(bars, rmse_per_regime):
            if val > 0:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_rmse * 0.012,
                    f"{val:.1e}",
                    ha="center", va="bottom", fontsize=7, rotation=0,
                )

    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels(
        [f"{REGIME_NAMES[i]} Volatility\n({int((regimes == i).sum())} test days)"
         for i in range(3)],
        fontsize=10,
    )
    ax_bar.set_ylabel("RMSE", fontsize=10, labelpad=8)
    ax_bar.set_ylim(0, max_rmse * 1.25)
    ax_bar.set_title(
        "Per-Regime RMSE — Does forecast accuracy hold up during high-volatility periods?",
        fontsize=11, pad=10,
    )
    ax_bar.legend(fontsize=9, ncol=n_models, loc="upper left",
                  framealpha=0.9, borderpad=0.7)
    ax_bar.tick_params(axis="y", labelsize=9)

    fig.suptitle(
        "Volatility Regime Dashboard — S&P 500 (test period)",
        fontsize=14, y=0.97,
    )

    path = PLOTS / "regime_dashboard.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 10. Regime classification accuracy bar chart
# ---------------------------------------------------------------------------

def plot_regime_accuracy(model_accuracies: dict) -> Path:
    """Bar chart: % of test days each model correctly classifies the vol regime."""
    models = list(model_accuracies.keys())
    accs = [model_accuracies[m] * 100 for m in models]
    colors = [PALETTE.get(m, "grey") for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Volatility Regime Classification Accuracy\n(Low / Medium / High — HMM ground truth)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.axhline(33.3, color="grey", lw=1, ls="--", label="Random baseline (33%)")
    ax.legend(fontsize=8)

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    path = PLOTS / "regime_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


# ---------------------------------------------------------------------------
# 11. Sharpe ratio comparison bar chart
# ---------------------------------------------------------------------------

def plot_sharpe_comparison(sharpe_dict: dict) -> Path:
    """Bar chart comparing annualised Sharpe ratios across models + buy-and-hold."""
    models = list(sharpe_dict.keys())
    sharpes = [sharpe_dict[m] for m in models]
    colors = [
        "#888888" if m == "Buy & Hold" else PALETTE.get(m, "grey")
        for m in models
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(models, sharpes, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    bnh = sharpe_dict.get("Buy & Hold", None)
    if bnh is not None:
        ax.axhline(bnh, color="#888888", lw=1.2, ls="--", label=f"Buy & Hold ({bnh:.2f})")
        ax.legend(fontsize=8)

    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.set_title(
        "Volatility-Timing Strategy Sharpe Ratios\n"
        "(Position: Low vol→full, Med vol→half, High vol→cash)",
        fontsize=11,
    )

    label_offset = max(abs(s) for s in sharpes) * 0.03
    for bar, val in zip(bars, sharpes):
        if val >= 0:
            ypos, va = val + label_offset, "bottom"
        else:
            ypos, va = val - label_offset, "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos, f"{val:.2f}", ha="center", va=va, fontsize=9,
        )

    fig.tight_layout()
    path = PLOTS / "sharpe_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path
