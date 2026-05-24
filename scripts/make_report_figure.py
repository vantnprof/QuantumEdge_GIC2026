"""Run methods and create a QuantumEdge comparison figure.

Default behavior:
  1. Run the classical baseline pipeline.
  2. Run QRC variants for the selected dataset.
  3. Merge actual metrics from this run.
  4. Save PNG/PDF/CSV report artifacts.

Use --reported-only only when you want to recreate the old presentation figure
from fixed summary values without executing methods.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantumedge.config import ensure_runtime_dirs
from quantumedge.evaluation.metrics import mae, qlike, rmse
from quantumedge.quantum.features import extract_statevector_features, make_statevector_qnode
from quantumedge.quantum.readouts import train_ridge_readout
from quantumedge.quantum.reservoirs import DEFAULT_N_QUBITS, make_reservoir_pair
from quantumedge.quantum.series import build_angle_series


REPORTED_ROWS = [
    {"dataset": "sp500_rv", "model": "QRC Dual+Ent+Regime *", "source": "qrc", "family": "qrc_best", "rmse": 1.07e-4, "mae": np.nan, "qlike": 0.299},
    {"dataset": "sp500_rv", "model": "QRC Dual+Ent", "source": "qrc", "family": "qrc", "rmse": 1.09e-4, "mae": np.nan, "qlike": 0.362},
    {"dataset": "sp500_rv", "model": "LSTM (log)", "source": "classical", "family": "lstm", "rmse": 1.17e-4, "mae": np.nan, "qlike": 0.421},
    {"dataset": "sp500_rv", "model": "QRC Dual+Ent+Regime", "source": "qrc", "family": "qrc_residual", "rmse": 1.19e-4, "mae": np.nan, "qlike": 0.356},
    {"dataset": "sp500_rv", "model": "ESN (log)", "source": "classical", "family": "esn", "rmse": 1.20e-4, "mae": np.nan, "qlike": 0.375},
    {"dataset": "sp500_rv", "model": "XGBoost (log)", "source": "classical", "family": "xgboost", "rmse": 1.27e-4, "mae": np.nan, "qlike": 0.379},
    {"dataset": "sp500_rv", "model": "HAR-RV", "source": "classical", "family": "har", "rmse": 1.28e-4, "mae": np.nan, "qlike": 0.348},
    {"dataset": "sp500_rv", "model": "QRC Single+Ent", "source": "qrc", "family": "qrc", "rmse": 1.32e-4, "mae": np.nan, "qlike": 0.374},
    {"dataset": "sp500_rv", "model": "ARIMA (log)", "source": "classical", "family": "arima", "rmse": 1.37e-4, "mae": np.nan, "qlike": 0.420},
    {"dataset": "sp500_rv", "model": "GARCH", "source": "classical", "family": "garch", "rmse": 1.56e-4, "mae": np.nan, "qlike": 0.488},
]

COLORS = {
    "qrc_best": "#168038",
    "qrc_residual": "#1f6b2e",
    "qrc": "#0f5b8c",
    "har": "#343a40",
    "garch": "#7f8c8d",
    "arima": "#c47c00",
    "esn": "#b5541f",
    "lstm": "#7d3c98",
    "xgboost": "#d7301f",
    "classical": "#5f6c72",
}

CLASSICAL_FAMILY = {
    "GARCH": "garch",
    "HAR-RV": "har",
    "ARIMA": "arima",
    "ESN": "esn",
    "LSTM": "lstm",
    "XGBoost": "xgboost",
}

DATASET_LABELS = {
    "sp500_rv": "S&P 500 realized variance",
    "oxford_man_rv": "Oxford-Man realized variance",
    "vix": "VIX",
    "mackey_glass": "Mackey-Glass",
    "lorenz": "Lorenz",
}


def _normalise_dataset(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def _inverse_angle_series(scaler, values: np.ndarray) -> np.ndarray:
    raw = scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()
    return raw


def _positive_original_scale(dataset: str, scaler, values: np.ndarray) -> np.ndarray:
    raw = _inverse_angle_series(scaler, values)
    if dataset in {"sp500_rv", "oxford_man_rv", "vix"}:
        return np.exp(raw)
    return raw


def run_classical_results(write_plots: bool) -> pd.DataFrame:
    from quantumedge.pipelines.classical_baselines import run_pipeline

    return run_pipeline(write_plots=write_plots)


def classical_rows(metrics_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = metrics_df[metrics_df["dataset"] == dataset].copy()
    if rows.empty:
        return rows
    rows["source"] = "classical"
    rows["family"] = rows["model"].map(CLASSICAL_FAMILY).fillna("classical")
    rows["n_train"] = np.nan
    rows["n_test"] = np.nan
    rows["n_qubits"] = np.nan
    rows["backend"] = "classical"
    keep = ["dataset", "model", "source", "family", "backend", "n_qubits", "n_train", "n_test", "rmse", "mae"]
    if "qlike" in rows.columns:
        keep.append("qlike")
    else:
        rows["qlike"] = np.nan
        keep.append("qlike")
    return rows[keep]


def run_qrc_results(args: argparse.Namespace, dataset: str) -> pd.DataFrame:
    n_train = args.quantum_n_train
    n_test = args.quantum_n_test
    angles, scaler = build_angle_series(dataset, n_train=n_train, n_test=n_test)
    short_cfg, long_cfg = make_reservoir_pair(
        n_qubits=args.quantum_n_qubits,
        short_coupling_j=args.short_j,
        short_transverse_h=args.short_h,
        short_trotter_depth=args.short_depth,
        long_coupling_j=args.long_j,
        long_transverse_h=args.long_h,
        long_trotter_depth=args.long_depth,
    )

    short_node = make_statevector_qnode(short_cfg, device_name=args.device_name)
    long_node = make_statevector_qnode(long_cfg, device_name=args.device_name)
    features = extract_statevector_features(
        angles,
        short_node,
        long_node,
        n_qubits=args.quantum_n_qubits,
        label=f"QRC {dataset}",
        progress_every=args.progress_every,
    )

    experiments = {
        "QRC dual Pauli": train_ridge_readout(
            features["dual_pauli"],
            features["target"],
            n_train=n_train,
            n_test=n_test,
        ),
        "QRC dual Pauli+Ent": train_ridge_readout(
            features["dual_ent"],
            features["target"],
            n_train=n_train,
            n_test=n_test,
        ),
        "QRC single long Pauli+Ent": train_ridge_readout(
            features["single_long_ent"],
            features["target"],
            n_train=n_train,
            n_test=n_test,
        ),
    }

    rows = []
    for model_name, result in experiments.items():
        if dataset in {"sp500_rv", "oxford_man_rv", "vix"}:
            y_true = _positive_original_scale(dataset, scaler, result["y_test"])
            y_pred = np.maximum(_positive_original_scale(dataset, scaler, result["y_pred"]), 1e-10)
            row_rmse = rmse(y_true, y_pred)
            row_mae = mae(y_true, y_pred)
            row_qlike = qlike(y_true, y_pred) if dataset in {"sp500_rv", "oxford_man_rv"} else np.nan
        else:
            y_true = result["y_test"]
            y_pred = result["y_pred"]
            row_rmse = result["rmse"]
            row_mae = result["mae"]
            row_qlike = np.nan

        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "source": "qrc",
                "family": "qrc",
                "backend": "statevector",
                "n_qubits": args.quantum_n_qubits,
                "n_train": n_train,
                "n_test": n_test,
                "rmse": row_rmse,
                "mae": row_mae,
                "qlike": row_qlike,
                "ridge_alpha": result["alpha"],
                "short_j": short_cfg.coupling_j,
                "short_h": short_cfg.transverse_h,
                "short_depth": short_cfg.trotter_depth,
                "long_j": long_cfg.coupling_j,
                "long_h": long_cfg.transverse_h,
                "long_depth": long_cfg.trotter_depth,
            }
        )
    return pd.DataFrame(rows)


def build_actual_results(args: argparse.Namespace) -> pd.DataFrame:
    dataset = _normalise_dataset(args.dataset)
    parts = []

    if args.run_classical:
        classical = run_classical_results(write_plots=not args.skip_classical_plots)
        parts.append(classical_rows(classical, dataset))
    elif args.classical_csv:
        classical = pd.read_csv(args.classical_csv)
        parts.append(classical_rows(classical, dataset))

    if args.run_quantum:
        parts.append(run_qrc_results(args, dataset))
    elif args.quantum_csv:
        quantum = pd.read_csv(args.quantum_csv)
        parts.append(quantum[quantum["dataset"] == dataset].copy())

    parts = [part for part in parts if part is not None and not part.empty]
    if not parts:
        raise RuntimeError("No result rows were produced. Enable a run or provide CSV inputs.")

    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined["dataset"] = dataset
    return combined


def reported_results() -> pd.DataFrame:
    return pd.DataFrame(REPORTED_ROWS)


def _metric_for_second_panel(df: pd.DataFrame) -> str:
    if "qlike" in df.columns and df["qlike"].notna().sum() >= 2:
        return "qlike"
    return "mae"


def _draw_panel(ax, df: pd.DataFrame, metric: str, title: str) -> None:
    plot_df = df[df[metric].notna()].copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, f"No {metric.upper()} values", ha="center", va="center")
        ax.set_axis_off()
        return

    ordered = plot_df.sort_values(metric, ascending=True).reset_index(drop=True)
    y_pos = range(len(ordered))
    colors = [COLORS.get(row.family, "#5f6c72") for row in ordered.itertuples()]

    ax.barh(y_pos, ordered[metric], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(ordered["model"], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(f"{metric.upper()} ↓", fontsize=10)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)

    best = float(ordered[metric].min())
    ax.axvline(best, color="#168038", linestyle="--", linewidth=1.0, alpha=0.75)

    x_max = float(ordered[metric].max())
    x_pad = x_max * 0.02 if x_max > 0 else 0.02
    for idx, value in enumerate(ordered[metric]):
        label = f"{value:.2e}" if metric == "rmse" else f"{value:.4f}"
        is_best = value == ordered[metric].min()
        ax.text(
            x_max + x_pad,
            idx,
            label,
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold" if is_best else "normal",
            color="#168038" if is_best else "#333333",
        )

    ax.set_xlim(0, x_max * 1.25 if x_max > 0 else 1.0)


def make_figure(df: pd.DataFrame, output_dir: Path, basename: str, title_suffix: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}_data.csv"
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    df.to_csv(csv_path, index=False)

    second_metric = _metric_for_second_panel(df)

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.0), constrained_layout=False)
    fig.suptitle(
        f"QuantumEdge Method Run Results | {title_suffix}",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    _draw_panel(axes[0], df, "rmse", "RMSE  (lower is better)")
    _draw_panel(axes[1], df, second_metric, f"{second_metric.upper()}  (lower is better)")

    used_families = list(dict.fromkeys(df["family"].fillna("classical")))
    handles = [plt.Line2D([0], [0], color=COLORS.get(family, "#5f6c72"), lw=8) for family in used_families]
    labels = [family.replace("_", " ").title() for family in used_families]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(labels)),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.055),
    )

    footnote = (
        "Figure is generated from this script's run outputs. "
        "Check the companion CSV for dataset, backend, train/test sizes, and QRC reservoir settings."
    )
    fig.text(0.5, 0.02, footnote, ha="center", va="bottom", fontsize=8, color="#333333")
    fig.subplots_adjust(left=0.13, right=0.95, top=0.89, bottom=0.17, wspace=0.30)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run methods and create a QuantumEdge comparison figure.")
    parser.add_argument("--reported-only", action="store_true", help="Use fixed reported values; do not run methods.")
    parser.add_argument("--dataset", default="sp500_rv", choices=["sp500_rv", "oxford_man_rv", "vix", "mackey_glass", "lorenz"])
    parser.add_argument("--run-classical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-quantum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classical-csv", default=None, help="Use an existing classical metrics CSV instead of running classical methods.")
    parser.add_argument("--quantum-csv", default=None, help="Use an existing quantum metrics CSV instead of running QRC methods.")
    parser.add_argument("--skip-classical-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantum-n-qubits", type=int, default=DEFAULT_N_QUBITS)
    parser.add_argument("--quantum-n-train", type=int, default=3200)
    parser.add_argument("--quantum-n-test", type=int, default=800)
    parser.add_argument("--device-name", default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--short-j", type=float, default=0.3)
    parser.add_argument("--short-h", type=float, default=1.0)
    parser.add_argument("--short-depth", type=int, default=2)
    parser.add_argument("--long-j", type=float, default=1.2)
    parser.add_argument("--long-h", type=float, default=0.4)
    parser.add_argument("--long-depth", type=int, default=6)
    parser.add_argument("--output-dir", default="artifacts/results/figures")
    parser.add_argument("--basename", default="quantumedge_run_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ensure_runtime_dirs()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = settings.project_root / output_dir

    if args.reported_only:
        df = reported_results()
        title_suffix = "reported summary values"
    else:
        df = build_actual_results(args)
        dataset = _normalise_dataset(args.dataset)
        title_suffix = (
            f"{DATASET_LABELS.get(dataset, dataset)} | "
            f"QRC n={args.quantum_n_qubits}, train={args.quantum_n_train}, test={args.quantum_n_test}"
        )

    png_path, pdf_path, csv_path = make_figure(df, output_dir, args.basename, title_suffix)
    print(f"Saved figure PNG: {png_path}")
    print(f"Saved figure PDF: {pdf_path}")
    print(f"Saved data CSV  : {csv_path}")


if __name__ == "__main__":
    main()
