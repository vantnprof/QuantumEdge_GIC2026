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
from quantumedge.evaluation.metrics import mae, qlike, regime_accuracy, rmse, volatility_timing_sharpe
from quantumedge.experiments.benchmark_config import (
    CLASSICAL_BENCHMARK,
    CLASSICAL_FAMILY,
    DATASET_LABELS,
    QUANTUM_BENCHMARK,
    REPORT_COLORS,
    REPORT_FIGURE,
)
from quantumedge.quantum.financial import (
    angle_predictions_to_rv,
    build_financial_qrc_inputs,
    extract_financial_statevector_features,
)
from quantumedge.quantum.features import extract_statevector_features, make_statevector_qnode
from quantumedge.quantum.readouts import train_qlike_readout, train_regime_readout, train_ridge_readout
from quantumedge.quantum.reservoirs import make_reservoir_pair
from quantumedge.quantum.series import build_angle_series, build_project_chaotic_angles


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

COLORS = REPORT_COLORS

FINANCIAL_QRC_DISPLAY = {
    "dual_pauli": ("QRC Dual", "qrc"),
    "dual_ent": ("QRC Dual+Ent", "qrc"),
    "dual_ent_regime": ("QRC Dual+Ent+Regime", "qrc_residual"),
    "single_long_ent": ("QRC Single+Ent", "qrc"),
    "regime_conditioned": ("QRC Dual+Ent Regime-Blend", "qrc_residual"),
    "dual_ent_qlike": ("QRC Dual+Ent *", "qrc_best"),
    "dual_ent_regime_qlike": ("QRC Dual+Ent+Regime *", "qrc_best"),
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


def target_type_for_model(dataset: str, model: str) -> str:
    """Return the evaluation target represented by one result row."""
    dataset = _normalise_dataset(dataset)
    if dataset in {"sp500_rv", "oxford_man_rv"}:
        return "realized_variance"
    if dataset == "vix":
        return "vix_return_variance" if model == "GARCH" else "vix_level"
    if dataset in {"mackey_glass", "lorenz"}:
        return "normalized_state"
    return "unknown"


def run_classical_results(write_plots: bool) -> pd.DataFrame:
    from quantumedge.pipelines.classical_baselines import run_pipeline

    return run_pipeline(write_plots=write_plots)


def classical_split_lengths(dataset: str) -> tuple[int, int]:
    """Return the same temporal split lengths used by the classical pipeline."""
    from quantumedge.data.loaders import load_all
    from quantumedge.features.builders import prepare_all

    prepared = prepare_all(load_all())
    if dataset in {"sp500_rv", "oxford_man_rv"}:
        split = prepared["financial_split"]
        return len(split["train"]), len(split["test"])
    if dataset == "vix":
        split = prepared["vix_split"]
        return len(split["train"]), len(split["test"])
    if dataset == "mackey_glass":
        split = prepared["mackey_glass_split"]
        return len(split["X_train"]), len(split["X_test"])
    if dataset == "lorenz":
        split = prepared["lorenz_split"]
        return len(split["X_train"]), len(split["X_test"])
    raise ValueError(f"Unsupported dataset for split lookup: {dataset}")


def classical_rows(metrics_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = metrics_df[metrics_df["dataset"] == dataset].copy()
    if rows.empty:
        return rows
    rows["source"] = "classical"
    rows["family"] = rows["model"].map(CLASSICAL_FAMILY).fillna("classical")
    n_train, n_test = classical_split_lengths(dataset)
    rows["n_train"] = n_train
    rows["n_test"] = n_test
    rows["n_qubits"] = np.nan
    rows["backend"] = "classical"
    rows["split_policy"] = "classical_temporal_split"
    rows["target_type"] = rows["model"].map(lambda model: target_type_for_model(dataset, model))
    keep = [
        "dataset",
        "model",
        "source",
        "family",
        "backend",
        "target_type",
        "n_qubits",
        "n_train",
        "n_test",
        "split_policy",
        "rmse",
        "mae",
    ]
    if "qlike" in rows.columns:
        keep.append("qlike")
    else:
        rows["qlike"] = np.nan
        keep.append("qlike")
    return rows[keep]


def _reservoir_configs_from_args(args: argparse.Namespace):
    return make_reservoir_pair(
        n_qubits=args.quantum_n_qubits,
        short_coupling_j=args.short_j,
        short_transverse_h=args.short_h,
        short_trotter_depth=args.short_depth,
        long_coupling_j=args.long_j,
        long_transverse_h=args.long_h,
        long_trotter_depth=args.long_depth,
    )


def run_financial_qrc_results(
    args: argparse.Namespace,
    dataset: str,
    n_train: int,
    n_test: int,
    split_policy: str,
) -> pd.DataFrame:
    """Run the notebook-style financial QRC variants."""
    inputs = build_financial_qrc_inputs(dataset, n_train=n_train, n_test=n_test)
    short_cfg, long_cfg = _reservoir_configs_from_args(args)

    short_node = make_statevector_qnode(short_cfg, device_name=args.device_name)
    long_node = make_statevector_qnode(long_cfg, device_name=args.device_name)
    features = extract_financial_statevector_features(
        inputs,
        short_node,
        long_node,
        n_qubits=args.quantum_n_qubits,
        label=f"QRC {dataset}",
        progress_every=args.progress_every,
    )

    target = features["target"]
    experiments = {
        "dual_pauli": train_ridge_readout(features["dual_pauli"], target, n_train=n_train, n_test=n_test),
        "dual_ent": train_ridge_readout(features["dual_ent"], target, n_train=n_train, n_test=n_test),
        "dual_ent_regime": train_ridge_readout(
            features["dual_ent_regime"],
            target,
            n_train=n_train,
            n_test=n_test,
        ),
        "single_long_ent": train_ridge_readout(
            features["single_long_ent"],
            target,
            n_train=n_train,
            n_test=n_test,
        ),
        "regime_conditioned": train_regime_readout(
            features["dual_ent"],
            target,
            n_train=n_train,
            n_test=n_test,
            regime_labels_train=inputs.train_regimes,
            regime_posteriors_test=inputs.test_posteriors,
        ),
        "dual_ent_qlike": train_qlike_readout(
            features["dual_ent"],
            target,
            n_train=n_train,
            n_test=n_test,
            angle_scaler=inputs.angle_scaler,
        ),
        "dual_ent_regime_qlike": train_qlike_readout(
            features["dual_ent_regime"],
            target,
            n_train=n_train,
            n_test=n_test,
            angle_scaler=inputs.angle_scaler,
        ),
    }

    rows = []
    for key, result in experiments.items():
        model_name, family = FINANCIAL_QRC_DISPLAY[key]
        y_pred = angle_predictions_to_rv(inputs.angle_scaler, result["y_pred"])
        n = min(len(y_pred), len(inputs.target_rv))
        y_true = inputs.target_rv[:n]
        y_pred = np.maximum(y_pred[:n], 1e-10)
        row = {
            "dataset": dataset,
            "model": model_name,
            "model_key": key,
            "source": "qrc",
            "family": family,
            "backend": "statevector",
            "feature_set": "notebook_multifeature_hmm",
            "target_type": "realized_variance",
            "n_qubits": args.quantum_n_qubits,
            "n_train": n_train,
            "n_test": n_test,
            "split_policy": split_policy,
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "qlike": qlike(y_true, y_pred),
            "regime_acc": regime_accuracy(inputs.true_test_regimes[:n], y_pred, inputs.hmm_thresholds),
            "sharpe": volatility_timing_sharpe(y_pred, inputs.test_returns[:n], inputs.hmm_thresholds),
            "readout_alpha": result["alpha"],
            "validation_qlike": result.get("validation_qlike", np.nan),
            "optimizer_success": result.get("optimizer_success", np.nan),
            "short_j": short_cfg.coupling_j,
            "short_h": short_cfg.transverse_h,
            "short_depth": short_cfg.trotter_depth,
            "long_j": long_cfg.coupling_j,
            "long_h": long_cfg.transverse_h,
            "long_depth": long_cfg.trotter_depth,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_qrc_results(
    args: argparse.Namespace,
    dataset: str,
    n_train: int,
    n_test: int,
    split_policy: str,
) -> pd.DataFrame:
    if dataset in {"sp500_rv", "oxford_man_rv"}:
        return run_financial_qrc_results(args, dataset, n_train, n_test, split_policy)

    if dataset in {"mackey_glass", "lorenz"} and split_policy == "matched_classical_temporal_split":
        angles, scaler = build_project_chaotic_angles(dataset, n_train=n_train, n_test=n_test)
        feature_set = "project_classical_split_scalar"
    else:
        angles, scaler = build_angle_series(dataset, n_train=n_train, n_test=n_test)
        feature_set = "scalar_series"
    short_cfg, long_cfg = _reservoir_configs_from_args(args)

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
            y_true = _inverse_angle_series(scaler, result["y_test"])
            y_pred = _inverse_angle_series(scaler, result["y_pred"])
            row_rmse = rmse(y_true, y_pred)
            row_mae = mae(y_true, y_pred)
            row_qlike = np.nan

        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "source": "qrc",
                "family": "qrc",
                "backend": "statevector",
                "feature_set": feature_set,
                "target_type": target_type_for_model(dataset, model_name),
                "n_qubits": args.quantum_n_qubits,
                "n_train": n_train,
                "n_test": n_test,
                "split_policy": split_policy,
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
    qrc_n_train = args.quantum_n_train
    qrc_n_test = args.quantum_n_test
    split_policy = "explicit_quantum_split"

    should_match_classical = args.match_classical_split and (
        args.run_classical or args.classical_csv is not None
    )
    if should_match_classical:
        qrc_n_train, qrc_n_test = classical_split_lengths(dataset)
        split_policy = "matched_classical_temporal_split"
        print(
            "[fairness] Matching QRC split to classical temporal split: "
            f"train={qrc_n_train}, test={qrc_n_test}"
        )

    if args.run_classical:
        classical = run_classical_results(write_plots=not args.skip_classical_plots)
        parts.append(classical_rows(classical, dataset))
    elif args.classical_csv:
        classical = pd.read_csv(args.classical_csv)
        parts.append(classical_rows(classical, dataset))

    if args.run_quantum:
        parts.append(run_qrc_results(args, dataset, qrc_n_train, qrc_n_test, split_policy))
    elif args.quantum_csv:
        quantum = pd.read_csv(args.quantum_csv)
        parts.append(quantum[quantum["dataset"] == dataset].copy())

    parts = [part for part in parts if part is not None and not part.empty]
    if not parts:
        raise RuntimeError("No result rows were produced. Enable a run or provide CSV inputs.")

    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined["dataset"] = dataset
    validate_fair_comparison(combined, dataset)
    return combined


def validate_fair_comparison(df: pd.DataFrame, dataset: str) -> None:
    """Fail fast on obvious unfair comparisons."""
    datasets = set(df["dataset"].dropna().unique())
    if datasets != {dataset}:
        raise ValueError(f"Figure can compare one dataset at a time; got {sorted(datasets)}")

    grouped = df.groupby("target_type", dropna=False) if "target_type" in df.columns else [(None, df)]
    for target_type, group in grouped:
        if {"classical", "qrc"} <= set(group["source"].dropna().unique()):
            split_pairs = set(
                tuple(pair)
                for pair in group[["n_train", "n_test"]].dropna().astype(int).drop_duplicates().values
            )
            if len(split_pairs) > 1:
                raise ValueError(
                    "Unfair comparison: methods use different train/test sizes. "
                    f"Dataset={dataset}, target_type={target_type}, "
                    f"observed split pairs: {sorted(split_pairs)}. "
                    "Use --match-classical-split or compare methods separately."
                )

    if "target_type" in df.columns and df["target_type"].nunique(dropna=True) > 1:
        print(
            "[fairness] Multiple target types are present; compare rows only within "
            "the same target_type:",
            ", ".join(sorted(df["target_type"].dropna().astype(str).unique())),
        )

    if dataset in {"sp500_rv", "oxford_man_rv"} and "qlike" in df.columns:
        missing = df[df["qlike"].isna()]
        if not missing.empty:
            print(
                "[fairness] Warning: some volatility rows have no QLIKE and will be omitted "
                "from the QLIKE panel:",
                ", ".join(missing["model"].astype(str).tolist()),
            )


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
    parser.add_argument(
        "--dataset",
        default=REPORT_FIGURE.dataset,
        choices=["sp500_rv", "oxford_man_rv", "vix", "mackey_glass", "lorenz"],
    )
    parser.add_argument("--run-classical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-quantum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classical-csv", default=None, help="Use an existing classical metrics CSV instead of running classical methods.")
    parser.add_argument("--quantum-csv", default=None, help="Use an existing quantum metrics CSV instead of running QRC methods.")
    parser.add_argument(
        "--match-classical-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When classical rows are included, force QRC to use the same temporal train/test split.",
    )
    parser.add_argument(
        "--skip-classical-plots",
        action=argparse.BooleanOptionalAction,
        default=not CLASSICAL_BENCHMARK.write_plots,
    )
    parser.add_argument("--quantum-n-qubits", type=int, default=QUANTUM_BENCHMARK.n_qubits)
    parser.add_argument("--quantum-n-train", type=int, default=QUANTUM_BENCHMARK.n_train)
    parser.add_argument("--quantum-n-test", type=int, default=QUANTUM_BENCHMARK.n_test)
    parser.add_argument("--device-name", default=QUANTUM_BENCHMARK.device_name)
    parser.add_argument("--progress-every", type=int, default=QUANTUM_BENCHMARK.progress_every)
    parser.add_argument("--short-j", type=float, default=QUANTUM_BENCHMARK.short_j)
    parser.add_argument("--short-h", type=float, default=QUANTUM_BENCHMARK.short_h)
    parser.add_argument("--short-depth", type=int, default=QUANTUM_BENCHMARK.short_depth)
    parser.add_argument("--long-j", type=float, default=QUANTUM_BENCHMARK.long_j)
    parser.add_argument("--long-h", type=float, default=QUANTUM_BENCHMARK.long_h)
    parser.add_argument("--long-depth", type=int, default=QUANTUM_BENCHMARK.long_depth)
    parser.add_argument("--output-dir", default=REPORT_FIGURE.output_dir)
    parser.add_argument("--basename", default=REPORT_FIGURE.basename)
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
        qrc_rows = df[df["source"] == "qrc"]
        if not qrc_rows.empty:
            qrc_train = int(qrc_rows["n_train"].iloc[0])
            qrc_test = int(qrc_rows["n_test"].iloc[0])
        else:
            qrc_train = args.quantum_n_train
            qrc_test = args.quantum_n_test
        title_suffix = (
            f"{DATASET_LABELS.get(dataset, dataset)} | "
            f"QRC n={args.quantum_n_qubits}, train={qrc_train}, test={qrc_test}"
        )

    png_path, pdf_path, csv_path = make_figure(df, output_dir, args.basename, title_suffix)
    print(f"Saved figure PNG: {png_path}")
    print(f"Saved figure PDF: {pdf_path}")
    print(f"Saved data CSV  : {csv_path}")


if __name__ == "__main__":
    main()
