"""Run Phase II methods and generate the comparison figure.

This file is intentionally runnable from the repository root without installing
the package first. By default it executes the classical baselines and proposed
QRC variants using the same shared configuration as the notebooks/report
scripts, then writes a PNG, PDF, and companion CSV from the actual run outputs.

Use --quick-smoke for a small QRC-only execution that verifies the environment.
Use --reported-only only to recreate a fixed-value presentation figure.
"""

from __future__ import annotations

import argparse
from importlib.util import find_spec
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quantumedge.config import ensure_runtime_dirs
from quantumedge.experiments.benchmark_config import DATASET_LABELS, QUANTUM_BENCHMARK, REPORT_FIGURE
from scripts.make_report_figure import (
    _normalise_dataset,
    build_actual_results,
    classical_rows,
    classical_split_lengths,
    make_figure,
    reported_results,
    run_classical_results,
    run_qrc_results,
    target_type_for_model,
    validate_fair_comparison,
)


MODULE_TO_REQUIREMENT = {
    "arch": "arch>=6.0",
    "hmmlearn": "hmmlearn>=0.3",
    "pennylane": "pennylane>=0.40",
    "reservoirpy": "reservoirpy>=0.4",
    "scipy": "scipy>=1.11",
    "sklearn": "scikit-learn>=1.3",
    "statsmodels": "statsmodels>=0.14",
    "torch": "torch>=2.0",
    "xgboost": "xgboost>=2.0",
    "yfinance": "yfinance>=0.2.0",
}
BASE_MODULES = {
    "arch",
    "hmmlearn",
    "reservoirpy",
    "scipy",
    "sklearn",
    "statsmodels",
    "torch",
    "xgboost",
    "yfinance",
}
QUANTUM_MODULES = {"pennylane"}
ALL_DATASETS = ("sp500_rv", "oxford_man_rv", "vix", "mackey_glass", "lorenz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase II classical and QRC methods, then generate a report "
            "figure from the metrics produced by this run."
        )
    )
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Run a tiny QRC-only Mackey-Glass job to verify the script quickly.",
    )
    parser.add_argument(
        "--reported-only",
        action="store_true",
        help="Do not run methods; recreate the fixed presentation summary values.",
    )
    parser.add_argument(
        "--dataset",
        default=REPORT_FIGURE.dataset,
        choices=["sp500_rv", "oxford_man_rv", "vix", "mackey_glass", "lorenz"],
        help="Dataset to benchmark. Default matches the Phase II report figure.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run all supported datasets and write a consolidated benchmark table.",
    )
    parser.add_argument("--run-classical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-quantum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--classical-csv",
        default=None,
        help="Use an existing classical metrics CSV instead of rerunning classical baselines.",
    )
    parser.add_argument(
        "--quantum-csv",
        default=None,
        help="Use an existing quantum metrics CSV instead of rerunning QRC variants.",
    )
    parser.add_argument(
        "--match-classical-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When classical rows are included, force QRC to use the same "
            "temporal train/test split for a fair comparison."
        ),
    )
    parser.add_argument(
        "--skip-classical-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip classical diagnostic plots while running the report comparison.",
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
    parser.add_argument("--basename", default="phaseII_result")
    return parser.parse_args()


def apply_quick_smoke_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.quick_smoke:
        return args

    args.dataset = "mackey_glass"
    args.run_classical = False
    args.run_quantum = True
    args.all_datasets = False
    args.classical_csv = None
    args.quantum_csv = None
    args.match_classical_split = False
    args.quantum_n_qubits = 3
    args.quantum_n_train = 8
    args.quantum_n_test = 4
    args.device_name = "default.qubit"
    args.progress_every = 0
    if args.basename == "phaseII_result":
        args.basename = "phaseII_result_smoke"
    return args


def output_dir_from_args(args: argparse.Namespace) -> Path:
    settings = ensure_runtime_dirs()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = settings.project_root / output_dir
    return output_dir


def _market_data_cache_present() -> bool:
    settings = ensure_runtime_dirs()
    return (settings.raw_data_dir / "sp500_ohlcv.csv").exists() and (settings.raw_data_dir / "vix.csv").exists()


def required_modules_for_run(args: argparse.Namespace) -> set[str]:
    if args.reported_only:
        return set()

    datasets = ALL_DATASETS if args.all_datasets else (_normalise_dataset(args.dataset),)
    modules = set()
    has_financial_dataset = any(dataset in {"sp500_rv", "oxford_man_rv", "vix"} for dataset in datasets)

    if args.run_classical:
        modules.update({"arch", "hmmlearn", "reservoirpy", "statsmodels", "torch", "xgboost"})

    if args.run_quantum:
        modules.update({"sklearn", "scipy", "pennylane"})

    if args.run_quantum and any(dataset in {"sp500_rv", "oxford_man_rv"} for dataset in datasets):
        modules.add("hmmlearn")

    needs_project_data = (
        args.run_classical
        or args.classical_csv is not None
        or (args.run_quantum and has_financial_dataset)
        or (args.match_classical_split and (args.run_classical or args.classical_csv is not None))
    )
    if needs_project_data and has_financial_dataset and not _market_data_cache_present():
        modules.add("yfinance")

    return modules


def check_required_modules(args: argparse.Namespace) -> None:
    missing = sorted(module for module in required_modules_for_run(args) if find_spec(module) is None)
    if not missing:
        return

    print("Missing Python packages for this Phase II run:")
    for module in missing:
        print(f"  - {module}  ({MODULE_TO_REQUIREMENT[module]})")
    print()
    print("Install the benchmark dependencies in your active environment:")
    if any(module in BASE_MODULES for module in missing):
        print("  python -m pip install -r requirements/base.txt")
    if any(module in QUANTUM_MODULES for module in missing):
        print("  python -m pip install -r requirements/quantum.txt")
    print()
    print("Then rerun:")
    print("  python phaseII_result.py")
    raise SystemExit(2)


def title_suffix_for_run(df, args: argparse.Namespace) -> str:
    dataset = _normalise_dataset(args.dataset)
    qrc_rows = df[df["source"] == "qrc"]
    if not qrc_rows.empty:
        qrc_train = int(qrc_rows["n_train"].iloc[0])
        qrc_test = int(qrc_rows["n_test"].iloc[0])
    else:
        qrc_train = args.quantum_n_train
        qrc_test = args.quantum_n_test

    return (
        f"{DATASET_LABELS.get(dataset, dataset)} | "
        f"QRC n={args.quantum_n_qubits}, train={qrc_train}, test={qrc_test}"
    )


def print_run_configuration(args: argparse.Namespace) -> None:
    print("Phase II benchmark configuration")
    dataset_label = ", ".join(ALL_DATASETS) if args.all_datasets else _normalise_dataset(args.dataset)
    print(f"  dataset              : {dataset_label}")
    print(f"  run classical        : {args.run_classical}")
    print(f"  run quantum          : {args.run_quantum}")
    print(f"  match classical split: {args.match_classical_split}")
    print(f"  qrc qubits           : {args.quantum_n_qubits}")
    print(
        "  short reservoir      : "
        f"J={args.short_j}, h={args.short_h}, depth={args.short_depth}"
    )
    print(
        "  long reservoir       : "
        f"J={args.long_j}, h={args.long_h}, depth={args.long_depth}"
    )
    print(f"  qrc requested split  : train={args.quantum_n_train}, test={args.quantum_n_test}")
    if args.classical_csv:
        print(f"  classical csv        : {args.classical_csv}")
    if args.quantum_csv:
        print(f"  quantum csv          : {args.quantum_csv}")


def result_table_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "dataset",
            "target_type",
            "model",
            "source",
            "n_train",
            "n_test",
            "rmse",
            "mae",
            "qlike",
            "regime_acc",
            "sharpe",
        ]
        if column in df.columns
    ]


def print_result_table(df: pd.DataFrame) -> None:
    print("Result table")
    sort_cols = [column for column in ["dataset", "target_type", "rmse"] if column in df.columns]
    print(df[result_table_columns(df)].sort_values(sort_cols).to_string(index=False))


def dataset_title_suffix(dataset: str, df: pd.DataFrame, args: argparse.Namespace) -> str:
    qrc_rows = df[df["source"] == "qrc"]
    if not qrc_rows.empty:
        qrc_train = int(qrc_rows["n_train"].iloc[0])
        qrc_test = int(qrc_rows["n_test"].iloc[0])
    else:
        qrc_train = args.quantum_n_train
        qrc_test = args.quantum_n_test
    return (
        f"{DATASET_LABELS.get(dataset, dataset)} | "
        f"QRC n={args.quantum_n_qubits}, train={qrc_train}, test={qrc_test}"
    )


def build_all_dataset_results(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = list(ALL_DATASETS)
    parts = []
    audit_rows = []

    classical_metrics = None
    if args.run_classical:
        classical_metrics = run_classical_results(write_plots=not args.skip_classical_plots)
    elif args.classical_csv:
        classical_metrics = pd.read_csv(args.classical_csv)

    quantum_metrics = pd.read_csv(args.quantum_csv) if args.quantum_csv else None

    for dataset in datasets:
        print(f"\n[all-datasets] Dataset: {dataset}")
        dataset_parts = []
        qrc_n_train = args.quantum_n_train
        qrc_n_test = args.quantum_n_test
        split_policy = "explicit_quantum_split"

        if args.match_classical_split and classical_metrics is not None:
            qrc_n_train, qrc_n_test = classical_split_lengths(dataset)
            split_policy = "matched_classical_temporal_split"
            print(
                "[fairness] Matching QRC split to classical temporal split: "
                f"train={qrc_n_train}, test={qrc_n_test}"
            )

        if classical_metrics is not None:
            dataset_parts.append(classical_rows(classical_metrics, dataset))

        if args.run_quantum:
            dataset_parts.append(run_qrc_results(args, dataset, qrc_n_train, qrc_n_test, split_policy))
        elif quantum_metrics is not None:
            quantum_rows = quantum_metrics[quantum_metrics["dataset"] == dataset].copy()
            if not quantum_rows.empty:
                if "source" not in quantum_rows.columns:
                    quantum_rows["source"] = "qrc"
                if "family" not in quantum_rows.columns:
                    quantum_rows["family"] = "qrc"
                if "target_type" not in quantum_rows.columns:
                    quantum_rows["target_type"] = quantum_rows["model"].map(
                        lambda model: target_type_for_model(dataset, model)
                    )
                dataset_parts.append(quantum_rows)

        dataset_parts = [part for part in dataset_parts if part is not None and not part.empty]
        if not dataset_parts:
            print(f"[all-datasets] No rows produced for {dataset}; skipping.")
            continue

        dataset_df = pd.concat(dataset_parts, ignore_index=True, sort=False)
        dataset_df["dataset"] = dataset
        if "target_type" not in dataset_df.columns:
            dataset_df["target_type"] = "unknown"
        validate_fair_comparison(dataset_df, dataset)
        parts.append(dataset_df)

        for target_type, group in dataset_df.groupby("target_type", dropna=False):
            split_pairs = sorted(
                {
                    tuple(pair)
                    for pair in group[["n_train", "n_test"]].dropna().astype(int).drop_duplicates().values
                }
            )
            audit_rows.append(
                {
                    "dataset": dataset,
                    "target_type": target_type,
                    "n_rows": len(group),
                    "sources": ",".join(sorted(group["source"].dropna().astype(str).unique())),
                    "split_pairs": ";".join(f"{train}/{test}" for train, test in split_pairs),
                    "fair_split": len(split_pairs) == 1,
                }
            )

    if not parts:
        raise RuntimeError("No all-dataset result rows were produced.")

    return pd.concat(parts, ignore_index=True, sort=False), pd.DataFrame(audit_rows)


def write_all_dataset_outputs(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    output_dir: Path,
    basename: str,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    settings = ensure_runtime_dirs()
    results_csv = settings.results_dir / f"{basename}_all_datasets_data.csv"
    audit_csv = settings.results_dir / f"{basename}_split_audit.csv"
    coverage_csv = settings.results_dir / f"{basename}_coverage.csv"
    figure_csv = output_dir / f"{basename}_all_datasets_data.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(results_csv, index=False)
    df.to_csv(figure_csv, index=False)
    audit_df.to_csv(audit_csv, index=False)
    pd.crosstab([df["dataset"], df["target_type"]], [df["source"], df["model"]]).to_csv(coverage_csv)

    for (dataset, target_type), group in df.groupby(["dataset", "target_type"], dropna=False):
        comparable = group.copy()
        if comparable["rmse"].notna().sum() < 2:
            continue
        safe_target = str(target_type).replace("/", "_").replace(" ", "_")
        figure_base = f"{basename}_{dataset}_{safe_target}"
        title_suffix = f"{dataset_title_suffix(dataset, comparable, args)} | target={target_type}"
        make_figure(comparable, output_dir, figure_base, title_suffix)

    return results_csv, audit_csv, coverage_csv


def main() -> None:
    args = apply_quick_smoke_defaults(parse_args())
    output_dir = output_dir_from_args(args)
    check_required_modules(args)

    if args.reported_only:
        df = reported_results()
        title_suffix = "reported summary values"
        print("Using fixed reported values; no methods are being executed.")
    elif args.all_datasets:
        print_run_configuration(args)
        df, audit_df = build_all_dataset_results(args)
        results_csv, audit_csv, coverage_csv = write_all_dataset_outputs(df, audit_df, output_dir, args.basename, args)
        print(f"Saved all-dataset data CSV : {results_csv}")
        print(f"Saved split audit CSV      : {audit_csv}")
        print(f"Saved coverage CSV         : {coverage_csv}")
        print()
        print_result_table(df)
        print()
        print("Split audit")
        print(audit_df.to_string(index=False))
        return
    else:
        print_run_configuration(args)
        df = build_actual_results(args)
        title_suffix = title_suffix_for_run(df, args)

    png_path, pdf_path, csv_path = make_figure(df, output_dir, args.basename, title_suffix)
    print(f"Saved figure PNG: {png_path}")
    print(f"Saved figure PDF: {pdf_path}")
    print(f"Saved data CSV  : {csv_path}")
    print()
    print_result_table(df)


if __name__ == "__main__":
    main()
