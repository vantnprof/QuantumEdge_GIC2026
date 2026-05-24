"""Shared benchmark defaults for scripts and report generation."""

from __future__ import annotations

from dataclasses import dataclass

from quantumedge.quantum.reservoirs import DEFAULT_N_QUBITS


@dataclass(frozen=True)
class ClassicalBenchmarkConfig:
    dataset: str = "sp500_rv"
    write_plots: bool = False
    metrics_filename: str = "metrics.csv"


@dataclass(frozen=True)
class QuantumBenchmarkConfig:
    dataset: str = "sp500_rv"
    backend: str = "statevector"
    n_qubits: int = DEFAULT_N_QUBITS
    n_train: int = 3200
    n_test: int = 800
    device_name: str | None = None
    progress_every: int = 250
    short_j: float = 0.3
    short_h: float = 1.0
    short_depth: int = 2
    long_j: float = 1.2
    long_h: float = 0.4
    long_depth: int = 6


@dataclass(frozen=True)
class ReportFigureConfig:
    dataset: str = "sp500_rv"
    output_dir: str = "artifacts/results/figures"
    basename: str = "quantumedge_run_comparison"
    reported_basename: str = "reported_summary_figure"


CLASSICAL_BENCHMARK = ClassicalBenchmarkConfig()
QUANTUM_BENCHMARK = QuantumBenchmarkConfig()
REPORT_FIGURE = ReportFigureConfig()


DATASET_LABELS = {
    "sp500_rv": "S&P 500 realized variance",
    "oxford_man_rv": "Oxford-Man realized variance",
    "vix": "VIX",
    "mackey_glass": "Mackey-Glass",
    "lorenz": "Lorenz",
}


CLASSICAL_FAMILY = {
    "GARCH": "garch",
    "HAR-RV": "har",
    "ARIMA": "arima",
    "ESN": "esn",
    "LSTM": "lstm",
    "XGBoost": "xgboost",
}


REPORT_COLORS = {
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
