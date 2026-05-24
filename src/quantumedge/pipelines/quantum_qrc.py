"""Run the proposed dual-timescale Quantum Reservoir Computing pipeline."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantumedge.config import ensure_runtime_dirs
from quantumedge.quantum.features import extract_statevector_features, make_statevector_qnode
from quantumedge.quantum.hardware import (
    connect_ibm_service,
    extract_hardware_features,
    get_least_busy_backend,
    make_aer_device,
    make_hardware_qnode,
    make_remote_device,
    save_features,
)
from quantumedge.quantum.readouts import summarize_paper_metrics, train_ridge_readout
from quantumedge.quantum.reservoirs import DEFAULT_N_QUBITS, DEFAULT_N_TEST, DEFAULT_N_TRAIN, make_reservoir_pair
from quantumedge.quantum.series import build_angle_series

DATASET_LABELS = {
    "mackey_glass": "Mackey-Glass",
    "lorenz": "Lorenz",
    "sp500_rv": "SP500 RV",
    "oxford_man_rv": "Oxford-Man RV",
    "vix": "VIX",
}


def _ts(label: str) -> None:
    print(f"\n{'='*60}\n{label}\n{'='*60}")


def _normalise_dataset_name(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def _default_split(backend: str, n_train: int | None, n_test: int | None) -> tuple[int, int]:
    if n_train is not None and n_test is not None:
        return n_train, n_test

    if backend == "statevector":
        return n_train or DEFAULT_N_TRAIN, n_test or DEFAULT_N_TEST

    # Hardware-shaped runs should be intentionally small unless the caller opts in.
    return n_train or 40, n_test or 10


def _safe_key(value: str) -> str:
    return value.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")


def _extract_features_for_dataset(
    dataset: str,
    backend: str,
    angles: np.ndarray,
    n_qubits: int,
    short_cfg,
    long_cfg,
    device_name: str | None,
    shots: int,
    progress_every: int,
    ibm_token: str | None,
    ibm_instance: str | None,
    ibm_backend_name: str | None,
) -> tuple[dict[str, np.ndarray], str]:
    if backend == "statevector":
        short_node = make_statevector_qnode(short_cfg, device_name=device_name)
        long_node = make_statevector_qnode(long_cfg, device_name=device_name)
        features = extract_statevector_features(
            angles,
            short_node,
            long_node,
            n_qubits=n_qubits,
            label=dataset,
            progress_every=progress_every,
        )
        return features, device_name or "pennylane-statevector"

    if backend == "aer":
        device = make_aer_device(n_qubits=n_qubits, shots=shots)
        short_node = make_hardware_qnode(short_cfg, device)
        long_node = make_hardware_qnode(long_cfg, device)
        features = extract_hardware_features(
            angles,
            short_node,
            long_node,
            label=dataset,
            progress_every=progress_every,
        )
        return features, "qiskit-aer"

    if backend == "ibm":
        token = ibm_token or os.getenv("QE_IBM_TOKEN")
        instance = ibm_instance or os.getenv("QE_IBM_INSTANCE")
        service = connect_ibm_service(token=token, instance=instance)
        qpu_backend = service.backend(ibm_backend_name) if ibm_backend_name else get_least_busy_backend(service, n_qubits)
        device = make_remote_device(qpu_backend, n_qubits=n_qubits, shots=shots)
        short_node = make_hardware_qnode(short_cfg, device)
        long_node = make_hardware_qnode(long_cfg, device)
        features = extract_hardware_features(
            angles,
            short_node,
            long_node,
            label=dataset,
            progress_every=progress_every,
        )
        return features, getattr(qpu_backend, "name", str(qpu_backend))

    raise ValueError(f"unsupported QRC backend: {backend}")


def _train_experiments(
    features: dict[str, np.ndarray],
    backend: str,
    n_train: int,
    n_test: int,
) -> dict[str, dict]:
    experiments = {
        "QRC dual Pauli": train_ridge_readout(
            features["dual_pauli"],
            features["target"],
            n_train=n_train,
            n_test=n_test,
        )
    }

    if backend == "statevector":
        experiments["QRC dual Pauli+Ent"] = train_ridge_readout(
            features["dual_ent"],
            features["target"],
            n_train=n_train,
            n_test=n_test,
        )
        experiments["QRC single long Pauli+Ent"] = train_ridge_readout(
            features["single_long_ent"],
            features["target"],
            n_train=n_train,
            n_test=n_test,
        )
    else:
        experiments["QRC single long Pauli"] = train_ridge_readout(
            features["single_long"],
            features["target"],
            n_train=n_train,
            n_test=n_test,
        )

    return experiments


def _records_for_dataset(
    dataset: str,
    backend: str,
    backend_detail: str,
    n_qubits: int,
    n_train: int,
    n_test: int,
    short_cfg,
    long_cfg,
    experiments: dict[str, dict],
) -> list[dict]:
    rows = []
    for model_name, result in experiments.items():
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "backend": backend,
                "backend_detail": backend_detail,
                "n_qubits": n_qubits,
                "n_train": n_train,
                "n_test": n_test,
                "short_j": short_cfg.coupling_j,
                "short_h": short_cfg.transverse_h,
                "short_depth": short_cfg.trotter_depth,
                "long_j": long_cfg.coupling_j,
                "long_h": long_cfg.transverse_h,
                "long_depth": long_cfg.trotter_depth,
                "rmse": result["rmse"],
                "mae": result["mae"],
                "ridge_alpha": result["alpha"],
            }
        )
    return rows


def _save_predictions(
    predictions: dict[str, dict[str, dict]],
    output_path: Path,
) -> Path:
    arrays = {}
    for dataset, experiments in predictions.items():
        for model_name, result in experiments.items():
            prefix = f"{_safe_key(dataset)}__{_safe_key(model_name)}"
            arrays[f"{prefix}__y_test"] = np.asarray(result["y_test"], dtype=float)
            arrays[f"{prefix}__y_pred"] = np.asarray(result["y_pred"], dtype=float)
    np.savez_compressed(output_path, **arrays)
    return output_path


def _plot_forecasts(
    predictions: dict[str, dict[str, dict]],
    output_path: Path,
    n_show: int = 150,
) -> Path:
    import matplotlib.pyplot as plt

    n_datasets = len(predictions)
    fig, axes = plt.subplots(n_datasets, 1, figsize=(13, 3.4 * n_datasets), sharex=False)
    if n_datasets == 1:
        axes = [axes]

    colors = {
        "QRC dual Pauli": "#4c78a8",
        "QRC dual Pauli+Ent": "#b35c00",
        "QRC single long Pauli+Ent": "#6b8e23",
        "QRC single long Pauli": "#6b8e23",
    }

    for ax, (dataset, experiments) in zip(axes, predictions.items()):
        first = next(iter(experiments.values()))
        n = min(n_show, len(first["y_test"]))
        ax.plot(first["y_test"][:n], color="#111111", lw=1.4, label="target")
        for model_name, result in experiments.items():
            ax.plot(
                result["y_pred"][:n],
                lw=1.1,
                color=colors.get(model_name, "tab:blue"),
                label=f"{model_name} RMSE={result['rmse']:.4f}",
            )
        ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} QRC one-step forecast")
        ax.set_ylabel("normalized angle")
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("test time step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_quantum_qrc(
    datasets: list[str],
    backend: str = "statevector",
    n_qubits: int = DEFAULT_N_QUBITS,
    short_j: float = 0.3,
    short_h: float = 1.0,
    short_depth: int = 2,
    long_j: float = 1.2,
    long_h: float = 0.4,
    long_depth: int = 6,
    n_train: int | None = None,
    n_test: int | None = None,
    device_name: str | None = None,
    shots: int = 4096,
    progress_every: int = 250,
    write_plots: bool = True,
    output_prefix: str = "quantum_qrc",
    ibm_token: str | None = None,
    ibm_instance: str | None = None,
    ibm_backend_name: str | None = None,
) -> pd.DataFrame:
    """Run the proposed dual-timescale QRC experiment."""
    settings = ensure_runtime_dirs()
    n_train, n_test = _default_split(backend, n_train, n_test)
    datasets = [_normalise_dataset_name(ds) for ds in datasets]
    short_cfg, long_cfg = make_reservoir_pair(
        n_qubits=n_qubits,
        short_coupling_j=short_j,
        short_transverse_h=short_h,
        short_trotter_depth=short_depth,
        long_coupling_j=long_j,
        long_transverse_h=long_h,
        long_trotter_depth=long_depth,
    )

    _ts("Quantum QRC configuration")
    print(f"  backend       : {backend}")
    print(f"  datasets      : {', '.join(datasets)}")
    print(f"  split         : train={n_train}, test={n_test}")
    print(
        "  reservoirs    : "
        f"n={n_qubits}, short J={short_cfg.coupling_j}, h={short_cfg.transverse_h}, p={short_cfg.trotter_depth}; "
        f"long J={long_cfg.coupling_j}, h={long_cfg.transverse_h}, p={long_cfg.trotter_depth}"
    )

    all_records = []
    all_predictions: dict[str, dict[str, dict]] = {}
    paper_inputs: dict[str, dict[str, dict]] = {}
    t0 = time.time()

    for dataset in datasets:
        _ts(f"Dataset: {dataset}")
        angles, _ = build_angle_series(dataset, n_train=n_train, n_test=n_test)
        features, backend_detail = _extract_features_for_dataset(
            dataset=dataset,
            backend=backend,
            angles=angles,
            n_qubits=n_qubits,
            short_cfg=short_cfg,
            long_cfg=long_cfg,
            device_name=device_name,
            shots=shots,
            progress_every=progress_every,
            ibm_token=ibm_token,
            ibm_instance=ibm_instance,
            ibm_backend_name=ibm_backend_name,
        )

        feature_path = settings.results_dir / f"{output_prefix}_{dataset}_{backend}_features.npz"
        save_features(
            features,
            feature_path,
            metadata={
                "dataset": dataset,
                "backend": backend,
                "backend_detail": backend_detail,
                "n_qubits": n_qubits,
                "n_train": n_train,
                "n_test": n_test,
                "short_reservoir": short_cfg.__dict__,
                "long_reservoir": long_cfg.__dict__,
            },
        )
        print(f"  features      : {feature_path}")

        experiments = _train_experiments(features, backend=backend, n_train=n_train, n_test=n_test)
        for model_name, result in experiments.items():
            print(f"  {model_name:<28} RMSE={result['rmse']:.6f}  alpha={result['alpha']:.2e}")

        all_records.extend(
            _records_for_dataset(
                dataset,
                backend,
                backend_detail,
                n_qubits,
                n_train,
                n_test,
                short_cfg,
                long_cfg,
                experiments,
            )
        )
        all_predictions[dataset] = experiments
        paper_inputs[DATASET_LABELS.get(dataset, dataset)] = experiments

    metrics_df = pd.DataFrame(all_records)
    metrics_path = settings.results_dir / f"{output_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")

    paper_metrics = summarize_paper_metrics(paper_inputs)
    if paper_metrics:
        paper_df = pd.DataFrame.from_dict(paper_metrics, orient="index")
        paper_path = settings.results_dir / f"{output_prefix}_paper_metrics.csv"
        paper_df.to_csv(paper_path, index_label="dataset")
        print(f"Paper metrics saved to: {paper_path}")

    predictions_path = settings.results_dir / f"{output_prefix}_predictions.npz"
    _save_predictions(all_predictions, predictions_path)
    print(f"Predictions saved to: {predictions_path}")

    if write_plots:
        plot_path = settings.plots_dir / f"{output_prefix}_forecasts.png"
        _plot_forecasts(all_predictions, plot_path)
        print(f"Forecast plot saved to: {plot_path}")

    print(f"\nTotal quantum QRC time: {time.time() - t0:.1f}s")
    return metrics_df


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Run the proposed dual-timescale QRC pipeline.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mackey_glass", "lorenz"],
        choices=["mackey_glass", "lorenz", "sp500_rv", "oxford_man_rv", "vix"],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--backend",
        choices=["statevector", "aer", "ibm"],
        default="statevector",
        help="Quantum execution backend.",
    )
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_N_QUBITS)
    parser.add_argument("--short-j", type=float, default=0.3)
    parser.add_argument("--short-h", type=float, default=1.0)
    parser.add_argument("--short-depth", type=int, default=2)
    parser.add_argument("--long-j", type=float, default=1.2)
    parser.add_argument("--long-h", type=float, default=0.4)
    parser.add_argument("--long-depth", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--device-name", default=None, help="PennyLane statevector device override.")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--output-prefix", default="quantum_qrc")
    parser.add_argument("--ibm-token", default=None)
    parser.add_argument("--ibm-instance", default=None)
    parser.add_argument("--ibm-backend", default=None, help="Explicit IBM backend name.")
    args = parser.parse_args(argv)

    return run_quantum_qrc(
        datasets=args.datasets,
        backend=args.backend,
        n_qubits=args.n_qubits,
        short_j=args.short_j,
        short_h=args.short_h,
        short_depth=args.short_depth,
        long_j=args.long_j,
        long_h=args.long_h,
        long_depth=args.long_depth,
        n_train=args.n_train,
        n_test=args.n_test,
        device_name=args.device_name,
        shots=args.shots,
        progress_every=args.progress_every,
        write_plots=not args.skip_plots,
        output_prefix=args.output_prefix,
        ibm_token=args.ibm_token,
        ibm_instance=args.ibm_instance,
        ibm_backend_name=args.ibm_backend,
    )


if __name__ == "__main__":
    main()
