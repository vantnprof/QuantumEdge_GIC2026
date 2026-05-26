"""Run the MNIST QRC expressivity benchmark.

This script keeps the notebook benchmark reproducible from the command line:

  python scripts/run_mnist_benchmark.py
  python scripts/run_mnist_benchmark.py --quick-smoke

The full default uses OpenML MNIST with a stratified 10k/2k subset and a
5/10/15-qubit sweep. The smoke mode uses sklearn's small digits dataset and is
intended only to verify the experiment path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantumedge.config import ensure_runtime_dirs
from quantumedge.quantum.features import (
    choose_pennylane_device,
    entanglement_energies_from_state,
    make_statevector_qnode,
    pauli_z_features_from_state,
)
from quantumedge.quantum.reservoirs import ReservoirConfig


ANGLE_RANGE = (-np.pi / 2.0, np.pi / 2.0)
RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def _safe_token(value: object) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def _allocate_class_counts(total: int, classes: np.ndarray) -> dict[int, int]:
    base = total // len(classes)
    rem = total % len(classes)
    return {int(label): base + (idx < rem) for idx, label in enumerate(classes)}


def _stratified_sample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    counts = _allocate_class_counts(n_samples, classes)
    selected = []
    for label in classes:
        pool = np.flatnonzero(y == label)
        take = counts[int(label)]
        if take > len(pool):
            raise ValueError(f"Requested {take} samples for class {label}, only {len(pool)} available.")
        selected.append(rng.choice(pool, take, replace=False))
    idx = np.concatenate(selected)
    rng.shuffle(idx)
    return X[idx], y[idx]


def load_openml_mnist(
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    print("Loading MNIST from OpenML. First run may download the dataset.")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = np.asarray(mnist.data, dtype=np.float32) / 255.0
    y = np.asarray(mnist.target, dtype=int)

    rng = np.random.default_rng(seed)
    X_train, y_train = _stratified_sample(X[:60_000], y[:60_000], n_train, rng)
    X_test, y_test = _stratified_sample(X[60_000:], y[60_000:], n_test, rng)
    return X_train, y_train, X_test, y_test, "mnist_openml"


def load_digits_smoke(
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    digits = load_digits()
    X = np.asarray(digits.data, dtype=np.float32) / 16.0
    y = np.asarray(digits.target, dtype=int)
    if n_train + n_test > len(X):
        raise ValueError(f"digits smoke source has only {len(X)} samples.")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=n_train,
        test_size=n_test,
        random_state=seed,
        stratify=y,
    )
    return X_train, y_train, X_test, y_test, "digits_smoke"


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if args.source == "openml":
        return load_openml_mnist(args.n_train, args.n_test, args.seed)
    if args.source == "digits":
        return load_digits_smoke(args.n_train, args.n_test, args.seed)
    raise ValueError(f"Unsupported source: {args.source}")


def fit_angle_encoder(X_train: np.ndarray, n_components: int, seed: int) -> tuple[PCA, MinMaxScaler, float]:
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    train_components = pca.fit_transform(X_train)
    scaler = MinMaxScaler(feature_range=ANGLE_RANGE)
    scaler.fit(train_components)
    explained = float(pca.explained_variance_ratio_.sum())
    return pca, scaler, explained


def encode_images(X: np.ndarray, pca: PCA, scaler: MinMaxScaler) -> np.ndarray:
    angles = scaler.transform(pca.transform(X))
    return np.clip(angles, ANGLE_RANGE[0], ANGLE_RANGE[1])


def fit_classifier(X_train: np.ndarray, y_train: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        RidgeClassifierCV(alphas=RIDGE_ALPHAS),
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float]:
    y_pred = model.predict(X_test)
    ridge = model.named_steps["ridgeclassifiercv"]
    alpha = float(getattr(ridge, "alpha_", np.nan))
    return (
        float(accuracy_score(y_test, y_pred)),
        float(balanced_accuracy_score(y_test, y_pred)),
        alpha,
    )


def extract_qrc_features(
    angles: np.ndarray,
    qnode,
    n_qubits: int,
    n_ent: int,
    label: str,
    progress_every: int,
) -> dict[str, np.ndarray]:
    pauli_rows = []
    ent_rows = []
    start = time.perf_counter()
    for idx, row in enumerate(angles, start=1):
        state = qnode(row)
        pauli_rows.append(pauli_z_features_from_state(state, n_qubits))
        ent_rows.append(entanglement_energies_from_state(state, n_qubits, n_keep=n_ent))
        if progress_every and (idx % progress_every == 0 or idx == len(angles)):
            elapsed = time.perf_counter() - start
            print(f"  {label:<14} {idx:5d}/{len(angles)} feature rows in {elapsed:7.1f}s")

    pauli = np.vstack(pauli_rows)
    ent = np.vstack(ent_rows)
    return {
        "pauli_only": pauli,
        "pauli_ent": np.hstack([pauli, ent]),
    }


def cache_path(settings, args: argparse.Namespace, dataset_id: str, n_qubits: int) -> Path:
    name = "_".join(
        [
            args.output_prefix,
            dataset_id,
            f"train{args.n_train}",
            f"test{args.n_test}",
            f"{n_qubits}q",
            f"J{_safe_token(args.coupling_j)}",
            f"h{_safe_token(args.transverse_h)}",
            f"p{args.trotter_depth}",
            f"ent{args.n_ent}",
            f"seed{args.seed}",
            "features.npz",
        ]
    )
    return settings.results_dir / "mnist" / name


def load_or_extract_features(
    settings,
    args: argparse.Namespace,
    dataset_id: str,
    n_qubits: int,
    A_train: np.ndarray,
    A_test: np.ndarray,
    device_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    path = cache_path(settings, args, dataset_id, n_qubits)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.no_cache:
        data = np.load(path)
        print(f"  features cache: {path}")
        return (
            {"pauli_only": data["train_pauli_only"], "pauli_ent": data["train_pauli_ent"]},
            {"pauli_only": data["test_pauli_only"], "pauli_ent": data["test_pauli_ent"]},
            "cache",
        )

    cfg = ReservoirConfig(
        name=f"mnist-{n_qubits}q",
        n_qubits=n_qubits,
        coupling_j=args.coupling_j,
        transverse_h=args.transverse_h,
        trotter_depth=args.trotter_depth,
    )
    qnode = make_statevector_qnode(cfg, device_name=device_name)

    print(f"  extracting train features ({len(A_train)} images)")
    train_features = extract_qrc_features(
        A_train,
        qnode,
        n_qubits=n_qubits,
        n_ent=args.n_ent,
        label=f"train-{n_qubits}q",
        progress_every=args.progress_every,
    )
    print(f"  extracting test features ({len(A_test)} images)")
    test_features = extract_qrc_features(
        A_test,
        qnode,
        n_qubits=n_qubits,
        n_ent=args.n_ent,
        label=f"test-{n_qubits}q",
        progress_every=args.progress_every,
    )

    np.savez_compressed(
        path,
        train_pauli_only=train_features["pauli_only"],
        train_pauli_ent=train_features["pauli_ent"],
        test_pauli_only=test_features["pauli_only"],
        test_pauli_ent=test_features["pauli_ent"],
    )
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "n_train": args.n_train,
                "n_test": args.n_test,
                "n_qubits": n_qubits,
                "coupling_j": args.coupling_j,
                "transverse_h": args.transverse_h,
                "trotter_depth": args.trotter_depth,
                "n_ent": args.n_ent,
                "seed": args.seed,
                "device_name": device_name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  features saved: {path}")
    return train_features, test_features, "computed"


def plot_results(df: pd.DataFrame, settings, output_prefix: str) -> list[Path]:
    import matplotlib.pyplot as plt

    plot_df = df.sort_values("n_qubits")
    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.6))

    ax.bar(x - width, plot_df["pca_ridge_acc"] * 100, width=width, label="PCA+Ridge", color="#9fb8d0")
    ax.bar(x, plot_df["qrc_pauli_acc"] * 100, width=width, label="QRC Pauli", color="#2878a8")
    ax.bar(x + width, plot_df["qrc_pauli_ent_acc"] * 100, width=width, label="QRC Pauli+Ent", color="#2ca58d")

    for xpos, (_, row) in zip(x + width, plot_df.iterrows()):
        gain = row["qrc_pauli_ent_gain_pp"]
        ax.annotate(
            f"{gain:+.1f} pp",
            xy=(xpos, row["qrc_pauli_ent_acc"] * 100),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={int(n)}" for n in plot_df["n_qubits"]])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("QRC MNIST Expressivity Benchmark")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    settings.plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        settings.plots_dir / f"{output_prefix}_qubit_sweep.png",
        settings.plots_dir / f"{output_prefix}_qubit_sweep.pdf",
    ]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return outputs


def run(args: argparse.Namespace) -> pd.DataFrame:
    settings = ensure_runtime_dirs()
    device_name = args.device_name or choose_pennylane_device()

    X_train, y_train, X_test, y_test, dataset_id = load_dataset(args)
    print("MNIST/QRC benchmark configuration")
    print(f"  source        : {dataset_id}")
    print(f"  split         : train={len(X_train)}, test={len(X_test)}")
    print(f"  qubits        : {args.qubits}")
    print(f"  reservoir     : J={args.coupling_j}, h={args.transverse_h}, p={args.trotter_depth}")
    print(f"  device        : {device_name}")

    rows = []
    for n_qubits in args.qubits:
        print(f"\n{'=' * 60}\n{n_qubits} qubits\n{'=' * 60}")
        started = time.perf_counter()

        pca, scaler, explained = fit_angle_encoder(X_train, n_qubits, args.seed)
        A_train = encode_images(X_train, pca, scaler)
        A_test = encode_images(X_test, pca, scaler)
        print(f"  PCA variance explained: {explained * 100:.2f}%")

        pca_model = fit_classifier(A_train, y_train)
        pca_acc, pca_bal_acc, pca_alpha = evaluate_classifier(pca_model, A_test, y_test)
        print(f"  PCA+Ridge             accuracy={pca_acc:.4f}  balanced={pca_bal_acc:.4f}")

        train_features, test_features, feature_source = load_or_extract_features(
            settings,
            args,
            dataset_id,
            n_qubits,
            A_train,
            A_test,
            device_name,
        )

        pauli_model = fit_classifier(train_features["pauli_only"], y_train)
        pauli_acc, pauli_bal_acc, pauli_alpha = evaluate_classifier(
            pauli_model,
            test_features["pauli_only"],
            y_test,
        )

        ent_model = fit_classifier(train_features["pauli_ent"], y_train)
        ent_acc, ent_bal_acc, ent_alpha = evaluate_classifier(
            ent_model,
            test_features["pauli_ent"],
            y_test,
        )

        elapsed = time.perf_counter() - started
        print(f"  QRC Pauli             accuracy={pauli_acc:.4f}  balanced={pauli_bal_acc:.4f}")
        print(f"  QRC Pauli+Ent         accuracy={ent_acc:.4f}  balanced={ent_bal_acc:.4f}")
        print(f"  elapsed               {elapsed:.1f}s")

        rows.append(
            {
                "dataset": dataset_id,
                "n_qubits": n_qubits,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "pca_components": n_qubits,
                "pca_variance": explained,
                "reservoir_j": args.coupling_j,
                "reservoir_h": args.transverse_h,
                "trotter_depth": args.trotter_depth,
                "n_pauli_features": 2 * n_qubits - 1,
                "n_ent_features": args.n_ent,
                "n_pauli_ent_features": train_features["pauli_ent"].shape[1],
                "pca_ridge_acc": pca_acc,
                "pca_ridge_balanced_acc": pca_bal_acc,
                "pca_ridge_alpha": pca_alpha,
                "qrc_pauli_acc": pauli_acc,
                "qrc_pauli_balanced_acc": pauli_bal_acc,
                "qrc_pauli_alpha": pauli_alpha,
                "qrc_pauli_ent_acc": ent_acc,
                "qrc_pauli_ent_balanced_acc": ent_bal_acc,
                "qrc_pauli_ent_alpha": ent_alpha,
                "qrc_pauli_ent_gain_pp": 100.0 * (ent_acc - pca_acc),
                "feature_source": feature_source,
                "elapsed_s": elapsed,
                "device_name": device_name,
                "seed": args.seed,
            }
        )

    df = pd.DataFrame(rows)
    output_dir = settings.results_dir / "mnist"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}_qubit_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")
    print(df.to_string(index=False))

    if not args.skip_plots:
        for plot_path in plot_results(df, settings, args.output_prefix):
            print(f"Plot saved   : {plot_path}")

    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QRC MNIST expressivity benchmark.")
    parser.add_argument("--source", choices=["openml", "digits"], default="openml")
    parser.add_argument("--qubits", nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-test", type=int, default=2_000)
    parser.add_argument("--coupling-j", type=float, default=0.8)
    parser.add_argument("--transverse-h", type=float, default=0.5)
    parser.add_argument("--trotter-depth", type=int, default=4)
    parser.add_argument("--n-ent", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-name", default=None, help="PennyLane statevector device override.")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--output-prefix", default="mnist_qrc")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Use the sklearn digits smoke dataset with a tiny split and n=5 only.",
    )
    args = parser.parse_args(argv)

    if args.quick_smoke:
        args.source = "digits"
        args.qubits = [5]
        args.n_train = 120
        args.n_test = 40
        args.progress_every = min(args.progress_every, 50)
        args.output_prefix = "mnist_qrc_smoke"

    if args.n_train <= 0 or args.n_test <= 0:
        raise ValueError("--n-train and --n-test must be positive.")
    if any(n < 2 for n in args.qubits):
        raise ValueError("All qubit counts must be at least 2.")
    return args


def main(argv: list[str] | None = None) -> pd.DataFrame:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
