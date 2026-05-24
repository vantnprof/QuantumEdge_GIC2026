"""
qrc_financial_hardware.py
=========================

Hardware-compatible companion to ``QuantumEdge_QRC_Financial.ipynb``.

Implements the dual-timescale multi-feature QRC using **measurement-based
Pauli observables** so the same reservoirs can run on IBM Quantum hardware
via the ``pennylane-qiskit`` ``qiskit.remote`` device (EstimatorV2).

Design decisions for hardware runs:
  * Multi-feature cyclic encoding is preserved exactly.
  * Pauli-Z and ZZ expectations only — no statevector, no entanglement spectrum.
  * Entanglement-spectrum features require RDM eigenvalues and are deferred to
    Phase 3 via classical shadow tomography.
  * HMM regime posteriors are computed classically (no quantum cost).
  * N_HW_STEPS default: 50 (stays well within Open Plan 10-min quota).

Team QuantumEdge, GIC 2026.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Reservoir configs (must match the main notebook exactly)
# ---------------------------------------------------------------------------

SEED = 42
N_QUBITS = 9
ANGLE_RANGE = (-np.pi / 2, np.pi / 2)


@dataclass(frozen=True)
class ReservoirConfig:
    name: str
    n_qubits: int
    coupling_j: float
    transverse_h: float
    trotter_depth: int
    feature_cols: tuple


SHORT_RESERVOIR = ReservoirConfig(
    name="short-memory",
    n_qubits=N_QUBITS,
    coupling_j=0.3,
    transverse_h=1.0,
    trotter_depth=2,
    feature_cols=("rv_d", "log_ret", "gk"),
)

LONG_RESERVOIR = ReservoirConfig(
    name="long-memory",
    n_qubits=N_QUBITS,
    coupling_j=1.2,
    transverse_h=0.4,
    trotter_depth=6,
    feature_cols=("rv_w", "rv_m", "vix", "vix_rv_spread"),
)

SHORT_COLS = list(SHORT_RESERVOIR.feature_cols)
LONG_COLS  = list(LONG_RESERVOIR.feature_cols)


# ---------------------------------------------------------------------------
# Feature encoding (identical to main notebook)
# ---------------------------------------------------------------------------

def fit_encoder(df_train: pd.DataFrame, feature_cols: list) -> dict:
    """Fit per-feature MinMaxScaler on training data. No look-ahead."""
    scalers = {}
    for col in feature_cols:
        s = MinMaxScaler(feature_range=ANGLE_RANGE)
        s.fit(df_train[[col]])
        scalers[col] = s
    return scalers


def encode_series(df: pd.DataFrame, feature_cols: list, scalers: dict) -> np.ndarray:
    """Return angle matrix of shape (T, n_features), clipped to ANGLE_RANGE."""
    out = np.zeros((len(df), len(feature_cols)))
    for j, col in enumerate(feature_cols):
        vals = scalers[col].transform(df[[col]]).ravel()
        out[:, j] = np.clip(vals, *ANGLE_RANGE)
    return out


# ---------------------------------------------------------------------------
# Hardware-compatible QNode factory
# ---------------------------------------------------------------------------

def make_hardware_qnode(config: ReservoirConfig, device) -> Callable:
    """
    QNode returning Pauli expectations as a flat array (hardware-compatible).

    Returns 2*n_qubits - 1 values:
      n_qubits <Z_i> + (n_qubits-1) <Z_i Z_{i+1}>

    Dispatched as a single EstimatorV2 PUB per time step on IBM hardware,
    meaning one network round-trip per reservoir per step.
    """
    n_feats = len(config.feature_cols)

    @qml.qnode(device, interface="numpy")
    def reservoir_pauli(feature_angles):
        for _ in range(config.trotter_depth):
            for wire in range(config.n_qubits):
                qml.RY(float(feature_angles[wire % n_feats]), wires=wire)
            for wire in range(config.n_qubits - 1):
                qml.IsingZZ(2.0 * config.coupling_j, wires=[wire, wire + 1])
            for wire in range(config.n_qubits):
                qml.RX(2.0 * config.transverse_h, wires=wire)
        single_z = [qml.expval(qml.PauliZ(w)) for w in range(config.n_qubits)]
        pair_zz  = [qml.expval(qml.PauliZ(w) @ qml.PauliZ(w + 1))
                    for w in range(config.n_qubits - 1)]
        return single_z + pair_zz

    return reservoir_pauli


def pauli_vector(qnode_output) -> np.ndarray:
    return np.asarray(qnode_output, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def make_aer_device(n_qubits: int = N_QUBITS, shots: int = 4096,
                    optimization_level: int = 1):
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise ImportError(
            "qiskit-aer required for dry-run. "
            "Install with: pip install qiskit-aer"
        ) from exc
    backend = AerSimulator()
    return qml.device("qiskit.remote", wires=n_qubits, backend=backend,
                       shots=shots, optimization_level=optimization_level)


def connect_ibm_service(token: str | None = None, instance: str | None = None,
                         channel: str = "ibm_quantum_platform"):
    from qiskit_ibm_runtime import QiskitRuntimeService
    if token is not None:
        return QiskitRuntimeService(channel=channel, token=token, instance=instance)
    return QiskitRuntimeService()


def get_least_busy_backend(service, n_qubits: int = N_QUBITS,
                            prefer_names: Sequence[str] = (
                                "ibm_kingston", "ibm_fez",
                                "ibm_marrakesh", "ibm_brisbane")):
    """Preferred-name backend selection with least_busy fallback."""
    backend_map = {b.name: b for b in service.backends(
        operational=True, simulator=False, min_num_qubits=n_qubits)}
    for name in prefer_names:
        if name in backend_map:
            return backend_map[name]
    return service.least_busy(operational=True, simulator=False,
                               min_num_qubits=n_qubits)


def make_remote_device(backend, n_qubits: int = N_QUBITS, shots: int = 4096,
                        optimization_level: int = 2, resilience_level: int = 1):
    return qml.device("qiskit.remote", wires=n_qubits, backend=backend,
                       shots=shots, optimization_level=optimization_level,
                       resilience_level=resilience_level)


# ---------------------------------------------------------------------------
# Hardware feature extraction (Pauli-only, no entanglement)
# ---------------------------------------------------------------------------

def extract_hardware_features(
    short_angles: np.ndarray,
    long_angles: np.ndarray,
    short_qnode: Callable,
    long_qnode: Callable,
    label: str = "series",
    progress_every: int = 5,
) -> dict:
    """
    Drive both reservoirs over a slice of financial angle matrices.

    short_angles, long_angles : (N_HW_STEPS+1, n_features) arrays
    Returns dict with:
        dual_pauli   (T, 2*(2n-1) + 1)  short Pauli || long Pauli || prev_rv
        single_long  (T, (2n-1) + 1)    long Pauli || prev_rv  (ablation)
        target_angle (T,)               next-step rv_d angle
        timings      (T,)               wall-clock seconds per step
    """
    T = short_angles.shape[0] - 1
    dual_rows, single_rows = [], []
    timings = np.zeros(T, dtype=float)
    t0 = time.perf_counter()

    for t in range(T):
        step_start = time.perf_counter()
        pau_s = pauli_vector(short_qnode(short_angles[t]))
        pau_l = pauli_vector(long_qnode(long_angles[t]))
        prev_rv = np.array([float(short_angles[t, 0])])

        dual_rows.append(np.concatenate([pau_s, pau_l, prev_rv]))
        single_rows.append(np.concatenate([pau_l, prev_rv]))
        timings[t] = time.perf_counter() - step_start

        if progress_every and ((t + 1) % progress_every == 0 or t + 1 == T):
            elapsed = time.perf_counter() - t0
            eta = elapsed / (t + 1) * (T - t - 1)
            print(f"{label:<15} {t+1:4d}/{T}  elapsed {elapsed:6.1f}s  "
                  f"eta {eta:6.1f}s  last step {timings[t]:.2f}s")

    return {
        "dual_pauli":   np.vstack(dual_rows),
        "single_long":  np.vstack(single_rows),
        "target_angle": short_angles[1:, 0],
        "timings":      timings,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_features(features: dict, path: str | Path,
                  metadata: dict | None = None) -> Path:
    path = Path(path).with_suffix(".npz")
    np.savez_compressed(path, **{k: v for k, v in features.items()
                                  if isinstance(v, np.ndarray)})
    if metadata is not None:
        Path(path).with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, default=str))
    return path


def load_features(path: str | Path) -> dict:
    blob = np.load(Path(path).with_suffix(".npz"))
    return {k: blob[k] for k in blob.files}


__all__ = [
    "ReservoirConfig", "SHORT_RESERVOIR", "LONG_RESERVOIR",
    "SHORT_COLS", "LONG_COLS", "SEED", "N_QUBITS", "ANGLE_RANGE",
    "fit_encoder", "encode_series",
    "make_hardware_qnode", "pauli_vector",
    "make_aer_device", "connect_ibm_service",
    "get_least_busy_backend", "make_remote_device",
    "extract_hardware_features",
    "save_features", "load_features",
]
