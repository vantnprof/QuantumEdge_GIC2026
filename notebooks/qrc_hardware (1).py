"""
qrc_hardware.py
================

Hardware-compatible companion to ``QuantumEdge_QRC_Prototype.ipynb``.

The main notebook uses ``qml.state()`` and reduced-density-matrix readouts,
which are simulator-only. This module re-implements the dual-timescale
QRC architecture using **measurement-based observables** so the same
reservoirs can run on IBM Quantum hardware through the
``pennylane-qiskit`` ``qiskit.remote`` device (EstimatorV2 primitive).

Scope decisions for hardware runs:

  * Pauli-Z and ZZ expectation values only.
  * Entanglement-spectrum features are dropped — they require state
    tomography on the half chain (3**(n/2) bases per time step), which
    is not feasible inside an Open-Plan quantum-time budget.
  * Time-series length is a runtime parameter (``N_HW_STEPS``), with a
    recommended default of 50–100 for a hardware validation panel.

Team QuantumEdge, GIC 2026 Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
import json
import time

import numpy as np
import pennylane as qml
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Reservoir configuration (mirrors the main notebook so results align)
# ---------------------------------------------------------------------------

SEED = 42
N_QUBITS = 9
N_TRAIN = 1_000
N_TEST = 500
N_TOTAL = N_TRAIN + N_TEST + 1


@dataclass(frozen=True)
class ReservoirConfig:
    """Fixed Ising-reservoir parameters used by one branch of the dual QRC."""

    name: str
    n_qubits: int
    coupling_j: float
    transverse_h: float
    trotter_depth: int


SHORT_RESERVOIR = ReservoirConfig(
    name="short-memory",
    n_qubits=N_QUBITS,
    coupling_j=0.3,
    transverse_h=1.0,
    trotter_depth=2,
)

LONG_RESERVOIR = ReservoirConfig(
    name="long-memory",
    n_qubits=N_QUBITS,
    coupling_j=1.2,
    transverse_h=0.4,
    trotter_depth=6,
)


# ---------------------------------------------------------------------------
# Time-series generators (verbatim from the main notebook; reproduce same data)
# ---------------------------------------------------------------------------

def mackey_glass(n_steps: int, tau: int = 17, beta: float = 0.2,
                 gamma: float = 0.1, exponent: int = 10,
                 dt: float = 1.0, x0: float = 1.2) -> np.ndarray:
    """Mackey-Glass delay-differential series via Euler stepping."""
    delay = int(tau / dt)
    history = delay + 1
    x = np.empty(n_steps + history, dtype=float)
    x[:history] = x0
    for t in range(history, n_steps + history):
        x_tau = x[t - delay]
        dx = beta * x_tau / (1.0 + x_tau ** exponent) - gamma * x[t - 1]
        x[t] = x[t - 1] + dt * dx
    return x[history:]


def lorenz_x_series(n_steps: int, sigma: float = 10.0, rho: float = 28.0,
                    beta: float = 8.0 / 3.0, dt: float = 0.02) -> np.ndarray:
    """Lorenz attractor, x-coordinate, integrated with RK45."""
    def rhs(_t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t_eval = np.arange(n_steps) * dt
    solution = solve_ivp(
        rhs, t_span=(t_eval[0], t_eval[-1]),
        y0=[1.0, 1.0, 1.0], t_eval=t_eval,
        method="RK45", rtol=1e-8, atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Lorenz integration failed: {solution.message}")
    return solution.y[0]


def discard_transient(series: np.ndarray, transient_steps: int,
                      n_total: int = N_TOTAL) -> np.ndarray:
    usable = np.asarray(series[transient_steps:transient_steps + n_total], dtype=float)
    if len(usable) != n_total:
        raise ValueError(f"Expected {n_total} observations after transient, got {len(usable)}")
    return usable


def normalize_to_angles(raw_series: np.ndarray, n_train: int = N_TRAIN,
                        feature_range: tuple[float, float] = (-np.pi / 2, np.pi / 2)
                        ) -> tuple[np.ndarray, MinMaxScaler]:
    """Fit the angle scaler on the training segment only, then transform the whole series."""
    raw_series = np.asarray(raw_series, dtype=float)
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(raw_series[:n_train + 1].reshape(-1, 1))
    angles = scaler.transform(raw_series.reshape(-1, 1)).ravel()
    angles = np.clip(angles, feature_range[0], feature_range[1])
    return angles, scaler


def build_lorenz_angles() -> np.ndarray:
    """Convenience: full Lorenz angle series, identical to the main notebook."""
    np.random.seed(SEED)
    raw = discard_transient(lorenz_x_series(N_TOTAL + 300), transient_steps=300)
    angles, _ = normalize_to_angles(raw)
    return angles


def build_mackey_glass_angles() -> np.ndarray:
    """Convenience: full Mackey-Glass angle series, identical to the main notebook."""
    np.random.seed(SEED)
    raw = discard_transient(mackey_glass(N_TOTAL + 150), transient_steps=150)
    angles, _ = normalize_to_angles(raw)
    return angles


# ---------------------------------------------------------------------------
# Hardware-compatible QNode
# ---------------------------------------------------------------------------

def _ising_reservoir_layers(x_angle: float, config: ReservoirConfig) -> None:
    """The transverse-field Ising Trotter pattern used in both reservoirs.

    Identical gate sequence to the main notebook; only the readout differs.
    """
    for _ in range(config.trotter_depth):
        for wire in range(config.n_qubits):
            qml.RY(x_angle, wires=wire)
        for wire in range(config.n_qubits - 1):
            qml.IsingZZ(2.0 * config.coupling_j, wires=[wire, wire + 1])
        for wire in range(config.n_qubits):
            qml.RX(2.0 * config.transverse_h, wires=wire)


def make_hardware_qnode(config: ReservoirConfig, device: qml.devices.Device) -> Callable:
    """Build a QNode that returns Pauli features as expectation values.

    Returns 2n-1 expectations per call:
        - n single-qubit <Z_i>
        - n-1 nearest-neighbour <Z_i Z_{i+1}>

    On a ``qiskit.remote`` device this is dispatched as a single
    EstimatorV2 PUB (one network round-trip per time step), not one job
    per observable.
    """

    @qml.qnode(device, interface="numpy")
    def reservoir_pauli(x_angle):
        _ising_reservoir_layers(x_angle, config)
        single_z = [qml.expval(qml.PauliZ(w)) for w in range(config.n_qubits)]
        pair_zz = [
            qml.expval(qml.PauliZ(w) @ qml.PauliZ(w + 1))
            for w in range(config.n_qubits - 1)
        ]
        return single_z + pair_zz

    return reservoir_pauli


def pauli_vector_from_qnode_output(output) -> np.ndarray:
    """Flatten the QNode output (list of scalars or 1-D array) to a NumPy row."""
    return np.asarray(output, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Device construction helpers
# ---------------------------------------------------------------------------

def make_aer_device(n_qubits: int, shots: int = 4096,
                    optimization_level: int = 1) -> qml.devices.Device:
    """Local Aer simulator wrapped via ``qiskit.remote``.

    Useful for end-to-end testing of the hardware code path without
    consuming any IBM Quantum runtime.
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qiskit-aer is required for local hardware-shape testing. "
            "Install it with: pip install qiskit-aer"
        ) from exc

    backend = AerSimulator()
    return qml.device(
        "qiskit.remote",
        wires=n_qubits,
        backend=backend,
        shots=shots,
        optimization_level=optimization_level,
    )


def connect_ibm_service(token: str | None = None,
                        instance: str | None = None,
                        channel: str = "ibm_quantum_platform"):
    """Return a QiskitRuntimeService instance.

    If ``token`` and ``instance`` are provided, they are used directly
    (no save_account). Otherwise the saved default account is loaded.
    See ``QiskitRuntimeService.save_account`` to persist credentials once.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    if token is not None:
        return QiskitRuntimeService(channel=channel, token=token, instance=instance)
    return QiskitRuntimeService()


def get_least_busy_backend(service, n_qubits: int = N_QUBITS,
                           prefer_names: Sequence[str] = ("ibm_kingston",
                                                          "ibm_fez",
                                                          "ibm_marrakesh",
                                                          "ibm_brisbane")):
    """Return the least-busy operational backend with at least n_qubits.

    If any of ``prefer_names`` is operational, return it; otherwise fall
    back to ``service.least_busy(...)``. Heron r2 (``ibm_kingston``) is
    a good default for Open Plan since it has low error rates and is
    explicitly enabled for Open Plan users.
    """
    backend_names = {b.name: b for b in service.backends(operational=True,
                                                          simulator=False,
                                                          min_num_qubits=n_qubits)}
    for name in prefer_names:
        if name in backend_names:
            return backend_names[name]
    return service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)


def make_remote_device(backend, n_qubits: int = N_QUBITS,
                       shots: int = 4096,
                       optimization_level: int = 2,
                       resilience_level: int = 1) -> qml.devices.Device:
    """Wrap a Qiskit Runtime backend as a PennyLane ``qiskit.remote`` device.

    ``resilience_level=1`` enables readout error mitigation by default.
    ``optimization_level=2`` is a good compromise between transpile time
    and circuit depth on Heron-class processors.
    """
    return qml.device(
        "qiskit.remote",
        wires=n_qubits,
        backend=backend,
        shots=shots,
        optimization_level=optimization_level,
        resilience_level=resilience_level,
    )


# ---------------------------------------------------------------------------
# Feature extraction (hardware-aware)
# ---------------------------------------------------------------------------

def extract_hardware_features(angles: np.ndarray,
                              short_qnode: Callable,
                              long_qnode: Callable,
                              n_qubits: int = N_QUBITS,
                              label: str = "series",
                              progress_every: int = 10
                              ) -> dict[str, np.ndarray]:
    """Drive both reservoirs over an angle series; collect Pauli features.

    Each step submits two EstimatorV2 PUBs (one per reservoir), so on
    hardware the dominant cost is queueing + transpilation rather than
    QPU time per shot. Expect 0.5–5 s per (reservoir, time step) on a
    busy queue.

    Returns
    -------
    dict with keys:
        ``dual_pauli``      (T, 2*(2n-1)+1)  short Pauli || long Pauli || prev_x
        ``single_long``     (T, (2n-1)+1)    long Pauli || prev_x   (ablation)
        ``target``          (T,)             one-step-ahead targets
        ``timings``         (T,)             wall-clock seconds per step
    """
    angles = np.asarray(angles, dtype=float)
    n_steps = len(angles) - 1
    dual_rows: list[np.ndarray] = []
    single_long_rows: list[np.ndarray] = []
    timings = np.zeros(n_steps, dtype=float)

    prev_x = float(angles[0])
    t_start = time.perf_counter()

    for t, x_t in enumerate(angles[:-1]):
        step_start = time.perf_counter()
        pauli_short = pauli_vector_from_qnode_output(short_qnode(float(x_t)))
        pauli_long = pauli_vector_from_qnode_output(long_qnode(float(x_t)))
        feedback = np.array([prev_x])

        dual_rows.append(np.concatenate([pauli_short, pauli_long, feedback]))
        single_long_rows.append(np.concatenate([pauli_long, feedback]))

        timings[t] = time.perf_counter() - step_start
        prev_x = float(x_t)

        if progress_every and ((t + 1) % progress_every == 0 or (t + 1) == n_steps):
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (t + 1) * (n_steps - t - 1)
            print(f"{label:<13} {t + 1:4d}/{n_steps}  "
                  f"elapsed {elapsed:6.1f}s  eta {eta:6.1f}s  "
                  f"last step {timings[t]:.2f}s")

    return {
        "dual_pauli": np.vstack(dual_rows),
        "single_long": np.vstack(single_long_rows),
        "target": angles[1:],
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Persistence (don't re-run hardware extraction unnecessarily)
# ---------------------------------------------------------------------------

def save_features(features: dict[str, np.ndarray], path: str | Path,
                  metadata: dict | None = None) -> Path:
    """Save extracted features to .npz alongside a JSON metadata sidecar."""
    path = Path(path).with_suffix(".npz")
    np.savez_compressed(
        path,
        dual_pauli=features["dual_pauli"],
        single_long=features["single_long"],
        target=features["target"],
        timings=features["timings"],
    )
    if metadata is not None:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    return path


def load_features(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).with_suffix(".npz")
    blob = np.load(path)
    return {key: blob[key] for key in blob.files}


__all__ = [
    "ReservoirConfig", "SHORT_RESERVOIR", "LONG_RESERVOIR",
    "SEED", "N_QUBITS", "N_TRAIN", "N_TEST", "N_TOTAL",
    "mackey_glass", "lorenz_x_series", "discard_transient",
    "normalize_to_angles", "build_lorenz_angles", "build_mackey_glass_angles",
    "make_hardware_qnode", "pauli_vector_from_qnode_output",
    "make_aer_device", "connect_ibm_service",
    "get_least_busy_backend", "make_remote_device",
    "extract_hardware_features",
    "save_features", "load_features",
]
