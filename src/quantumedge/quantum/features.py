"""Statevector QRC feature extraction."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from quantumedge.quantum.reservoirs import (
    DEFAULT_N_ENT_FEATURES,
    ReservoirConfig,
    apply_ising_reservoir_layers,
)


def choose_pennylane_device() -> str:
    """Use lightning.qubit when available, otherwise fall back to default.qubit."""
    import pennylane as qml

    for device_name in ("lightning.qubit", "default.qubit"):
        try:
            qml.device(device_name, wires=1)
            return device_name
        except Exception:
            continue
    raise RuntimeError("No compatible PennyLane statevector device is available.")


def make_statevector_qnode(
    config: ReservoirConfig,
    device_name: str | None = None,
) -> Callable[[float], np.ndarray]:
    """Create one fixed Ising-reservoir statevector QNode."""
    import pennylane as qml

    dev = qml.device(device_name or choose_pennylane_device(), wires=config.n_qubits)

    @qml.qnode(dev, interface="numpy")
    def reservoir_state(x_angle):
        apply_ising_reservoir_layers(x_angle, config)
        return qml.state()

    return reservoir_state


def pauli_z_features_from_state(statevector, n_qubits: int) -> np.ndarray:
    """Return `[<Z_i>, <Z_i Z_{i+1}>]` from a statevector."""
    probabilities = np.abs(np.asarray(statevector, dtype=np.complex128)) ** 2
    tensor = probabilities.reshape((2,) * n_qubits)

    single_z = []
    for wire in range(n_qubits):
        marginal = tensor.sum(axis=tuple(axis for axis in range(n_qubits) if axis != wire))
        single_z.append(marginal[0] - marginal[1])

    pair_zz = []
    for wire in range(n_qubits - 1):
        keep_axes = (wire, wire + 1)
        marginal = tensor.sum(axis=tuple(axis for axis in range(n_qubits) if axis not in keep_axes))
        pair_zz.append(marginal[0, 0] + marginal[1, 1] - marginal[0, 1] - marginal[1, 0])

    return np.asarray(single_z + pair_zz, dtype=float)


def entanglement_energies_from_state(
    statevector,
    n_qubits: int,
    n_keep: int = DEFAULT_N_ENT_FEATURES,
) -> np.ndarray:
    """Return half-chain entanglement energies `xi_k = -log(lambda_k)`."""
    n_left = n_qubits // 2
    n_right = n_qubits - n_left
    psi = np.asarray(statevector, dtype=np.complex128).reshape(2 ** n_left, 2 ** n_right)

    rho_left = psi @ psi.conj().T
    eigvals = np.linalg.eigvalsh(rho_left).real
    eigvals = np.sort(np.clip(eigvals, 1e-12, 1.0))[::-1]
    energies = -np.log(eigvals)

    if len(energies) < n_keep:
        energies = np.pad(energies, (0, n_keep - len(energies)), constant_values=0.0)
    return energies[:n_keep]


def extract_statevector_features(
    series_angles: np.ndarray,
    short_node: Callable[[float], np.ndarray],
    long_node: Callable[[float], np.ndarray],
    n_qubits: int,
    n_ent: int | None = None,
    label: str = "series",
    progress_every: int = 250,
) -> dict[str, np.ndarray]:
    """Drive both statevector reservoirs and collect QRC readout matrices."""
    n_ent = n_ent if n_ent is not None else min(2 ** (n_qubits // 2), 8)
    series_angles = np.asarray(series_angles, dtype=float)
    dual_pauli_rows = []
    dual_ent_rows = []
    single_long_ent_rows = []
    timings = np.zeros(len(series_angles) - 1, dtype=float)

    prev_x = float(series_angles[0])
    start = time.perf_counter()
    for t, x_t in enumerate(series_angles[:-1]):
        step_start = time.perf_counter()
        state_short = short_node(float(x_t))
        state_long = long_node(float(x_t))

        pauli_short = pauli_z_features_from_state(state_short, n_qubits)
        pauli_long = pauli_z_features_from_state(state_long, n_qubits)
        ent_short = entanglement_energies_from_state(state_short, n_qubits, n_ent)
        ent_long = entanglement_energies_from_state(state_long, n_qubits, n_ent)
        feedback = np.array([prev_x])

        dual_pauli_rows.append(np.concatenate([pauli_short, pauli_long, feedback]))
        dual_ent_rows.append(np.concatenate([pauli_short, pauli_long, ent_short, ent_long, feedback]))
        single_long_ent_rows.append(np.concatenate([pauli_long, ent_long, feedback]))
        timings[t] = time.perf_counter() - step_start

        if progress_every and ((t + 1) % progress_every == 0 or (t + 1) == len(series_angles) - 1):
            elapsed = time.perf_counter() - start
            print(f"{label:<15} {t + 1:4d}/{len(series_angles) - 1} feature rows in {elapsed:6.1f}s")

        prev_x = float(x_t)

    return {
        "dual_pauli": np.vstack(dual_pauli_rows),
        "dual_ent": np.vstack(dual_ent_rows),
        "single_long_ent": np.vstack(single_long_ent_rows),
        "target": series_angles[1:],
        "timings": timings,
    }
