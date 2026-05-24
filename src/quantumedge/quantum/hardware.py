"""Hardware-compatible QRC feature extraction through PennyLane/Qiskit."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from quantumedge.quantum.reservoirs import DEFAULT_N_QUBITS, ReservoirConfig, apply_ising_reservoir_layers


def make_hardware_qnode(config: ReservoirConfig, device) -> Callable[[float], np.ndarray]:
    """Build a QNode that returns Pauli and nearest-neighbor ZZ expectations."""
    import pennylane as qml

    @qml.qnode(device, interface="numpy")
    def reservoir_pauli(x_angle):
        apply_ising_reservoir_layers(x_angle, config)
        single_z = [qml.expval(qml.PauliZ(w)) for w in range(config.n_qubits)]
        pair_zz = [
            qml.expval(qml.PauliZ(w) @ qml.PauliZ(w + 1))
            for w in range(config.n_qubits - 1)
        ]
        return single_z + pair_zz

    return reservoir_pauli


def pauli_vector_from_qnode_output(output) -> np.ndarray:
    """Flatten hardware QNode output to a numeric feature row."""
    return np.asarray(output, dtype=float).ravel()


def make_aer_device(
    n_qubits: int,
    shots: int = 4096,
    optimization_level: int = 1,
):
    """Local Aer simulator wrapped as a PennyLane `qiskit.remote` device."""
    import pennylane as qml

    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qiskit-aer is required for local hardware-shape testing. "
            "Install it with: pip install -r requirements/quantum.txt"
        ) from exc

    backend = AerSimulator()
    return qml.device(
        "qiskit.remote",
        wires=n_qubits,
        backend=backend,
        shots=shots,
        optimization_level=optimization_level,
    )


def connect_ibm_service(
    token: str | None = None,
    instance: str | None = None,
    channel: str = "ibm_quantum_platform",
):
    """Return a QiskitRuntimeService from explicit or saved credentials."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    if token is not None:
        return QiskitRuntimeService(channel=channel, token=token, instance=instance)
    return QiskitRuntimeService()


def get_least_busy_backend(
    service,
    n_qubits: int = DEFAULT_N_QUBITS,
    prefer_names: Sequence[str] = ("ibm_kingston", "ibm_fez", "ibm_marrakesh", "ibm_brisbane"),
):
    """Return a preferred operational backend, otherwise the least-busy backend."""
    backend_names = {
        b.name: b
        for b in service.backends(
            operational=True,
            simulator=False,
            min_num_qubits=n_qubits,
        )
    }
    for name in prefer_names:
        if name in backend_names:
            return backend_names[name]
    return service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)


def make_remote_device(
    backend,
    n_qubits: int = DEFAULT_N_QUBITS,
    shots: int = 4096,
    optimization_level: int = 2,
    resilience_level: int = 1,
):
    """Wrap a Qiskit Runtime backend as a PennyLane `qiskit.remote` device."""
    import pennylane as qml

    return qml.device(
        "qiskit.remote",
        wires=n_qubits,
        backend=backend,
        shots=shots,
        optimization_level=optimization_level,
        resilience_level=resilience_level,
    )


def extract_hardware_features(
    angles: np.ndarray,
    short_qnode: Callable[[float], np.ndarray],
    long_qnode: Callable[[float], np.ndarray],
    label: str = "series",
    progress_every: int = 10,
) -> dict[str, np.ndarray]:
    """Drive both hardware-compatible reservoirs and collect Pauli features."""
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
            print(f"{label:<15} {t + 1:4d}/{n_steps}  "
                  f"elapsed {elapsed:6.1f}s  eta {eta:6.1f}s  "
                  f"last step {timings[t]:.2f}s")

    return {
        "dual_pauli": np.vstack(dual_rows),
        "single_long": np.vstack(single_long_rows),
        "target": angles[1:],
        "timings": timings,
    }


def save_features(
    features: dict[str, np.ndarray],
    path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Save extracted QRC feature matrices to `.npz` plus optional JSON metadata."""
    path = Path(path).with_suffix(".npz")
    np.savez_compressed(path, **features)
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return path


def load_features(path: str | Path) -> dict[str, np.ndarray]:
    """Load QRC feature matrices saved by `save_features`."""
    blob = np.load(Path(path).with_suffix(".npz"))
    return {key: blob[key] for key in blob.files}
