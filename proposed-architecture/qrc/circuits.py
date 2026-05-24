"""PennyLane QNode factory for multi-feature Ising reservoirs."""

import numpy as np
import pennylane as qml

from .config import ReservoirConfig


def _choose_device() -> str:
    for name in ("lightning.qubit", "default.qubit"):
        try:
            qml.device(name, wires=1)
            return name
        except Exception:
            continue
    raise RuntimeError("No PennyLane statevector device available.")


DEVICE_NAME = _choose_device()


def make_reservoir_qnode(config: ReservoirConfig, device_name: str = DEVICE_NAME):
    """
    Build a statevector QNode for one fixed Ising reservoir.

    Encoding: feature_angles[i % n_features] is applied as RY on qubit i,
    repeated before every Trotter layer. This gives each feature O(depth)
    injection opportunities and encodes all n_features into the entanglement
    structure rather than broadcasting one scalar to all qubits.

    Gate sequence per Trotter layer:
        1. RY(feature_angles[i % n_f]) on each qubit i   — data injection
        2. IsingZZ(2J) on each (i, i+1) pair             — ZZ coupling
        3. RX(2h) on each qubit                           — transverse field

    Parameters
    ----------
    config : ReservoirConfig
        Hamiltonian parameters and feature column assignment.
    device_name : str
        PennyLane device; defaults to lightning.qubit when available.

    Returns
    -------
    Callable: (feature_angles: np.ndarray of shape (n_features,)) -> statevector
    """
    dev = qml.device(device_name, wires=config.n_qubits)
    n_feats = len(config.feature_cols)

    @qml.qnode(dev, interface="numpy")
    def reservoir(feature_angles):
        for _ in range(config.trotter_depth):
            for wire in range(config.n_qubits):
                qml.RY(float(feature_angles[wire % n_feats]), wires=wire)
            for wire in range(config.n_qubits - 1):
                qml.IsingZZ(2.0 * config.coupling_j, wires=[wire, wire + 1])
            for wire in range(config.n_qubits):
                qml.RX(2.0 * config.transverse_h, wires=wire)
        return qml.state()

    return reservoir


def make_hardware_pauli_qnode(config: ReservoirConfig, device) -> object:
    """
    Hardware-compatible QNode returning Pauli expectations only.
    Identical gate sequence to make_reservoir_qnode; returns
    n_qubits <Z_i> + (n_qubits-1) <Z_i Z_{i+1}> as a flat array.
    Use with qiskit.remote device for EstimatorV2 dispatch.
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
        pair_zz = [
            qml.expval(qml.PauliZ(w) @ qml.PauliZ(w + 1))
            for w in range(config.n_qubits - 1)
        ]
        return single_z + pair_zz

    return reservoir_pauli
