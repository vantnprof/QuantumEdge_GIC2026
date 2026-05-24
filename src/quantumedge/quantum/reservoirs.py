"""Fixed transverse-field Ising reservoirs for QRC."""

from __future__ import annotations

from dataclasses import dataclass

SEED = 42
DEFAULT_N_QUBITS = 9
DEFAULT_N_TRAIN = 1_000
DEFAULT_N_TEST = 500
DEFAULT_N_TOTAL = DEFAULT_N_TRAIN + DEFAULT_N_TEST + 1
DEFAULT_N_ENT_FEATURES = min(2 ** (DEFAULT_N_QUBITS // 2), 8)


@dataclass(frozen=True)
class ReservoirConfig:
    """Fixed Ising-reservoir parameters used by one QRC branch."""

    name: str
    n_qubits: int
    coupling_j: float
    transverse_h: float
    trotter_depth: int


SHORT_RESERVOIR = ReservoirConfig(
    name="short-memory",
    n_qubits=DEFAULT_N_QUBITS,
    coupling_j=0.3,
    transverse_h=1.0,
    trotter_depth=2,
)

LONG_RESERVOIR = ReservoirConfig(
    name="long-memory",
    n_qubits=DEFAULT_N_QUBITS,
    coupling_j=1.2,
    transverse_h=0.4,
    trotter_depth=6,
)


def make_reservoir_pair(
    n_qubits: int = DEFAULT_N_QUBITS,
    short_coupling_j: float = 0.3,
    short_transverse_h: float = 1.0,
    short_trotter_depth: int = 2,
    long_coupling_j: float = 1.2,
    long_transverse_h: float = 0.4,
    long_trotter_depth: int = 6,
) -> tuple[ReservoirConfig, ReservoirConfig]:
    """Return the proposed dual-timescale reservoir configuration."""
    short = ReservoirConfig(
        name="short-memory",
        n_qubits=n_qubits,
        coupling_j=short_coupling_j,
        transverse_h=short_transverse_h,
        trotter_depth=short_trotter_depth,
    )
    long = ReservoirConfig(
        name="long-memory",
        n_qubits=n_qubits,
        coupling_j=long_coupling_j,
        transverse_h=long_transverse_h,
        trotter_depth=long_trotter_depth,
    )
    return short, long


def apply_ising_reservoir_layers(x_angle: float, config: ReservoirConfig) -> None:
    """Apply repeated RY encoding and transverse-field Ising Trotter layers."""
    import pennylane as qml

    for _ in range(config.trotter_depth):
        for wire in range(config.n_qubits):
            qml.RY(x_angle, wires=wire)

        for wire in range(config.n_qubits - 1):
            qml.IsingZZ(2.0 * config.coupling_j, wires=[wire, wire + 1])

        for wire in range(config.n_qubits):
            qml.RX(2.0 * config.transverse_h, wires=wire)
