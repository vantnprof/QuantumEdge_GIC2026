"""Quantum Reservoir Computing components."""

from quantumedge.quantum.reservoirs import (
    DEFAULT_N_QUBITS,
    DEFAULT_N_TEST,
    DEFAULT_N_TRAIN,
    DEFAULT_N_TOTAL,
    LONG_RESERVOIR,
    SHORT_RESERVOIR,
    ReservoirConfig,
    make_reservoir_pair,
)

__all__ = [
    "DEFAULT_N_QUBITS",
    "DEFAULT_N_TEST",
    "DEFAULT_N_TRAIN",
    "DEFAULT_N_TOTAL",
    "LONG_RESERVOIR",
    "SHORT_RESERVOIR",
    "ReservoirConfig",
    "make_reservoir_pair",
]
