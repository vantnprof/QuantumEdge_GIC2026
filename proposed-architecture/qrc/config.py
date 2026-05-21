"""Reservoir configurations and pipeline constants."""

from dataclasses import dataclass
import numpy as np

ANGLE_RANGE = (-np.pi / 2, np.pi / 2)
N_QUBITS = 9
N_ENT_FEATURES = 8
RIDGE_ALPHAS = np.logspace(-6, 6, 25)
TRAIN_FRAC = 0.80


@dataclass(frozen=True)
class ReservoirConfig:
    """Fixed-Hamiltonian Ising reservoir parameters.

    feature_cols: which financial feature columns this reservoir encodes.
    They are assigned cyclically across qubits: qubit i encodes
    feature_cols[i % len(feature_cols)].
    """
    name: str
    n_qubits: int
    coupling_j: float
    transverse_h: float
    trotter_depth: int
    feature_cols: tuple


# Short-memory reservoir: fast/daily financial features
SHORT_RESERVOIR = ReservoirConfig(
    name="short-memory",
    n_qubits=N_QUBITS,
    coupling_j=0.3,
    transverse_h=1.0,
    trotter_depth=2,
    feature_cols=("rv_d", "log_ret", "gk"),
)

# Long-memory reservoir: slow/multi-day financial features
LONG_RESERVOIR = ReservoirConfig(
    name="long-memory",
    n_qubits=N_QUBITS,
    coupling_j=1.2,
    transverse_h=0.4,
    trotter_depth=6,
    feature_cols=("rv_w", "rv_m", "vix", "vix_rv_spread"),
)

ALL_FEATURE_COLS = list(SHORT_RESERVOIR.feature_cols) + list(LONG_RESERVOIR.feature_cols)
