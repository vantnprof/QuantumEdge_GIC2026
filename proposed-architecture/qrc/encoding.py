"""Multi-feature angle encoding: per-feature MinMaxScaler with cyclic qubit assignment."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .config import ANGLE_RANGE


class MultiFeatureEncoder:
    """
    Encodes a financial feature DataFrame into per-qubit rotation angles.

    Each feature column gets its own MinMaxScaler, fit on the training portion
    only to prevent look-ahead bias. Qubits are assigned features cyclically:
    qubit i encodes feature_cols[i % n_features].

    This makes HAR decomposition structurally explicit in the quantum layer:
    - Short reservoir (rv_d, log_ret, gk)      → fast-timescale entanglement
    - Long reservoir  (rv_w, rv_m, vix, spread) → slow-timescale entanglement
    """

    def __init__(self, feature_cols, angle_range=ANGLE_RANGE):
        self.feature_cols = list(feature_cols)
        self.angle_range = angle_range
        self.scalers_: dict = {}

    def fit(self, df_train):
        """Fit one MinMaxScaler per feature column on training data."""
        for col in self.feature_cols:
            s = MinMaxScaler(feature_range=self.angle_range)
            s.fit(df_train[[col]])
            self.scalers_[col] = s
        return self

    def transform(self, df) -> np.ndarray:
        """Return angle matrix of shape (T, n_features), clipped to angle_range."""
        out = np.zeros((len(df), len(self.feature_cols)))
        for j, col in enumerate(self.feature_cols):
            vals = self.scalers_[col].transform(df[[col]]).ravel()
            out[:, j] = np.clip(vals, *self.angle_range)
        return out

    def inverse_transform_col(self, col: str, angles: np.ndarray) -> np.ndarray:
        """Invert angle → original feature scale for a single column."""
        return self.scalers_[col].inverse_transform(
            angles.reshape(-1, 1)
        ).ravel()

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)
