"""Gaussian HMM volatility regime detection.

Fits a 3-state Gaussian HMM on log(RV) from the training set.
States are mapped to Low / Medium / High volatility by ascending mean RV.
Regime thresholds (midpoints between adjacent state means) are used to
classify any RV forecast into a regime without re-running the HMM.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM


REGIME_NAMES = {0: "Low", 1: "Medium", 2: "High"}
REGIME_COLORS = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}  # green / amber / red


def fit_hmm(financial_split: dict, n_states: int = 3, random_state: int = 42):
    """Fit Gaussian HMM on training log(RV).

    Returns
    -------
    model : fitted GaussianHMM
    state_map : dict  raw HMM state → regime index (0=Low, 1=Med, 2=High)
    thresholds : tuple  (low_mid_rv, mid_high_rv) RV decision boundaries
    """
    rv_train = financial_split["train"]["rv"].values
    log_rv = np.log(rv_train + 1e-10).reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
    )
    model.fit(log_rv)

    # Sort HMM states by ascending mean → Low < Medium < High
    means = model.means_.flatten()
    sorted_states = np.argsort(means)
    state_map = {int(sorted_states[i]): i for i in range(n_states)}

    # Decision boundaries: midpoints between adjacent state means in RV space
    sorted_means_rv = np.exp(np.sort(means))
    thresholds = (
        float((sorted_means_rv[0] + sorted_means_rv[1]) / 2),
        float((sorted_means_rv[1] + sorted_means_rv[2]) / 2),
    )

    return model, state_map, thresholds


def predict_regimes(model, state_map: dict, rv_series: np.ndarray) -> np.ndarray:
    """Run HMM Viterbi on a RV series and map to regime indices."""
    log_rv = np.log(np.asarray(rv_series, dtype=float) + 1e-10).reshape(-1, 1)
    raw_states = model.predict(log_rv)
    return np.array([state_map[int(s)] for s in raw_states])


def classify_by_threshold(rv_pred: np.ndarray, thresholds: tuple) -> np.ndarray:
    """Classify RV forecasts into regimes using fixed RV thresholds.

    regime = 0 (Low)    if rv < thresholds[0]
    regime = 1 (Medium) if thresholds[0] <= rv < thresholds[1]
    regime = 2 (High)   if rv >= thresholds[1]
    """
    low_t, high_t = thresholds
    rv = np.asarray(rv_pred, dtype=float)
    out = np.zeros(len(rv), dtype=int)
    out[rv >= low_t] = 1
    out[rv >= high_t] = 2
    return out
