"""Ridge regression readouts: standard and regime-conditioned."""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RIDGE_ALPHAS, TRAIN_FRAC


def _make_ridge(alphas=RIDGE_ALPHAS):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))


def fit_ridge_variants(
    feature_dict: dict,
    n_train: int,
    n_test: int,
    regime_labels_train: np.ndarray = None,
    regime_posteriors_test: np.ndarray = None,
) -> dict:
    """
    Fit all readout variants from a feature_dict produced by
    extract_financial_features.

    Returns a dict keyed by variant name, each with:
        y_pred     : (n_test,) angle-space predictions
        rmse_angle : RMSE in normalised angle space
        alpha      : selected ridge regularisation parameter
        model      : fitted sklearn Pipeline (for inversion)
    """
    results = {}

    for key in ("dual_pauli", "dual_ent", "dual_ent_regime", "single_long_ent"):
        if key not in feature_dict:
            continue

        X = feature_dict[key]
        y = feature_dict["target_angle"]

        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:n_train + n_test], y[n_train:n_train + n_test]

        model = _make_ridge()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        alpha = float(model.named_steps["ridgecv"].alpha_)

        results[key] = {
            "y_pred": y_pred,
            "y_test": y_test,
            "rmse_angle": rmse,
            "alpha": alpha,
            "model": model,
        }

    # Regime-conditioned readout: per-regime Ridge with soft posterior blending
    if (regime_labels_train is not None and "dual_ent" in feature_dict):
        rc = RegimeReadout()
        X = feature_dict["dual_ent"]
        y = feature_dict["target_angle"]
        X_train, y_train = X[:n_train], y[:n_train]
        X_test = X[n_train:n_train + n_test]
        y_test = y[n_train:n_train + n_test]

        rc.fit(X_train, y_train, regime_labels_train)
        y_pred = rc.predict(X_test, regime_posteriors_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        results["regime_conditioned"] = {
            "y_pred": y_pred,
            "y_test": y_test,
            "rmse_angle": rmse,
            "alpha": None,
            "model": rc,
        }

    return results


class RegimeReadout:
    """
    Per-regime Ridge readouts with soft posterior blending at inference.

    Training: one Ridge per HMM regime, using only training samples from
    that regime. Falls back to a global Ridge if a regime has < 10 samples.

    Inference: weighted average of per-regime predictions using HMM
    posteriors P(regime | observations), shape (n_test, n_regimes).
    """

    def __init__(self, n_regimes: int = 3, alphas=RIDGE_ALPHAS):
        self.n_regimes = n_regimes
        self.alphas = alphas
        self.models_: dict = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            regime_labels: np.ndarray):
        global_model = _make_ridge(self.alphas)
        global_model.fit(X_train, y_train)

        for r in range(self.n_regimes):
            mask = regime_labels == r
            if mask.sum() < 10:
                self.models_[r] = global_model
            else:
                m = _make_ridge(self.alphas)
                m.fit(X_train[mask], y_train[mask])
                self.models_[r] = m
        return self

    def predict(self, X_test: np.ndarray,
                regime_posteriors: np.ndarray) -> np.ndarray:
        """
        Soft-blend per-regime predictions.
        regime_posteriors : (n_test, n_regimes), rows sum to 1.
        """
        preds = np.column_stack([
            self.models_[r].predict(X_test) for r in range(self.n_regimes)
        ])
        return (preds * regime_posteriors).sum(axis=1)


def invert_rv_predictions(y_angle: np.ndarray, rv_scaler) -> np.ndarray:
    """
    Convert normalised angle predictions back to RV space.
    rv_scaler is the MinMaxScaler fit on the rv_d training column.
    """
    clipped = np.clip(y_angle, rv_scaler.feature_range[0],
                      rv_scaler.feature_range[1])
    return rv_scaler.inverse_transform(clipped.reshape(-1, 1)).ravel()
