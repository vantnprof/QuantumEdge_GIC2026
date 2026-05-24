"""Classical readout heads for fixed QRC feature maps."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-6, 6, 25)


def train_ridge_readout(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_test: int,
    alphas: np.ndarray = RIDGE_ALPHAS,
) -> dict:
    """Fit a standardized RidgeCV readout and evaluate one-step predictions."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:n_train + n_test]
    y_test = y[n_train:n_train + n_test]

    if len(y_test) != n_test:
        raise ValueError(f"Expected {n_test} test targets, got {len(y_test)}")

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ridge = model.named_steps["ridgecv"]

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "y_test": y_test,
        "y_pred": y_pred,
        "alpha": float(ridge.alpha_),
        "model": model,
    }


def percent_improvement(baseline_rmse: float, proposed_rmse: float) -> float:
    """Percent RMSE reduction of proposed relative to baseline."""
    return float(100.0 * (baseline_rmse - proposed_rmse) / baseline_rmse)


def summarize_paper_metrics(experiments: dict[str, dict[str, dict]]) -> dict[str, dict[str, float]]:
    """Compute the Phase 2 paper's QRC comparison metrics where available."""
    out = {}
    for dataset, models in experiments.items():
        if "QRC dual Pauli" not in models:
            continue
        pauli_rmse = models["QRC dual Pauli"]["rmse"]
        row = {"pauli_rmse": pauli_rmse}

        if "QRC dual Pauli+Ent" in models:
            ent_rmse = models["QRC dual Pauli+Ent"]["rmse"]
            row["ent_rmse"] = ent_rmse
            row["entanglement_improvement_pct"] = percent_improvement(pauli_rmse, ent_rmse)

        single_key = "QRC single long Pauli+Ent"
        if single_key in models and "ent_rmse" in row:
            single_rmse = models[single_key]["rmse"]
            row["single_rmse"] = single_rmse
            row["dual_vs_single_gain_pct"] = percent_improvement(single_rmse, row["ent_rmse"])

        out[dataset] = row
    return out
