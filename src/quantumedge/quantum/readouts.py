"""Classical readout heads for fixed QRC feature maps."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-6, 6, 25)
QLIKE_LAMBDAS = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], dtype=float)


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


class RegimeReadout:
    """Per-regime Ridge readouts with soft posterior blending."""

    def __init__(self, n_regimes: int = 3, alphas: np.ndarray = RIDGE_ALPHAS):
        self.n_regimes = n_regimes
        self.alphas = alphas
        self.models_: dict[int, object] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, regime_labels: np.ndarray) -> "RegimeReadout":
        global_model = make_pipeline(StandardScaler(), RidgeCV(alphas=self.alphas))
        global_model.fit(X_train, y_train)

        for regime in range(self.n_regimes):
            mask = np.asarray(regime_labels) == regime
            if int(mask.sum()) < 10:
                self.models_[regime] = global_model
            else:
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=self.alphas))
                model.fit(X_train[mask], y_train[mask])
                self.models_[regime] = model
        return self

    def predict(self, X_test: np.ndarray, regime_posteriors: np.ndarray) -> np.ndarray:
        preds = np.column_stack([self.models_[regime].predict(X_test) for regime in range(self.n_regimes)])
        return (preds * regime_posteriors).sum(axis=1)


def train_regime_readout(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_test: int,
    regime_labels_train: np.ndarray,
    regime_posteriors_test: np.ndarray,
    alphas: np.ndarray = RIDGE_ALPHAS,
) -> dict:
    """Fit per-regime Ridge readouts and blend predictions by HMM posteriors."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:n_train + n_test]
    y_test = y[n_train:n_train + n_test]

    model = RegimeReadout(alphas=alphas)
    model.fit(X_train, y_train, regime_labels_train)
    y_pred = model.predict(X_test, regime_posteriors_test)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "y_test": y_test,
        "y_pred": y_pred,
        "alpha": np.nan,
        "model": model,
    }


def train_qlike_readout(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_test: int,
    angle_scaler,
    lambdas: np.ndarray = QLIKE_LAMBDAS,
    alphas: np.ndarray = RIDGE_ALPHAS,
) -> dict:
    """Fit a linear readout by minimizing QLIKE on RV-space predictions.

    Lambda is selected on the last 20% of the training window only.
    """
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_train_raw = X[:n_train]
    y_train = y[:n_train]
    X_test_raw = X[n_train:n_train + n_test]
    y_test = y[n_train:n_train + n_test]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    log_rv_scale = 1.0 / float(angle_scaler.scale_[0])

    def rv_from_angles(angle_values: np.ndarray) -> np.ndarray:
        log_rv = angle_scaler.inverse_transform(np.asarray(angle_values).reshape(-1, 1)).ravel()
        return np.exp(log_rv)

    def loss_and_grad(weights: np.ndarray, X_aug: np.ndarray, y_rv: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
        pred_angles = X_aug @ weights
        y_pred = np.maximum(rv_from_angles(pred_angles), 1e-12)
        ratio = y_rv / y_pred
        loss = float(np.mean(ratio - np.log(ratio) - 1.0)) + lam * float(np.dot(weights[:-1], weights[:-1]))
        err = 1.0 - ratio
        grad = log_rv_scale * (X_aug.T @ err) / len(y_rv)
        grad[:-1] += 2.0 * lam * weights[:-1]
        return loss, grad

    def ridge_weights(X_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(X_fit, y_fit)
        return np.append(ridge.coef_, ridge.intercept_)

    if n_train < 20:
        n_val = max(1, n_train // 4)
    else:
        n_val = max(10, int(n_train * 0.20))
    n_val = min(n_val, n_train - 1)
    n_fit = n_train - n_val

    X_fit = X_train[:n_fit]
    X_val = X_train[n_fit:]
    y_fit = y_train[:n_fit]
    y_val = y_train[n_fit:]
    X_fit_aug = np.c_[X_fit, np.ones(n_fit)]
    X_val_aug = np.c_[X_val, np.ones(n_val)]
    y_fit_rv = rv_from_angles(y_fit)
    y_val_rv = rv_from_angles(y_val)

    initial_fit = ridge_weights(X_fit, y_fit)
    best_lambda = float(lambdas[0])
    best_qlike = np.inf
    for lam in lambdas:
        result = minimize(
            loss_and_grad,
            initial_fit,
            args=(X_fit_aug, y_fit_rv, float(lam)),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
        val_pred = np.maximum(rv_from_angles(X_val_aug @ result.x), 1e-12)
        ratio = y_val_rv / val_pred
        val_qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
        if val_qlike < best_qlike:
            best_qlike = val_qlike
            best_lambda = float(lam)

    X_train_aug = np.c_[X_train, np.ones(n_train)]
    X_test_aug = np.c_[X_test, np.ones(n_test)]
    initial_full = ridge_weights(X_train, y_train)
    result = minimize(
        loss_and_grad,
        initial_full,
        args=(X_train_aug, rv_from_angles(y_train), best_lambda),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    y_pred = X_test_aug @ result.x

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "y_test": y_test,
        "y_pred": y_pred,
        "alpha": best_lambda,
        "model": None,
        "validation_qlike": best_qlike,
        "optimizer_success": bool(result.success),
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
