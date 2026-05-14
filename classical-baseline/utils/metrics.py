"""RMSE, QLIKE, and MAE loss functions."""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE loss (Patton, 2011). Only meaningful for strictly positive targets.

    QLIKE = mean(y_true / y_pred - log(y_true / y_pred) - 1)
    Lower is better. Clips both y_true and y_pred to eps to handle exact zeros
    (e.g. squared VIX log-returns on unchanged days).
    """
    eps = 1e-10
    yt = np.maximum(np.asarray(y_true, dtype=float), eps)
    yp = np.maximum(np.asarray(y_pred, dtype=float), eps)
    ratio = yt / yp
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def compute_all(y_true: np.ndarray, y_pred: np.ndarray, include_qlike: bool = True) -> dict:
    out = {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}
    if include_qlike:
        out["qlike"] = qlike(y_true, y_pred)
    return out


def regime_accuracy(true_regimes: np.ndarray, pred_rv: np.ndarray, thresholds: tuple) -> float:
    """Fraction of test days where the model's RV forecast lands in the correct regime.

    true_regimes : HMM-predicted regime labels for the true RV series (0/1/2)
    pred_rv      : model's RV forecast for the same period
    thresholds   : (low_mid, mid_high) RV decision boundaries from HMM
    """
    from models.hmm_regimes import classify_by_threshold
    pred_regimes = classify_by_threshold(pred_rv, thresholds)
    n = min(len(true_regimes), len(pred_regimes))
    return float(np.mean(true_regimes[:n] == pred_regimes[:n]))


def volatility_timing_sharpe(
    pred_rv: np.ndarray,
    actual_returns: np.ndarray,
    thresholds: tuple,
    trading_days: int = 252,
) -> float:
    """Annualised Sharpe ratio of a simple vol-timing strategy.

    Position sizing based on predicted volatility regime:
      Low  → full position (1.0)
      Med  → half position (0.5)
      High → cash          (0.0)

    Timing: pred_rv[t] predicts RV for day t+1, so the position set at end of
    day t earns actual_returns[t+1].  This avoids look-ahead bias.

    actual_returns : log-returns of S&P 500 for the same aligned test period
    """
    from models.hmm_regimes import classify_by_threshold
    # Use n-1 pairs: position from pred[t] earns return at t+1
    n = min(len(pred_rv), len(actual_returns)) - 1
    if n <= 0:
        return 0.0
    pred_regimes = classify_by_threshold(pred_rv[:n], thresholds)
    positions = np.where(pred_regimes == 0, 1.0, np.where(pred_regimes == 1, 0.5, 0.0))
    strat_returns = positions * actual_returns[1 : n + 1]
    std = strat_returns.std()
    if std == 0:
        return 0.0
    return float(strat_returns.mean() / std * np.sqrt(trading_days))
