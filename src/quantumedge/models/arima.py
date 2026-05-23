"""ARIMA baseline for financial time series (finance-only).

SP500 RV  : ARIMA(5,0,0) — AR(5) on realized variance
VIX level : ARIMA(1,1,0) — first-differenced AR(1)

Walk-forward one-step-ahead: at each test step i, the model has seen all
observations up to and including test_rv[i] / test_vix[i], then forecasts
the next value (= target_rv[i] / target_vix[i]).  Parameters are estimated
once on the training window; subsequent steps extend the state without
re-estimating (refit=False).
"""

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def _rolling_forecast(train_values: np.ndarray, test_current: np.ndarray, order: tuple) -> np.ndarray:
    """Walk-forward one-step-ahead ARIMA without parameter refitting.

    At step i:
      1. Extend model state with test_current[i] (current observation).
      2. Forecast one step ahead → prediction for test_current[i]+1 = target[i].
    No future data is used; parameters are frozen from the training fit.
    """
    train_arr = np.asarray(train_values, dtype=float)
    result = ARIMA(train_arr, order=order).fit()

    preds = []
    for obs in test_current:
        result = result.append([float(obs)], refit=False)
        fc = result.forecast(1)
        yhat = float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0])
        preds.append(yhat)

    return np.array(preds)


def run_arima_financial(financial_split: dict) -> np.ndarray:
    """ARIMA(5,0,0) on realized variance — finance only."""
    train = financial_split["train"]
    test = financial_split["test"]
    preds = _rolling_forecast(
        train["rv"].values,
        test["rv"].values,
        order=(5, 0, 0),
    )
    return np.clip(preds, 1e-10, None)


def run_arima_financial_log(financial_split: dict) -> np.ndarray:
    """ARIMA(5,0,0) on log(RV) — log-space for fair QLIKE comparison. Outputs exp(pred)."""
    train = financial_split["train"]
    test  = financial_split["test"]
    log_train_rv = np.log(np.clip(train["rv"].values, 1e-12, None))
    log_test_rv  = np.log(np.clip(test["rv"].values,  1e-12, None))
    log_preds = _rolling_forecast(log_train_rv, log_test_rv, order=(5, 0, 0))
    return np.exp(log_preds)


def run_arima_vix(vix_split: dict) -> np.ndarray:
    """ARIMA(1,1,0) on VIX level — predicts next-day VIX."""
    train = vix_split["train"]
    test = vix_split["test"]
    preds = _rolling_forecast(
        train["vix"].values,
        test["vix"].values,
        order=(1, 1, 0),
    )
    return np.clip(preds, 1e-10, None)
