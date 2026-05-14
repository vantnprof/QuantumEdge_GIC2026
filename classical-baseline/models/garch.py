"""GARCH(1,1) one-step-ahead volatility forecast."""

import warnings

import numpy as np
import pandas as pd


def run_garch(returns_all: np.ndarray, n_train: int) -> np.ndarray:
    """Fit GARCH(1,1) on first n_train observations; return test-period variance forecasts.

    Uses arch's last_obs + forecast(start=n_train) for efficient out-of-sample rolling
    predictions without refitting at every step.
    """
    from arch import arch_model

    warnings.filterwarnings("ignore")

    returns_scaled = returns_all * 100  # arch prefers percentage returns

    am = arch_model(
        returns_scaled,
        vol="Garch",
        p=1,
        q=1,
        mean="Constant",
        dist="Normal",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(last_obs=n_train, disp="off", show_warning=False)

    forecasts = res.forecast(start=n_train, horizon=1, reindex=False)
    # Variance is in (% return)^2 — convert back to return^2
    h_pred = forecasts.variance.values.squeeze() / 10_000
    return np.array(h_pred)


def run_garch_on_vix(vix_split: dict) -> np.ndarray:
    """GARCH(1,1) on VIX log-returns; returns predicted conditional variance for test period.

    Target for QLIKE/RMSE: squared VIX log-returns (actual realized variance of VIX changes).
    """
    import pandas as pd

    train = vix_split["train"]
    test = vix_split["test"]
    all_returns = pd.concat([train["vix_log_ret"], test["vix_log_ret"]]).values
    n_train = len(train)
    h_pred = run_garch(all_returns, n_train)
    n_test = len(test)
    if len(h_pred) > n_test:
        h_pred = h_pred[:n_test]
    elif len(h_pred) < n_test:
        h_pred = np.pad(h_pred, (0, n_test - len(h_pred)), mode="edge")
    return h_pred


def run_garch_on_financial(financial_split: dict) -> np.ndarray:
    train = financial_split["train"]
    test = financial_split["test"]
    all_returns = pd.concat([train["log_ret"], test["log_ret"]]).values
    n_train = len(train)
    h_pred = run_garch(all_returns, n_train)

    # Align length with test set (arch may return n_test or n_test+1 values)
    n_test = len(test)
    if len(h_pred) > n_test:
        h_pred = h_pred[:n_test]
    elif len(h_pred) < n_test:
        h_pred = np.pad(h_pred, (0, n_test - len(h_pred)), mode="edge")

    return h_pred
