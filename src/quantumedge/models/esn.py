"""Echo State Network using reservoirpy."""

import warnings

import numpy as np


def run_esn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    units: int = 500,
    sr: float = 0.9,
    lr: float = 0.3,
    input_scaling: float = 0.5,
    ridge: float = 1e-6,
    warmup: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Train an ESN and return predictions on X_test.

    reservoirpy expects shapes (timesteps, features).
    """
    warnings.filterwarnings("ignore")

    import reservoirpy as rpy
    from reservoirpy.nodes import Reservoir, Ridge as RidgeReadout

    rpy.set_seed(seed)

    reservoir = Reservoir(
        units=units,
        sr=sr,
        lr=lr,
        input_scaling=input_scaling,
        seed=seed,
    )
    readout = RidgeReadout(output_dim=1, ridge=ridge)
    esn = reservoir >> readout

    # reservoirpy needs 2-D arrays: (timesteps, n_features)
    X_tr = X_train if X_train.ndim == 2 else X_train.reshape(-1, 1)
    y_tr = y_train.reshape(-1, 1)
    X_te = X_test if X_test.ndim == 2 else X_test.reshape(-1, 1)

    esn.fit(X_tr, y_tr, warmup=warmup)
    y_pred = esn.run(X_te).flatten()
    return y_pred


def run_esn_financial(financial_split: dict, **kwargs) -> np.ndarray:
    """ESN for financial data: input features are [rv_d, rv_w, rv_m, log_ret, vix, gk, vix_rv_spread]."""
    feature_cols = ["rv_d", "rv_w", "rv_m", "log_ret", "vix", "gk", "vix_rv_spread"]
    train = financial_split["train"]
    test = financial_split["test"]
    X_train = train[feature_cols].values
    y_train = train["target_rv"].values
    X_test = test[feature_cols].values
    return run_esn(X_train, y_train, X_test, **kwargs)


def run_esn_vix(vix_split: dict, **kwargs) -> np.ndarray:
    """ESN predicting next-day VIX level."""
    feature_cols = ["vix", "vix_w", "vix_m", "vix_log_ret"]
    train = vix_split["train"]
    test = vix_split["test"]
    return run_esn(
        train[feature_cols].values,
        train["target_vix"].values,
        test[feature_cols].values,
        **kwargs,
    )


def run_esn_chaotic(chaotic_split: dict, **kwargs) -> np.ndarray:
    """ESN for chaotic system: input is lagged embedding."""
    return run_esn(
        chaotic_split["X_train"],
        chaotic_split["y_train"],
        chaotic_split["X_test"],
        **kwargs,
    )
