"""XGBoost regressor for volatility and chaotic-system forecasting."""

import numpy as np
import xgboost as xgb


def run_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 20,
    val_frac: float = 0.1,
) -> tuple[np.ndarray, xgb.XGBRegressor]:
    """Train XGBoost; return (predictions, fitted_model)."""
    n_val = max(1, int(len(X_train) * val_frac))
    X_val = X_train[-n_val:]
    y_val = y_train[-n_val:]
    X_tr = X_train[:-n_val]
    y_tr = y_train[:-n_val]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  [XGBoost] best_iteration={model.best_iteration}")

    y_pred = model.predict(X_test)
    return y_pred, model


def run_xgboost_financial(financial_split: dict) -> tuple[np.ndarray, xgb.XGBRegressor]:
    feature_cols = ["rv_d", "rv_w", "rv_m", "log_ret", "vix", "gk", "vix_rv_spread"]
    train = financial_split["train"]
    test = financial_split["test"]
    return run_xgboost(
        train[feature_cols].values,
        train["target_rv"].values,
        test[feature_cols].values,
    )


def run_xgboost_financial_log(financial_split: dict) -> tuple[np.ndarray, xgb.XGBRegressor]:
    """Log-space XGBoost: log-transforms RV features and target, outputs exp(pred)."""
    raw_cols = ["rv_d", "rv_w", "rv_m", "log_ret", "vix", "gk", "vix_rv_spread"]
    train = financial_split["train"]
    test  = financial_split["test"]

    def _log_feats(df):
        X = df[raw_cols].values.copy()
        X[:, :3] = np.log(np.clip(X[:, :3], 1e-12, None))
        return X

    y_tr_log = np.log(np.clip(train["target_rv"].values, 1e-12, None))
    y_pred_log, model = run_xgboost(_log_feats(train), y_tr_log, _log_feats(test))
    return np.exp(y_pred_log), model


def run_xgboost_vix(vix_split: dict) -> tuple[np.ndarray, xgb.XGBRegressor]:
    feature_cols = ["vix", "vix_w", "vix_m", "vix_log_ret"]
    train = vix_split["train"]
    test = vix_split["test"]
    return run_xgboost(
        train[feature_cols].values,
        train["target_vix"].values,
        test[feature_cols].values,
    )


def run_xgboost_chaotic(chaotic_split: dict) -> tuple[np.ndarray, xgb.XGBRegressor]:
    return run_xgboost(
        chaotic_split["X_train"],
        chaotic_split["y_train"],
        chaotic_split["X_test"],
    )
