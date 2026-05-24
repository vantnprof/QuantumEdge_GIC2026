"""HAR-RV model (Corsi, 2009) — OLS regression on daily/weekly/monthly RV components."""

import numpy as np
from sklearn.linear_model import LinearRegression


def run_har_rv(financial_split: dict) -> tuple[np.ndarray, dict]:
    """Fit HAR-RV on training split, predict on test split.

    Returns (predictions, {'coefs': ..., 'intercept': ...}).
    """
    train = financial_split["train"]
    test = financial_split["test"]

    feature_cols = ["rv_d", "rv_w", "rv_m"]
    target_col = "target_rv"

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    coefs = dict(zip(feature_cols, model.coef_))
    coefs["intercept"] = model.intercept_

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 1e-10)  # RV must be positive

    print(f"  [HAR-RV] coefs: {coefs}")
    print(f"  [HAR-RV] coef sum (excl. intercept): {sum(model.coef_):.4f}")

    return y_pred, coefs
