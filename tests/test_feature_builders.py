import numpy as np
import pandas as pd

from quantumedge.features.builders import (
    build_chaotic_features,
    build_financial_features,
    split_financial,
)


def test_financial_features_and_split_are_temporal():
    dates = pd.bdate_range("2024-01-01", periods=50)
    base = np.linspace(100.0, 120.0, len(dates))
    sp500 = pd.DataFrame(
        {
            "Open": base,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base * 1.002,
            "Volume": np.arange(len(dates)) + 1,
        },
        index=dates,
    )
    vix = pd.Series(np.linspace(15.0, 20.0, len(dates)), index=dates, name="VIX")
    oxford_man = pd.DataFrame({"rv5": np.full(len(dates), 0.0001)}, index=dates)

    features = build_financial_features(sp500, vix, oxford_man)
    split = split_financial(features)

    assert {"rv", "rv_w", "rv_m", "vix_rv_spread", "target_rv"} <= set(features.columns)
    assert not features.isna().any().any()
    assert len(split["train"]) + len(split["test"]) == len(features)
    assert split["train"].index.max() < split["test"].index.min()


def test_chaotic_features_fit_scaler_on_training_window():
    series = np.linspace(-2.0, 2.0, 100)
    split = build_chaotic_features(series, lag=10)

    assert split["X_train"].shape[1] == 10
    assert len(split["X_train"]) + len(split["X_test"]) == 89
    assert split["X_train"].min() >= 0.0
    assert split["X_train"].max() <= 1.0
