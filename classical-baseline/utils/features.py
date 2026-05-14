"""Feature engineering and train/test splitting."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.80
LAG_CHAOTIC = 10
SEQ_LEN_FINANCIAL = 22
SEQ_LEN_CHAOTIC = 10


# ---------------------------------------------------------------------------
# Financial features
# ---------------------------------------------------------------------------

def build_financial_features(
    sp500: pd.DataFrame,
    vix: pd.Series,
    oxford_man: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature matrix for financial datasets."""
    df = sp500[["Open", "High", "Low", "Close"]].copy()

    # Log returns
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))

    # Realized variance — prefer oxford_man rv5, fall back to GK
    if "rv5" in oxford_man.columns:
        rv = oxford_man["rv5"].reindex(df.index).ffill().bfill()
    else:
        h = np.log(df["High"] / df["Open"])
        l = np.log(df["Low"] / df["Open"])
        c = np.log(df["Close"] / df["Open"])
        rv = (0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * c ** 2).clip(lower=0)
    df["rv"] = rv

    # Garman-Klass (always computed as a feature)
    h = np.log(df["High"] / df["Open"])
    l = np.log(df["Low"] / df["Open"])
    c = np.log(df["Close"] / df["Open"])
    df["gk"] = (0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * c ** 2).clip(lower=0)

    # HAR components
    df["rv_d"] = df["rv"]
    df["rv_w"] = df["rv"].rolling(5, min_periods=5).mean()
    df["rv_m"] = df["rv"].rolling(22, min_periods=22).mean()

    # VIX
    df["vix"] = vix.reindex(df.index).ffill()

    # VIX-RV spread: annualised VIX → daily variance, then diff with rv
    df["vix_rv_spread"] = (df["vix"] / 100) ** 2 / 252 - df["rv"]

    # Target: next-day RV
    df["target_rv"] = df["rv"].shift(-1)

    df = df.dropna()
    return df


def split_financial(df: pd.DataFrame) -> dict:
    n = len(df)
    cut = int(n * TRAIN_FRAC)
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return {"train": train, "test": test, "cut": cut}


# ---------------------------------------------------------------------------
# Chaotic system features
# ---------------------------------------------------------------------------

def build_chaotic_features(series: np.ndarray, lag: int = LAG_CHAOTIC) -> dict:
    """Create lagged-window feature matrix and normalised train/test splits.

    Scaler is fit on the training portion of the raw series only to prevent leakage.
    """
    n_raw = len(series)
    raw_cut = int(n_raw * TRAIN_FRAC)

    scaler = MinMaxScaler()
    scaler.fit(series[:raw_cut].reshape(-1, 1))           # fit on train only
    series_scaled = scaler.transform(series.reshape(-1, 1)).flatten()  # transform all

    X, y = [], []
    for i in range(lag, len(series_scaled) - 1):
        X.append(series_scaled[i - lag : i])
        y.append(series_scaled[i])

    X = np.array(X)
    y = np.array(y)

    n = len(X)
    cut = int(n * TRAIN_FRAC)
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "cut": cut,
    }


# ---------------------------------------------------------------------------
# VIX dataset features
# ---------------------------------------------------------------------------

def build_vix_features(vix: pd.Series) -> pd.DataFrame:
    """Feature matrix for next-day VIX level prediction."""
    df = pd.DataFrame({"vix": vix})
    df["vix_log_ret"] = np.log(df["vix"] / df["vix"].shift(1))
    df["vix_w"] = df["vix"].rolling(5, min_periods=5).mean()
    df["vix_m"] = df["vix"].rolling(22, min_periods=22).mean()
    df["target_vix"] = df["vix"].shift(-1)
    df = df.dropna()
    return df


def split_vix(df: pd.DataFrame) -> dict:
    n = len(df)
    cut = int(n * TRAIN_FRAC)
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return {"train": train, "test": test, "cut": cut}


# ---------------------------------------------------------------------------
# Convenience: prepare all splits
# ---------------------------------------------------------------------------

def prepare_all(data: dict) -> dict:
    sp500 = data["sp500"]
    vix = data["vix"]
    oxford_man = data["oxford_man"]
    mg = data["mackey_glass"]["value"].values
    lorenz_x = data["lorenz"]["x"].values

    fin = build_financial_features(sp500, vix, oxford_man)
    fin_split = split_financial(fin)

    vix_df = build_vix_features(vix)
    vix_split = split_vix(vix_df)

    mg_split = build_chaotic_features(mg)
    lorenz_split = build_chaotic_features(lorenz_x)

    return {
        "financial": fin,
        "financial_split": fin_split,
        "vix_df": vix_df,
        "vix_split": vix_split,
        "mackey_glass_split": mg_split,
        "lorenz_split": lorenz_split,
    }
