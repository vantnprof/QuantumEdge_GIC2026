"""Data acquisition: downloads/generates all 5 datasets."""

import io
import zipfile
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.integrate import solve_ivp

from quantumedge.config import ensure_runtime_dirs

RAW = ensure_runtime_dirs().raw_data_dir

OXFORD_MAN_URL = (
    "https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip"
)
START = "2010-01-01"
END = "2025-12-31"


# ---------------------------------------------------------------------------
# S&P 500 OHLCV
# ---------------------------------------------------------------------------

def load_sp500() -> pd.DataFrame:
    path = RAW / "sp500_ohlcv.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print("[data] SP500: loaded from cache")
        return df

    print("[data] SP500: downloading via yfinance …")
    df = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False,
                     multi_level_index=False)
    # yfinance 1.x may return MultiIndex columns — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df = df.ffill(limit=2)
    df = df.dropna()
    df.to_csv(path)
    print(f"[data] SP500: {len(df)} rows → {path.name}")
    return df


# ---------------------------------------------------------------------------
# CBOE VIX
# ---------------------------------------------------------------------------

def load_vix(sp500_index: pd.DatetimeIndex) -> pd.Series:
    path = RAW / "vix.csv"
    if path.exists():
        s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        print("[data] VIX: loaded from cache")
        return s.reindex(sp500_index).ffill()

    print("[data] VIX: downloading via yfinance …")
    df = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False,
                     multi_level_index=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].rename("VIX")
    s.index.name = "Date"
    s.to_csv(path)
    print(f"[data] VIX: {len(s)} rows → {path.name}")
    return s.reindex(sp500_index).ffill()


# ---------------------------------------------------------------------------
# Oxford-Man Realized Volatility (with GK fallback)
# ---------------------------------------------------------------------------

def _try_oxford_man_download() -> pd.DataFrame | None:
    """Try primary Oxford-Man URL; returns DataFrame or None."""
    try:
        import urllib.request

        print("[data] Oxford-Man: trying primary URL …")
        with urllib.request.urlopen(OXFORD_MAN_URL, timeout=30) as r:
            raw_bytes = r.read()

        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, index_col=0, parse_dates=True, low_memory=False)

        # Oxford-Man CSV: rows are (date, symbol) pairs; find Symbol column
        sym_col = next((c for c in df.columns if c.lower() in ("symbol", "sym")), None)
        if sym_col:
            spx = df[df[sym_col] == ".SPX"].copy()
        elif df.index.nlevels > 1:
            spx = df[df.index.get_level_values(-1) == ".SPX"].copy()
        else:
            spx = df  # assume already filtered to SPX

        spx.index = pd.to_datetime(spx.index)
        spx = spx.sort_index()
        spx = spx[(spx.index >= START) & (spx.index <= END)]
        rv_col = next((c for c in spx.columns if "rv5" in c.lower()), None)
        if rv_col is None:
            return None
        spx = spx[[rv_col]].rename(columns={rv_col: "rv5"})
        print(f"[data] Oxford-Man: {len(spx)} rows (primary URL)")
        return spx

    except Exception as e:
        print(f"[data] Oxford-Man URL failed: {e}")
        return None


def _garman_klass_rv(sp500: pd.DataFrame) -> pd.Series:
    """Garman-Klass realized volatility estimator from OHLCV."""
    h = np.log(sp500["High"] / sp500["Open"])
    l = np.log(sp500["Low"] / sp500["Open"])
    c = np.log(sp500["Close"] / sp500["Open"])
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * c ** 2
    return gk.rename("rv5").clip(lower=0)


def load_oxford_man(sp500: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe with 'rv5' column, source_label)."""
    path = RAW / "oxford_man_rv.csv"
    src_path = RAW / "oxford_man_source.txt"

    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        source = src_path.read_text().strip() if src_path.exists() else "cache"
        print(f"[data] Oxford-Man RV: loaded from cache (source={source})")
        return df, source

    om = _try_oxford_man_download()
    if om is not None:
        om.to_csv(path)
        src_path.write_text("oxford-man-primary")
        return om, "oxford-man-primary"

    print("[data] Oxford-Man: URL failed → using Garman-Klass from OHLCV")
    gk = _garman_klass_rv(sp500).to_frame()
    gk.to_csv(path)
    src_path.write_text("garman-klass-fallback")
    print(f"[data] Oxford-Man (GK fallback): {len(gk)} rows → {path.name}")
    return gk, "garman-klass-fallback"


# ---------------------------------------------------------------------------
# Mackey-Glass
# ---------------------------------------------------------------------------

def _mackey_glass_euler(
    beta: float = 0.2,
    gamma: float = 0.1,
    tau: int = 17,
    n: int = 10,
    n_steps: int = 10_000,
    dt: float = 1.0,
    discard: int = 1_000,
) -> np.ndarray:
    x = np.zeros(n_steps + discard)
    x[: tau + 1] = 0.9 + 0.1 * np.random.default_rng(42).random(tau + 1)
    for t in range(tau, n_steps + discard - 1):
        x_tau = x[t - tau]
        dxdt = beta * x_tau / (1.0 + x_tau ** n) - gamma * x[t]
        x[t + 1] = x[t] + dt * dxdt
    return x[discard:]


def load_mackey_glass() -> pd.DataFrame:
    path = RAW / "mackey_glass.csv"
    if path.exists():
        df = pd.read_csv(path)
        print("[data] Mackey-Glass: loaded from cache")
        return df

    print("[data] Mackey-Glass: generating synthetically …")
    vals = _mackey_glass_euler()
    df = pd.DataFrame({"value": vals})
    df.to_csv(path, index=False)
    print(f"[data] Mackey-Glass: {len(df)} rows → {path.name}")
    return df


# ---------------------------------------------------------------------------
# Lorenz System
# ---------------------------------------------------------------------------

def _lorenz_ode(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def load_lorenz() -> pd.DataFrame:
    path = RAW / "lorenz.csv"
    if path.exists():
        df = pd.read_csv(path)
        print("[data] Lorenz: loaded from cache")
        return df

    print("[data] Lorenz: generating synthetically …")
    sol = solve_ivp(
        _lorenz_ode,
        t_span=(0, 50),
        y0=[1.0, 1.0, 1.0],
        method="RK45",
        t_eval=np.arange(0, 50, 0.01),
        rtol=1e-9,
        atol=1e-9,
    )
    df = pd.DataFrame({"x": sol.y[0], "y": sol.y[1], "z": sol.y[2]})
    df.to_csv(path, index=False)
    print(f"[data] Lorenz: {len(df)} rows → {path.name}")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all() -> dict:
    """Load all five datasets and return as a dict."""
    warnings.filterwarnings("ignore")
    sp500 = load_sp500()
    vix = load_vix(sp500.index)
    oxford_man, oxford_source = load_oxford_man(sp500)
    mg = load_mackey_glass()
    lorenz = load_lorenz()
    return {
        "sp500": sp500,
        "vix": vix,
        "oxford_man": oxford_man,
        "oxford_source": oxford_source,
        "mackey_glass": mg,
        "lorenz": lorenz,
    }
