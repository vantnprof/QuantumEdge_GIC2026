"""
run_classical_benchmark.py
==========================
Runs all classical baselines on exactly the same data as the QRC notebook.
Loads df via the identical build_financial_features() pipeline, appends results
to fair_comparison.csv, then regenerates quantumedge_vs_classical.png.

Run from proposed-architecture/:
    .venv/bin/python notebook/run_classical_benchmark.py
"""

import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "qrc_financial_results"
RESULTS_DIR.mkdir(exist_ok=True)

START, END   = "2010-01-01", "2025-12-31"
TRAIN_FRAC   = 0.80
RIDGE_ALPHAS = np.logspace(-6, 6, 25)

# ── 1. Load data (identical to notebook) ─────────────────────────────────────
def _gk_rv(frame):
    h = np.log(frame["High"].astype(np.float64).values / frame["Open"].astype(np.float64).values)
    l = np.log(frame["Low"].astype(np.float64).values  / frame["Open"].astype(np.float64).values)
    c = np.log(frame["Close"].astype(np.float64).values / frame["Open"].astype(np.float64).values)
    return pd.Series(np.maximum(0.5*(h-l)**2 - (2*np.log(2)-1)*c**2, 0), index=frame.index)

print("Downloading data …")
sp500 = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)
vix   = yf.download("^VIX",  start=START, end=END, auto_adjust=True, progress=False)
if isinstance(sp500.columns, pd.MultiIndex): sp500.columns = sp500.columns.get_level_values(0)
if isinstance(vix.columns,   pd.MultiIndex): vix.columns   = vix.columns.get_level_values(0)

# Try Oxford-Man, fall back to GK
oxford_man = pd.DataFrame(columns=["rv5"])
try:
    import io, zipfile, urllib.request
    url = "https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip"
    with urllib.request.urlopen(url, timeout=20) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            raw = pd.read_csv(f, index_col=0, parse_dates=True, low_memory=False)
    spx = raw[raw.get("Symbol", raw.index) == ".SPX"] if "Symbol" in raw.columns else raw
    spx.index = pd.to_datetime(spx.index)
    spx = spx[(spx.index >= START) & (spx.index <= END)]
    oxford_man = spx[["rv5"]].rename(columns={"rv5": "rv5"})
    print("  Oxford-Man: downloaded")
except Exception:
    print("  Oxford-Man: unavailable, using Garman-Klass fallback")

def build_financial_features(sp500, vix, oxford_man):
    # cast all OHLC to float64 immediately — yfinance can return object dtype
    df = sp500[["Open", "High", "Low", "Close"]].astype(np.float64).copy()
    def _ln(s): return np.log(np.asarray(s, dtype=np.float64).clip(1e-10))
    df["log_ret"] = _ln(df["Close"] / df["Close"].shift(1))
    rv = (oxford_man["rv5"].reindex(df.index).ffill().bfill()
          if len(oxford_man) > 0 and "rv5" in oxford_man.columns
          else _gk_rv(df).reindex(df.index))
    df["rv"]  = np.asarray(rv, dtype=np.float64)
    h = _ln(df["High"] / df["Open"])
    l = _ln(df["Low"]  / df["Open"])
    c = _ln(df["Close"] / df["Open"])
    df["gk"]  = np.maximum(0.5 * (h - l)**2 - (2 * np.log(2) - 1) * c**2, 0)
    df["rv_d"] = df["rv"]
    df["rv_w"] = df["rv"].rolling(5,  min_periods=5).mean()
    df["rv_m"] = df["rv"].rolling(22, min_periods=22).mean()
    df["log_rv_d"] = _ln(df["rv_d"])
    df["log_rv_w"] = _ln(df["rv_w"])
    df["log_rv_m"] = _ln(df["rv_m"])
    df["log_gk"]   = _ln(df["gk"])
    df["vix"] = np.asarray(vix["Close"].reindex(df.index).ffill(), dtype=np.float64)
    df["vix_rv_spread"] = (df["vix"] / 100)**2 / 252 - df["rv"]
    df["target_rv"]     = df["rv"].shift(-1)
    df["target_log_rv"] = _ln(df["rv"].shift(-1))
    return df.dropna()

df = build_financial_features(sp500, vix, oxford_man)
n_total = len(df)
n_tr = int(n_total * TRAIN_FRAC)
n_te = n_total - n_tr
print(f"Data: {n_total} rows  train={n_tr}  test={n_te}")
print(f"Train: {df.index[0].date()} → {df.index[n_tr-1].date()}")
print(f"Test:  {df.index[n_tr].date()} → {df.index[-1].date()}")

y_true_rv    = df["target_rv"].values[n_tr:n_tr + n_te]
y_rv_train   = df["target_rv"].values[:n_tr]
y_logrv_train = np.log(y_rv_train.clip(1e-12))
test_returns  = df["log_ret"].values[n_tr:n_tr + n_te]

# ── HMM for regime accuracy / Sharpe (same as notebook) ─────────────────────
LOG_RV_TRAIN = np.log(df["rv"].values[:n_tr].clip(1e-10))
hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=SEED)
hmm.fit(LOG_RV_TRAIN.reshape(-1, 1))
means_order = np.argsort(hmm.means_.ravel())
state_map = {raw: sorted_idx for sorted_idx, raw in enumerate(means_order)}
sorted_means = hmm.means_.ravel()[means_order]
thresholds = ((sorted_means[0] + sorted_means[1]) / 2,
              (sorted_means[1] + sorted_means[2]) / 2)

def classify_by_threshold(rv_arr, thr):
    log_rv = np.log(np.clip(rv_arr, 1e-12, None))
    return np.where(log_rv < thr[0], 0, np.where(log_rv < thr[1], 1, 2))

def qlike(yt, yp, eps=1e-10):
    yt = np.asarray(yt, float); yp = np.maximum(np.asarray(yp, float), eps)
    r = yt / yp
    return float(np.mean(r - np.log(r) - 1))

def vol_timing_sharpe(pred_rv, log_rets, thr, ann=252):
    regimes = classify_by_threshold(pred_rv, thr)
    w = np.where(regimes == 0, 1.0, np.where(regimes == 1, 0.5, 0.0))
    ret = w[:-1] * log_rets[1:]
    return float(ret.mean() / (ret.std() + 1e-10) * np.sqrt(ann))

def eval_metrics(y_true, y_pred, name, thr):
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], np.maximum(y_pred[:n], 1e-10)
    rmse_v  = float(np.sqrt(np.mean((yt - yp)**2)))
    qlike_v = qlike(yt, yp)
    reg_acc = float(np.mean(classify_by_threshold(yt, thr) == classify_by_threshold(yp, thr)))
    sharpe_v = vol_timing_sharpe(yp, test_returns[:n], thr)
    print(f"  {name:<16}  RMSE={rmse_v:.4e}  QLIKE={qlike_v:.4f}  RegAcc={reg_acc:.3f}")
    return {"model": name, "rmse": rmse_v, "qlike": qlike_v,
            "regime_acc": reg_acc, "sharpe": sharpe_v}

# ── Feature matrices ─────────────────────────────────────────────────────────
LOG_COLS = ["log_rv_d", "log_rv_w", "log_rv_m", "log_ret", "vix", "log_gk", "vix_rv_spread"]
_Xtr = df[LOG_COLS].values[:n_tr]
_Xte = df[LOG_COLS].values[n_tr:n_tr + n_te]

results = []

# ── GARCH(1,1) — fit once on train, roll filter forward ─────────────────────
print("\n[1/4] GARCH(1,1) …")
from arch import arch_model as _arch_model
_lr_full = df["log_ret"].values * 100   # percent returns, full series
_garch_m = _arch_model(_lr_full[:n_tr], vol="Garch", p=1, q=1, dist="normal", rescale=False)
_garch_r = _garch_m.fit(disp="off", show_warning=False)
# Forecast test period: use the full series with parameters fixed at training estimates
_garch_full = _arch_model(_lr_full, vol="Garch", p=1, q=1, dist="normal", rescale=False)
_garch_full_r = _garch_full.fix(_garch_r.params)
_fc = _garch_full_r.forecast(start=n_tr, reindex=False)
garch_pred = np.maximum(_fc.variance.values[-n_te:, 0] / 1e4, 1e-10)
results.append(eval_metrics(y_true_rv, garch_pred, "GARCH(1,1)", thresholds))

# ── ARIMA(5,0,0)-log — fit once, roll state forward ─────────────────────────
print("\n[2/4] ARIMA(5,0,0)-log …")
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
_log_rv_train = np.log(df["rv"].values[:n_tr].clip(1e-12))
_log_rv_test  = np.log(df["rv"].values[n_tr:n_tr + n_te].clip(1e-12))
_ar = _ARIMA(_log_rv_train, order=(5, 0, 0)).fit()
_arima_preds = []
for _obs in _log_rv_test:
    _ar = _ar.append([float(_obs)], refit=False)
    fc = _ar.forecast(1)
    _arima_preds.append(float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0]))
arima_pred = np.exp(np.array(_arima_preds))
results.append(eval_metrics(y_true_rv, arima_pred, "ARIMA-log", thresholds))

# ── XGBoost-log ──────────────────────────────────────────────────────────────
print("\n[3/4] XGBoost-log …")
import xgboost as _xgb
_n_val_x = max(1, int(n_tr * 0.1))
_xgb_m = _xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=20, random_state=SEED, n_jobs=-1, verbosity=0,
)
_xgb_m.fit(
    _Xtr[:-_n_val_x], y_logrv_train[:-_n_val_x],
    eval_set=[(_Xtr[-_n_val_x:], y_logrv_train[-_n_val_x:])], verbose=False,
)
xgb_pred = np.exp(_xgb_m.predict(_Xte))
results.append(eval_metrics(y_true_rv, xgb_pred, "XGBoost-log", thresholds))

# ── LSTM-log ─────────────────────────────────────────────────────────────────
print("\n[4/4] LSTM-log …")
import torch as _torch, torch.nn as _nn
from torch.utils.data import DataLoader as _DL, TensorDataset as _TD
_torch.manual_seed(SEED)

class _LSTMNet(_nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.lstm = _nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = _nn.Dropout(0.2)
        self.fc   = _nn.Linear(hidden, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(self.drop(o[:, -1, :])).squeeze(-1)

_SEQ = 22
_sc  = StandardScaler().fit(_Xtr)
_Xtr_s = _sc.transform(_Xtr); _Xte_s = _sc.transform(_Xte)

def _make_seqs(X, y, seq):
    Xs, ys = [], []
    for t in range(seq - 1, len(X)):
        Xs.append(X[t - seq + 1:t + 1]); ys.append(y[t])
    return np.array(Xs, np.float32), np.array(ys, np.float32)

_Xtr_seq, _ytr_seq = _make_seqs(_Xtr_s, y_logrv_train, _SEQ)
_Xte_seq, _         = _make_seqs(_Xte_s, np.zeros(n_te), _SEQ)
_n_val_l = max(1, int(len(_Xtr_seq) * 0.1))
_loader  = _DL(_TD(_torch.tensor(_Xtr_seq[:-_n_val_l]),
                    _torch.tensor(_ytr_seq[:-_n_val_l])), batch_size=32, shuffle=False)
_Xv = _torch.tensor(_Xtr_seq[-_n_val_l:]); _yv = _torch.tensor(_ytr_seq[-_n_val_l:])
_net = _LSTMNet(len(LOG_COLS))
_opt = _torch.optim.Adam(_net.parameters(), lr=1e-3); _crit = _nn.MSELoss()
_best_val, _best_state, _no_imp = float("inf"), None, 0
for _ep in range(100):
    _net.train()
    for _xb, _yb in _loader:
        _opt.zero_grad(); _crit(_net(_xb), _yb).backward(); _opt.step()
    _net.eval()
    with _torch.no_grad():
        _vl = _crit(_net(_Xv), _yv).item()
    if _vl < _best_val:
        _best_val = _vl; _best_state = {k: v.clone() for k, v in _net.state_dict().items()}; _no_imp = 0
    else:
        _no_imp += 1
        if _no_imp >= 10: break
_net.load_state_dict(_best_state); _net.eval()
with _torch.no_grad():
    lstm_pred = np.exp(_net(_torch.tensor(_Xte_seq)).numpy())
_skip = _SEQ - 1
results.append(eval_metrics(y_true_rv[_skip:], lstm_pred, "LSTM-log", thresholds))

# ── Load existing QRC results and merge ──────────────────────────────────────
print("\nLoading existing QRC fair_comparison.csv …")
qrc_df = pd.read_csv(RESULTS_DIR / "fair_comparison.csv")

# Drop any stale classical rows that may be leftover
classical_names = {"GARCH(1,1)", "ARIMA-log", "XGBoost-log", "LSTM-log"}
qrc_df = qrc_df[~qrc_df["model"].isin(classical_names)].copy()

har_rmse = qrc_df.loc[qrc_df["model"] == "HAR-RV", "rmse"].values[0]

new_rows = []
for r in results:
    new_rows.append({
        "model":      r["model"],
        "rmse":       r["rmse"],
        "qlike":      r["qlike"],
        "regime_acc": r["regime_acc"],
        "sharpe":     r["sharpe"],
        "vs_HAR_pct": (har_rmse - r["rmse"]) / har_rmse * 100,
    })

combined = pd.concat([qrc_df, pd.DataFrame(new_rows)], ignore_index=True)
combined = combined.sort_values("rmse").reset_index(drop=True)
combined.to_csv(RESULTS_DIR / "fair_comparison.csv", index=False)
print(f"\nSaved {len(combined)} rows → {RESULTS_DIR}/fair_comparison.csv")
print(combined[["model", "rmse", "qlike", "vs_HAR_pct"]].to_string(index=False))
