"""
run_aer_noise.py
================
Noise-level hardware simulation using PennyLane default.mixed.

Depolarizing (p=0.003/gate) + amplitude damping (gamma=0.0002) matching
IBM Heron r2 calibration levels.  No IBM quota consumed.

Tests 4 models on N_HW_STEPS test days:
  1. HAR-RV                   classical baseline
  2. QRC single+ent           noiseless sim
  3. QRC dual-pauli           noiseless sim
  4. QRC dual-pauli           IBM-level noise

Note on dual+ent+regime: entanglement spectrum features (Schmidt eigenvalues
of the half-chain reduced density matrix) are highly noise-sensitive — noise
scrambles the Schmidt decomposition, making those features useless under
hardware-level depolarizing noise.  Only Pauli expectation values (single-
qubit marginals) survive noise robustly.  dual+ent+regime is the main
competition model in NOISELESS simulation; dual-pauli is the hardware proxy.

Run from proposed-architecture/:
    .venv/bin/python notebook/run_aer_noise.py

Outputs:
    notebook/qrc_financial_results/ibm_hardware_results.json
    notebook/qrc_financial_results/final_comparison_all_models.png
"""

import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)

# ── Constants ─────────────────────────────────────────────────────────────────
SEED         = 42
N_QUBITS     = 9
RIDGE_ALPHAS = np.logspace(-6, 6, 25)
ANGLE_RANGE  = (-np.pi / 2, np.pi / 2)
TRAIN_FRAC   = 0.80
RESULTS_DIR  = Path("qrc_financial_results")
RESULTS_DIR.mkdir(exist_ok=True)
N_HW_STEPS   = 50

# IBM Heron r2 noise levels
P_DEPOL  = 0.003
GAMMA_AD = 0.0002

np.random.seed(SEED)
START, END = "2010-01-01", "2025-12-31"
RAW = Path("data/raw")


@dataclass(frozen=True)
class ReservoirConfig:
    name: str
    n_qubits: int
    coupling_j: float
    transverse_h: float
    trotter_depth: int
    feature_cols: tuple


SHORT_RESERVOIR = ReservoirConfig("short-memory", N_QUBITS, 0.3, 1.0, 2,
                                   ("log_rv_d", "log_ret", "log_gk"))
LONG_RESERVOIR  = ReservoirConfig("long-memory",  N_QUBITS, 1.2, 0.4, 6,
                                   ("log_rv_w", "log_rv_m", "vix", "vix_rv_spread"))
SHORT_COLS = list(SHORT_RESERVOIR.feature_cols)
LONG_COLS  = list(LONG_RESERVOIR.feature_cols)
HAR_COLS   = ["rv_d", "rv_w", "rv_m"]


# ── 1. Data ───────────────────────────────────────────────────────────────────
def _load_sp500():
    p = RAW / "sp500_ohlcv.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    df = yf.download("^GSPC", start=START, end=END, auto_adjust=True,
                     progress=False, multi_level_index=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].ffill(limit=2).dropna()
    df.index.name = "Date"; df.to_csv(p); return df

def _load_vix(idx):
    p = RAW / "vix.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True).squeeze().reindex(idx).ffill()
    df = yf.download("^VIX", start=START, end=END, auto_adjust=True,
                     progress=False, multi_level_index=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].rename("VIX"); s.index.name = "Date"; s.to_csv(p)
    return s.reindex(idx).ffill()

def _load_oxford(sp500):
    p = RAW / "oxford_man_rv.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    h = np.log(sp500["High"]/sp500["Open"])
    l = np.log(sp500["Low"]/sp500["Open"])
    c = np.log(sp500["Close"]/sp500["Open"])
    gk = (0.5*(h-l)**2 - (2*np.log(2)-1)*c**2).clip(lower=0).rename("rv5")
    gk.to_frame().to_csv(p); return gk.to_frame()

def build_features(sp500, vix, oxford):
    df = sp500[["Open","High","Low","Close"]].copy()
    df["log_ret"] = np.log(df["Close"]/df["Close"].shift(1))
    rv = oxford["rv5"].reindex(df.index).ffill().bfill() if "rv5" in oxford.columns \
         else pd.Series(np.nan, index=df.index)
    if rv.isna().all():
        h = np.log(df["High"]/df["Open"]); l = np.log(df["Low"]/df["Open"])
        c = np.log(df["Close"]/df["Open"])
        rv = (0.5*(h-l)**2 - (2*np.log(2)-1)*c**2).clip(lower=0)
    df["rv"]  = rv
    h = np.log(df["High"]/df["Open"]); l = np.log(df["Low"]/df["Open"])
    c = np.log(df["Close"]/df["Open"])
    df["gk"]          = (0.5*(h-l)**2 - (2*np.log(2)-1)*c**2).clip(lower=0)
    df["rv_d"]        = df["rv"]
    df["rv_w"]        = df["rv"].rolling(5,  min_periods=5).mean()
    df["rv_m"]        = df["rv"].rolling(22, min_periods=22).mean()
    df["log_rv_d"]    = np.log(df["rv_d"].clip(lower=1e-10))
    df["log_rv_w"]    = np.log(df["rv_w"].clip(lower=1e-10))
    df["log_rv_m"]    = np.log(df["rv_m"].clip(lower=1e-10))
    df["log_gk"]      = np.log(df["gk"].clip(lower=1e-10))
    df["vix"]         = vix.reindex(df.index).ffill()
    df["vix_rv_spread"] = (df["vix"]/100)**2/252 - df["rv"]
    df["target_rv"]   = df["rv"].shift(-1)
    df["target_log_rv"] = np.log(df["rv"].shift(-1).clip(lower=1e-10))
    return df.dropna()

print("Loading data...")
sp500  = _load_sp500()
vix    = _load_vix(sp500.index)
oxford = _load_oxford(sp500)
df     = build_features(sp500, vix, oxford)
n_train = int(len(df) * TRAIN_FRAC)
df_train = df.iloc[:n_train]
print(f"  {len(df)} rows  train={n_train}  test={len(df)-n_train}")


# ── 2. HMM ────────────────────────────────────────────────────────────────────
hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=200,
                  random_state=SEED)
hmm.fit(np.log(df_train["rv"].values + 1e-10).reshape(-1, 1))
means     = hmm.means_.flatten()
s         = np.argsort(means)
state_map = {int(s[i]): i for i in range(3)}
smrv      = np.exp(np.sort(means))
thresholds = (float((smrv[0]+smrv[1])/2), float((smrv[1]+smrv[2])/2))

def predict_regimes(rv):
    return np.array([state_map[int(x)]
                     for x in hmm.predict(np.log(np.asarray(rv)+1e-10).reshape(-1,1))])

def regime_post(rv):
    raw = hmm.predict_proba(np.log(np.asarray(rv)+1e-10).reshape(-1,1))
    out = np.zeros_like(raw)
    for rs, ss in state_map.items():
        out[:, ss] = raw[:, rs]
    return out

all_post = regime_post(df["rv"].values)
print("HMM fitted.")


# ── 3. Angle encoding ─────────────────────────────────────────────────────────
def fit_enc(dtr, cols):
    return {c: MinMaxScaler(feature_range=ANGLE_RANGE).fit(dtr[[c]]) for c in cols}

def encode(df, cols, sc):
    out = np.zeros((len(df), len(cols)))
    for j, c in enumerate(cols):
        out[:, j] = np.clip(sc[c].transform(df[[c]]).ravel(), *ANGLE_RANGE)
    return out

short_sc     = fit_enc(df_train, SHORT_COLS)
long_sc      = fit_enc(df_train, LONG_COLS)
short_angles = encode(df, SHORT_COLS, short_sc)
long_angles  = encode(df, LONG_COLS,  long_sc)
lrvd_sc      = short_sc["log_rv_d"]

def angle_to_rv(a):
    return np.exp(lrvd_sc.inverse_transform(
        np.clip(a, *ANGLE_RANGE).reshape(-1, 1)).ravel())

print("Angles encoded.")


# ── 4. Cached features + readouts ────────────────────────────────────────────
npz = RESULTS_DIR / "financial_qrc_features.npz"
if not npz.exists():
    raise FileNotFoundError(f"Run main notebook first: {npz}")
feat_raw = np.load(npz)
feat = {k: feat_raw[k] for k in feat_raw.files}
print(f"Features: {', '.join(f'{k}:{v.shape}' for k, v in feat.items())}")

T_feat = len(feat["target_angle"])
n_tr   = int(T_feat * TRAIN_FRAC)
n_te   = T_feat - n_tr

def fit_ro(X, y, n_tr, n_te):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    m.fit(X[:n_tr], y[:n_tr])
    return {"y_pred": m.predict(X[n_tr:n_tr+n_te]), "model": m}

y = feat["target_angle"]
readouts = {k: fit_ro(feat[k], y, n_tr, n_te)
            for k in ("dual_pauli", "single_long_ent")}
print(f"Readouts fitted.  Train={n_tr}  Test={n_te}")


# ── 5. Metric helpers ─────────────────────────────────────────────────────────
def qlike(yt, yp):
    yt = np.maximum(yt, 1e-10); yp = np.maximum(yp, 1e-10)
    r  = yt / yp
    return float(np.mean(r - np.log(r) - 1.0))

y_true_rv = df["target_rv"].values[n_tr:n_tr+n_te]
har = LinearRegression().fit(df[HAR_COLS].values[:n_train],
                             df["target_rv"].values[:n_train])


# ── 6. Noisy QNode — Pauli expval (noise-robust features) ────────────────────
def make_noisy_qnode(config):
    dev     = qml.device("default.mixed", wires=config.n_qubits)
    n_feats = len(config.feature_cols)

    @qml.qnode(dev, interface="numpy")
    def noisy_reservoir(feature_angles):
        for _ in range(config.trotter_depth):
            for wire in range(config.n_qubits):
                qml.RY(float(feature_angles[wire % n_feats]), wires=wire)
                qml.DepolarizingChannel(P_DEPOL, wires=wire)
                qml.AmplitudeDamping(GAMMA_AD, wires=wire)
            for wire in range(config.n_qubits - 1):
                qml.IsingZZ(2.0 * config.coupling_j, wires=[wire, wire + 1])
                qml.DepolarizingChannel(P_DEPOL, wires=wire)
                qml.DepolarizingChannel(P_DEPOL, wires=wire + 1)
            for wire in range(config.n_qubits):
                qml.RX(2.0 * config.transverse_h, wires=wire)
                qml.DepolarizingChannel(P_DEPOL, wires=wire)
        return ([qml.expval(qml.PauliZ(i)) for i in range(config.n_qubits)] +
                [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i+1))
                 for i in range(config.n_qubits - 1)])

    return noisy_reservoir


# ── 7. Build noisy QNodes ─────────────────────────────────────────────────────
print(f"\nBuilding noisy QNodes (p_depol={P_DEPOL}, gamma_ad={GAMMA_AD})...")
short_noisy = make_noisy_qnode(SHORT_RESERVOIR)
long_noisy  = make_noisy_qnode(LONG_RESERVOIR)


# ── 8. Run noise simulation ───────────────────────────────────────────────────
hw_short_a = short_angles[n_tr:n_tr+N_HW_STEPS+1]
hw_long_a  = long_angles [n_tr:n_tr+N_HW_STEPS+1]

print(f"Running {N_HW_STEPS} steps on IBM-level noise...")
t0 = time.perf_counter()

dual_rows = []
for t in range(N_HW_STEPS):
    pau_s = np.asarray(short_noisy(hw_short_a[t]), dtype=float)
    pau_l = np.asarray(long_noisy (hw_long_a[t]),  dtype=float)
    prev  = np.array([float(hw_short_a[t, 0])])
    dual_rows.append(np.concatenate([pau_s, pau_l, prev]))

    if (t+1) % 10 == 0 or t+1 == N_HW_STEPS:
        elapsed = time.perf_counter() - t0
        eta     = elapsed/(t+1) * (N_HW_STEPS - t - 1)
        print(f"  {t+1:3d}/{N_HW_STEPS}  elapsed {elapsed:5.1f}s  eta {eta:5.1f}s")

hw_feat = np.vstack(dual_rows)
print(f"Done in {time.perf_counter()-t0:.1f}s.  hw_feat={hw_feat.shape}")


# ── 9. Metrics ────────────────────────────────────────────────────────────────
y_true_hw = y_true_rv[:N_HW_STEPS]
n_hw      = len(y_true_hw)

def metrics(y_true, y_pred):
    return (float(np.sqrt(np.mean((y_true - y_pred)**2))),
            qlike(y_true, y_pred))

# HAR-RV
har_pred           = np.maximum(har.predict(df[HAR_COLS].values[n_tr:n_tr+N_HW_STEPS]), 1e-10)
har_rmse,  har_ql  = metrics(y_true_hw, har_pred)

# QRC single+ent (noiseless — cached)
sgl_pred           = angle_to_rv(readouts["single_long_ent"]["y_pred"][:N_HW_STEPS])
sgl_rmse,  sgl_ql  = metrics(y_true_hw, sgl_pred)

# QRC dual-pauli (noiseless — cached)
sim_pred           = angle_to_rv(readouts["dual_pauli"]["y_pred"][:N_HW_STEPS])
sim_rmse,  sim_ql  = metrics(y_true_hw, sim_pred)

# QRC dual-pauli (IBM-noise)
hw_pred            = angle_to_rv(readouts["dual_pauli"]["model"].predict(hw_feat))
hw_rmse,   hw_ql   = metrics(y_true_hw, hw_pred)

print("\n" + "="*60)
print(f"RESULTS — {n_hw} test days  "
      f"(noise: p_depol={P_DEPOL}, gamma={GAMMA_AD})")
print("="*60)
print(f"{'Model':<36} {'RMSE':>11}  {'QLIKE':>8}")
print("-"*59)
print(f"{'HAR-RV (classical)':<36} {har_rmse:.4e}  {har_ql:.4f}")
print(f"{'QRC single+ent (noiseless sim)':<36} {sgl_rmse:.4e}  {sgl_ql:.4f}")
print(f"{'QRC dual-pauli (noiseless sim)':<36} {sim_rmse:.4e}  {sim_ql:.4f}")
print(f"{'QRC dual-pauli (IBM-level noise)':<36} {hw_rmse:.4e}  {hw_ql:.4f}")
print()
print(f"Noise penalty  (noisy vs noiseless): {(hw_rmse-sim_rmse)/sim_rmse*100:+.1f}% RMSE")
print(f"Noisy QRC vs HAR-RV RMSE:            {(har_rmse-hw_rmse)/har_rmse*100:+.1f}%")

hw_results = {
    "mode":    "aer_noise_simulation",
    "n_steps": n_hw,
    "p_depol": P_DEPOL,
    "gamma_ad": GAMMA_AD,
    "hw_rmse":  hw_rmse,  "hw_qlike":  hw_ql,
    "sim_rmse": sim_rmse, "sim_qlike": sim_ql,
    "sgl_rmse": sgl_rmse, "sgl_qlike": sgl_ql,
    "har_rmse": har_rmse, "har_qlike": har_ql,
    "noise_penalty_pct": (hw_rmse - sim_rmse)/sim_rmse*100,
    "noisy_vs_har_pct":  (har_rmse - hw_rmse)/har_rmse*100,
}
(RESULTS_DIR / "ibm_hardware_results.json").write_text(json.dumps(hw_results, indent=2))
print(f"\nMetrics → {RESULTS_DIR}/ibm_hardware_results.json")


# ── 10. 4-model comparison plot ───────────────────────────────────────────────
MODELS = {
    "HAR-RV\n(classical)":          (har_rmse, har_ql, har_pred),
    "QRC single\n(noiseless sim)":  (sgl_rmse, sgl_ql, sgl_pred),
    "QRC dual\n(noiseless sim)":    (sim_rmse, sim_ql, sim_pred),
    "QRC dual\n(IBM-level noise)":  (hw_rmse,  hw_ql,  hw_pred),
}
COLORS = ["#6b6b6b", "#f4a261", "#2176ff", "#e05c00"]

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle(
    f"Classical vs QRC single vs QRC dual (noiseless) vs QRC dual (IBM-level noise)\n"
    f"{n_hw} test days  |  depolarizing p={P_DEPOL}/gate  |  amplitude damping γ={GAMMA_AD}\n"
    f"Noise simulation (not real hardware)  —  main model dual+ent+regime uses noiseless sim",
    fontsize=9, y=1.04,
)

for ax, metric_idx, title in [
    (axes[0], 0, "RMSE"),
    (axes[1], 1, "QLIKE (lower=better)"),
]:
    vals  = [v[metric_idx] for v in MODELS.values()]
    har_v = list(MODELS.values())[0][metric_idx]
    bars  = ax.bar(list(MODELS.keys()), vals, color=COLORS, edgecolor="white")
    ax.axhline(har_v, color="#6b6b6b", linestyle="--", linewidth=1.3,
               alpha=0.8, label="HAR-RV level")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(title)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=7, loc="upper right")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                f"{v:.2e}" if metric_idx == 0 else f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)

ax = axes[2]
t = np.arange(n_hw)
ax.plot(t, y_true_hw, color="black", label="Actual RV", lw=1.5, zorder=4)
for (lbl, (_, _, pred)), color in zip(MODELS.items(), COLORS):
    ax.plot(t, pred, color=color, label=lbl.replace("\n", " "), lw=0.9, alpha=0.85)
ax.set_title("Forecast (50 test days)", fontweight="bold")
ax.set_xlabel("Test day"); ax.set_ylabel("Realized Variance")
ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
out_plot = RESULTS_DIR / "final_comparison_all_models.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot → {out_plot}")
print("Done.")
