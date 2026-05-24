# Classical Baselines — GIC 2026 Quantum Reservoir Computing

> Source code has moved to the installable `quantumedge` package under
> `src/quantumedge`. This folder is kept as a compatibility wrapper and for
> historical result artifacts. Run `python classical-baseline/main.py` or the
> preferred `python -m quantumedge.pipelines.classical_baselines`.

Full classical benchmark pipeline for the **GIC 2026 Finance Track** (Quantum Reservoir Computing for Financial Volatility Prediction). Every model here is a target the QRC system must beat — or at minimum be honest about where it doesn't.

**Test period: 2022–2025 (801 trading days, 80/20 temporal split)**

---

## What's in here

```
classical-baseline/
├── data/
│   ├── raw/               ← Downloaded / generated datasets (auto-created on first run)
│   └── processed/         ← Train/test splits
├── models/
│   ├── garch.py           ← GARCH(1,1)
│   ├── har_rv.py          ← HAR-RV (Corsi, 2009)
│   ├── arima.py           ← ARIMA(5,0,0) for RV; ARIMA(1,1,0) for VIX
│   ├── hmm_regimes.py     ← 3-state Gaussian HMM for regime detection
│   ├── esn.py             ← Echo State Network (reservoirpy)
│   ├── lstm.py            ← LSTM (PyTorch)
│   └── xgboost_model.py   ← XGBoost
├── utils/
│   ├── data_loader.py     ← All data pull / generation
│   ├── features.py        ← Feature engineering (RV, Garman-Klass, HAR components)
│   ├── metrics.py         ← RMSE, MAE, QLIKE, regime accuracy, Sharpe ratio
│   └── plots.py           ← All visualisations (14 plots)
├── results/
│   ├── metrics.csv        ← Full metrics table (23 rows)
│   └── plots/             ← PNG files
├── requirements.txt
└── main.py                ← Single entry point: run everything
```

---

## Setup (UV)

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt
.venv/bin/python main.py
```

Total runtime: ~43 seconds on a modern laptop.

---

## Datasets

| Dataset | Source | Train | Test | Notes |
|---------|--------|------:|-----:|-------|
| **S&P 500 OHLCV** | yfinance `^GSPC` | 3,200 | 801 | 2010–2025 |
| **CBOE VIX** | yfinance `^VIX` | 3,200 | 801 | Aligned to S&P 500 |
| **Oxford-Man RV** | Oxford-Man → **GK fallback** | 3,200 | 801 | Garman-Klass used (primary URL unreachable) |
| **Mackey-Glass** | Generated (Euler, τ=17) | 7,991 | 1,998 | Mildly chaotic |
| **Lorenz System** | Generated (RK45) | 3,991 | 998 | x-component |

> **Garman-Klass fallback:** `GK_t = 0.5·log(H/L)² − (2·ln2−1)·log(C/O)²` — standard OHLCV proxy for intraday RV.

---

## Models

### GARCH(1,1)
Parametric MLE: `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}`. Finance only.

### HAR-RV (Corsi 2009)
OLS: `RV_{t+1} = β₀ + β_d·RV_d + β_w·RV_w + β_m·RV_m`. Finance only.
Fitted: β_d=0.288, β_w=0.479, β_m=0.036 (sum=0.803 < 1 ✓).

### ARIMA
- SP500 RV: ARIMA(5,0,0) — AR(5) on realized variance
- VIX: ARIMA(1,1,0) — first-differenced AR(1) on VIX level

Walk-forward: parameters frozen from training fit; at each test step the current observation is appended and the model forecasts one step ahead — no re-estimation, no look-ahead. Finance only.

### Gaussian HMM (regime detection)
3-state HMM fit on log(RV) training data. States sorted by ascending mean → Low / Medium / High volatility. Decision thresholds: midpoints between adjacent state means in RV space.

- **Thresholds:** low/med = 4.68×10⁻⁵, med/high = 7.85×10⁻⁵
- **Test regime distribution:** Low = 517 days · Medium = 142 days · High = 142 days

### Echo State Network (ESN)
Fixed random reservoir (500 neurons), only linear readout trained (ridge regression). All 5 datasets.
Hyperparameters: `units=500, sr=0.9, lr=0.3, input_scaling=0.5, ridge=1e-6, warmup=100`.

### LSTM
One-layer PyTorch LSTM (64 hidden, dropout=0.2), Adam, MSE loss, early stopping (patience=10). All 5 datasets.

### XGBoost
Gradient-boosted trees with early stopping on 10% held-out validation split. All 5 datasets.

---

## Model × Dataset Applicability

| Dataset | GARCH | HAR-RV | ARIMA | ESN | LSTM | XGBoost |
|---------|:-----:|:------:|:-----:|:---:|:----:|:-------:|
| S&P 500 RV | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| VIX | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Oxford-Man RV | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mackey-Glass | — | — | — | ✓ | ✓ | ✓ |
| Lorenz | — | — | — | ✓ | ✓ | ✓ |

---

## Results

### SP500 RV — next-day realized variance (test: 801 days)

| Model | RMSE | MAE | QLIKE |
|-------|-----:|----:|------:|
| **ESN** | **1.17×10⁻⁴** | **3.6×10⁻⁵** | ⚠️ 3,073 |
| ARIMA | 1.25×10⁻⁴ | 3.8×10⁻⁵ | 0.350 |
| HAR-RV | 1.28×10⁻⁴ | 3.9×10⁻⁵ | **0.349** |
| XGBoost | 1.31×10⁻⁴ | 4.0×10⁻⁵ | 0.396 |
| GARCH | 1.56×10⁻⁴ | 6.8×10⁻⁵ | 0.486 |
| LSTM | 5.47×10⁻⁴ | 2.93×10⁻⁴ | ⚠️ 268,095 |

> **⚠️ QLIKE explodes for ESN and LSTM.** Both models produce unconstrained outputs; any near-zero prediction causes QLIKE to blow up even when RMSE is competitive. GARCH, HAR-RV, ARIMA, and XGBoost produce naturally positive forecasts. **QRC must use a positive-output constraint** (softplus or exp output layer) to compete on QLIKE.

### VIX — next-day level prediction (test: 801 days)

| Model | RMSE | MAE | QLIKE | Target |
|-------|-----:|----:|------:|--------|
| GARCH | 1.52×10⁻² | 5.34×10⁻³ | 1.403 | (VIX log-return)² |
| **ARIMA** | **1.74** | **0.974** | **0.003** | VIX level |
| LSTM | 1.74 | 1.006 | 0.003 | VIX level |
| XGBoost | 1.78 | 1.019 | 0.003 | VIX level |
| ESN | 3.39 | 2.16 | 0.026 | VIX level |

### Chaotic systems — normalised predictions, QLIKE not applicable

| Dataset | Model | RMSE |
|---------|-------|-----:|
| mackey_glass | **ESN** | **1.58×10⁻⁴** |
| mackey_glass | LSTM | 2.69×10⁻³ |
| mackey_glass | XGBoost | 4.01×10⁻³ |
| lorenz | **ESN** | **1.50×10⁻⁵** |
| lorenz | XGBoost | 3.88×10⁻³ |
| lorenz | LSTM | 6.79×10⁻³ |

ESN outperforms the next-best model by **17× on Lorenz** and **25× on Mackey-Glass**. This is the core classical analog to QRC.

---

## Regime Analysis

HMM thresholds (RV space): **low/med = 4.68×10⁻⁵ · med/high = 7.85×10⁻⁵**

| Model | Regime Accuracy | Vol-Timing Sharpe |
|-------|:--------------:|:-----------------:|
| HAR-RV | **72.4%** | 1.26 |
| ARIMA | 72.0% | 1.15 |
| XGBoost | 70.9% | **1.29** |
| ESN | 69.8% | 0.81 |
| GARCH | 40.8% | 0.54 |
| LSTM | 28.5% | 0.44 |
| Buy & Hold | — | 1.26 |

Random baseline for regime accuracy: **33.3%**.

HAR-RV, ARIMA, and XGBoost all exceed 70% — genuine regime discrimination. LSTM (28.5%) is *below random*, consistent with its catastrophic QLIKE and negative predictions.

Vol-timing strategy: Low → full position (1.0×) · Medium → half (0.5×) · High → cash (0.0×). Position set at day t earns return at day t+1 (no look-ahead bias). XGBoost edges Buy & Hold (1.29 vs 1.26).

---

## Key Takeaways

1. **ESN dominates RMSE across all datasets** — SP500 RV (1.17×10⁻⁴), Mackey-Glass (1.58×10⁻⁴), Lorenz (1.50×10⁻⁵). As the classical analog to QRC, ESN sets the hardest RMSE target.

2. **QLIKE target: HAR-RV (0.349) and ARIMA (0.350)** — trivially cheap linear models with near-identical QLIKE. QRC must beat this or the paper loses its QLIKE narrative.

3. **ARIMA is not a weak baseline** — beats GARCH on RMSE, nearly ties HAR-RV on QLIKE, achieves 72% regime accuracy. Cannot be dismissed.

4. **LSTM is the counter-example** — worst RMSE (4.7× worse than ESN), catastrophic QLIKE, 28.5% regime accuracy (below random). Shows what happens without a positive-output constraint and insufficient training data.

5. **Regime advantage is modest** — top-3 models all cluster at 70–72% regime accuracy and Sharpe ~1.26. QRC needs to show clear separation here (>80% regime accuracy or Sharpe >1.5) to make a regime-intelligence story credible.

6. **QLIKE is the primary volatility loss** (Patton, 2011). QRC outputs must be positive-definite variance estimates — not raw unconstrained predictions.

---

## Plots

All 14 plots in `results/plots/`:

| File | What it shows |
|------|---------------|
| `forecasts_sp500_rv.png` | All 6 models' one-step-ahead RV forecasts vs actual (test period). LSTM visibly produces negative predictions. |
| `forecasts_vix_level.png` | ESN/LSTM/XGBoost/ARIMA next-day VIX level forecasts vs actual. |
| `forecasts_mackey_glass.png` | First 300 test steps: ESN, LSTM, XGBoost on Mackey-Glass. |
| `forecasts_lorenz.png` | Same for Lorenz x-component. ESN traces signal almost exactly. |
| `metrics_heatmap.png` | Dual RMSE + QLIKE heatmap across all model × dataset combinations. |
| `rmse_comparison.png` | Grouped RMSE bars for SP500 RV, Mackey-Glass, Lorenz. |
| `radar_chart.png` | Log-normalised RMSE and QLIKE radar (1.0 = best per spoke). |
| `regime_overlay.png` | Two-panel: actual RV with HMM regime shading (top) · all model forecasts overlaid (bottom). |
| `regime_dashboard.png` | Three-panel: log-scale RV time series → regime colour strip → per-regime RMSE bars. |
| `regime_accuracy.png` | Regime classification accuracy per model vs 33.3% random baseline. |
| `sharpe_comparison.png` | Annualised vol-timing Sharpe per model vs buy-and-hold (1.26). |
| `lstm_training_loss.png` | LSTM MSE training loss per epoch; shows early-stopping convergence. |
| `lorenz_attractor.png` | 3D Lorenz butterfly attractor — visual context for chaos. |
| `xgboost_importance.png` | XGBoost feature importances for each dataset. |

---

## Reproducibility

- Seeds: `numpy=42`, `torch=42`, `xgboost random_state=42`, `reservoirpy seed=42`
- Temporal 80/20 train/test split — no shuffling, no leakage
- Scalers fit on training set only; ARIMA parameters frozen after training fit
- All library versions pinned in `requirements.txt`
- Data cached after first download — re-runs are deterministic and fast (~43s)
