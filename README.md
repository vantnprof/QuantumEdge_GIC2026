# QuantumEdge — GIC 2026

**Team:** QuantumEdge | **Track:** Finance | **Competition:** Global Investment Challenge 2026

Quantum Reservoir Computing for Financial Volatility Prediction and Regime Detection.

---

## Repository Structure (for now)

```
QuantumEdge_GIC2026/
├── classical-baseline/    ← Full classical benchmark pipeline (run this first)
│   ├── models/            ← GARCH, HAR-RV, ARIMA, HMM, ESN, LSTM, XGBoost
│   ├── utils/             ← Data loading, features, metrics, plots
│   ├── results/           ← metrics.csv + all PNG plots
│   ├── main.py            ← Single entry point
│   ├── requirements.txt
│   └── README.md          ← Detailed setup and results
└── README.md              ← This file
```

---

## Classical Baselines

The `classical-baseline/` folder establishes the benchmarks every QRC variant must beat. It covers:

- **6 models:** GARCH(1,1), HAR-RV, ARIMA, Echo State Network, LSTM, XGBoost
- **5 datasets:** S&P 500 RV, CBOE VIX, Oxford-Man RV, Mackey-Glass, Lorenz
- **Metrics:** RMSE, MAE, QLIKE (Patton 2011), regime classification accuracy, vol-timing Sharpe
- **Regime detection:** 3-state Gaussian HMM (Low / Medium / High volatility)
- **13 plots** including regime overlays, dashboards, radar charts

See [`classical-baseline/README.md`](classical-baseline/README.md) for full setup, results, and methodology.

