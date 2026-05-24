# QuantumEdge GIC 2026

**Team:** QuantumEdge | **Track:** Dynamic Systems Forecasting — Financial Volatility Prediction  
**Competition:** Global Investment Challenge (GIC) 2026

Dual-timescale Quantum Reservoir Computing for financial volatility forecasting and regime detection.

## Key Results

| Model | RMSE (RV) | QLIKE | Regime Acc | Sharpe |
|---|---|---|---|---|
| **QRC dual\_ent\_regime** (ours) | **1.07e-4** | **0.358** | **75.1%** | **1.268** |
| QRC dual\_ent (ours) | 1.08e-4 | 0.362 | 74.4% | 1.208 |
| LSTM-log | 1.26e-4 | 0.383 | 65.8% | 1.719 |
| XGBoost-log | 1.27e-4 | 0.379 | 65.7% | 1.283 |
| HAR-RV (classical) | 1.28e-4 | 0.348 | 65.9% | 1.268 |
| GARCH(1,1) | 1.56e-4 | 0.486 | 31.5% | 0.663 |

**QRC-dual\_ent\_regime vs HAR-RV: +15.9% RMSE, QLIKE readout variant: +14.1% QLIKE**  
Test window: 801 days (20% holdout), S&P 500 realized variance (Garman-Klass).  
All baselines use log-space preprocessing on identical train/test split.

---

## Architecture

Two fixed Ising Hamiltonian reservoirs process complementary timescales simultaneously:

**Short-memory reservoir** (J=0.3, h=1.0, depth=2, features: log\_rv\_d / log\_ret / log\_gk)  
**Long-memory reservoir** (J=1.2, h=0.4, depth=6, features: log\_rv\_w / log\_rv\_m / vix / vix\_rv\_spread)

Both use 9 qubits with multi-feature cyclic encoding (`RY(feature_angles[wire % n_features])`).  
Features: `⟨Zi⟩`, `⟨ZiZi+1⟩` (Pauli) + half-chain entanglement energies `−log λk`.  
Readout: Ridge regression with 3-state HMM regime conditioning.  
RV predictions: `exp(inverse_transform(angle))` — guarantees strict positivity.

---

## Repository Structure

```
QuantumEdge_GIC2026/
│
├── proposed-architecture/          ← QRC implementation (main submission)
│   ├── notebook/
│   │   ├── QuantumEdge_QRC_Financial.ipynb          ← Main financial forecasting notebook
│   │   ├── QuantumEdge_QRC_MNIST.ipynb              ← Expressivity benchmark (n=5/10/15 qubits)
│   │   ├── data/raw/                                ← SP500, VIX, Oxford-Man CSVs
│   │   ├── run_classical_benchmark.py               ← Fair classical baseline runner
│   │   └── qrc_financial_results/                   ← Plots and feature matrices
│   ├── qrc/
│   │   ├── circuits.py     ← PennyLane QNode factory (statevector + hardware)
│   │   ├── features.py     ← Pauli + entanglement feature extractors
│   │   ├── readout.py      ← Ridge readouts + RegimeReadout
│   │   ├── encoding.py     ← MinMaxScaler angle encoding
│   │   └── config.py       ← ReservoirConfig dataclass
│   └── pyproject.toml              ← uv environment spec
│
├── classical-baseline/             ← Full classical benchmark pipeline
│   ├── models/             ← GARCH, HAR-RV, ARIMA, ESN, LSTM, XGBoost
│   ├── utils/              ← Data loading, features, metrics, plots
│   ├── results/            ← metrics.csv + PNG plots
│   └── main.py
│
├── src/quantumedge/                ← Package (data, features, models, pipelines, API)
├── phaseII/                        ← Phase 2 prototype (archived)
└── phaseI/                         ← Phase 1 submission (archived)
```

---

## Reproducing Results

### Proposed QRC method (uv)

```bash
cd proposed-architecture
uv sync           # creates .venv with PennyLane, sklearn, yfinance, hmmlearn

# Run financial forecasting notebook
.venv/bin/jupyter nbconvert \
  --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  notebook/QuantumEdge_QRC_Financial.ipynb \
  --output notebook/QuantumEdge_QRC_Financial_executed.ipynb

# Run fair classical baseline comparison
.venv/bin/python notebook/run_classical_benchmark.py
```

### Classical baselines

```bash
cd classical-baseline
pip install -r requirements.txt
python main.py
```

### src/quantumedge package

```bash
python -m pip install -r requirements/dev.txt
python -m pip install -e . --no-deps

# Classical baselines pipeline
python -m quantumedge.pipelines.classical_baselines

# Quantum QRC pipeline
python -m quantumedge.pipelines.quantum_qrc \
  --datasets mackey_glass lorenz \
  --backend statevector
```

---

## Hardware Path

`proposed-architecture/notebook/qrc_financial_hardware.py` provides a hardware-compatible
companion using Pauli-only observables (EstimatorV2) for IBM Quantum.

Noise test: Aer simulation with IBM Kingston depolarizing profile (p=0.003/gate, γ=0.0002).

---

## Docker

```bash
docker compose up --build
docker compose run --rm pipeline        # classical jobs
docker compose --profile quantum run --rm quantum-sim   # QRC simulator
```

---

## Notebooks at a Glance

| Notebook | Purpose | Key Output |
|---|---|---|
| `QuantumEdge_QRC_Financial.ipynb` | Main pipeline: data → features → readout → metrics | Sections 1–15 |
| `QuantumEdge_QRC_MNIST.ipynb` | Expressivity benchmark across 5/10/15 qubits | Accuracy vs qubit count |
