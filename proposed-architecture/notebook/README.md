# QuantumEdge — Notebooks

**Team QuantumEdge | GIC 2026 | Dynamic Systems Forecasting — Financial Volatility Track**

---

## Directory Structure

```
notebook/
├── QuantumEdge_QRC_Financial.ipynb   ← Main submission notebook (financial forecasting)
├── QuantumEdge_QRC_MNIST.ipynb       ← Expressivity benchmark (MNIST, competition requirement)
├── data/raw/                          ← Downloaded market data (SP500, VIX, Oxford-Man)
├── qrc_financial_results/             ← Financial notebook outputs (plots, CSVs, features)
└── mnist_results/                     ← MNIST notebook outputs
```

---

## Notebook 1 — Financial Volatility Forecasting

### What it does
Dual-timescale QRC for one-day-ahead S&P 500 realized variance forecasting.

- **Short reservoir** (J=0.3, h=1.0, depth=2): daily RV features — `log_rv_d`, `log_ret`, `log_gk`
- **Long reservoir** (J=1.2, h=0.4, depth=6): weekly/monthly features — `log_rv_w`, `log_rv_m`, `vix`, `vix_rv_spread`
- 9 qubits per reservoir, multi-feature cyclic encoding
- Readout: Ridge regression on Pauli + entanglement spectrum features + HMM regime conditioning
- RV predictions via `exp(inverse_transform(angle))` — structurally positive

### Key results (800-day test window, 80/20 split)

| Model | RMSE | QLIKE | Regime Acc | Sharpe |
|---|---|---|---|---|
| **QRC dual+ent+regime** | **1.08e-4** | **0.314** | **79.5%** | **1.628** |
| QRC dual+ent | 1.08e-4 | 0.362 | 74.4% | 1.208 |
| HAR-RV (classical) | 1.28e-4 | 0.348 | 65.9% | 1.268 |
| QRC single-long+ent | 1.32e-4 | 0.374 | 72.6% | 1.138 |
| ESN-500-log (classical reservoir) | see notebook | see notebook | — | — |
| Buy-and-hold Sharpe | — | — | — | 1.266 |

**QRC dual vs HAR-RV: +15.6% RMSE, +9.8% QLIKE**
**QRC dual vs single reservoir: +18.3% RMSE**

### Section guide

| Section | Content |
|---|---|
| 1–5 | Data loading, HMM regime fitting, log-space angle encoding |
| 6–8 | Ising reservoir circuits, feature extraction, Ridge readout |
| 9–10 | Metrics (QLIKE, Sharpe, regime accuracy), fair classical comparison |
| 11 | Ablation plots and forecast visualisation |
| 12–13 | Paper numbers, Phase 3 notes |
| 14 | Qubit count sweep — n ∈ {5, 9, 15} on financial data |
| 15 | Noise robustness — depolarizing + amplitude damping (IBM Heron r2 levels) |
| 16 | IBM hardware section — see note below |

### IBM hardware note

Section 16 in the notebook requires an IBM Quantum token and available quota. The standalone
script `run_aer_noise.py` provides an equivalent comparison using **local Aer noise
simulation at IBM Heron r2 noise levels** (depolarizing p=0.003/gate, amplitude damping
γ=0.0002), which requires no quota and produces `final_comparison_all_models.png`.

**This is not real hardware execution.** It is a noise simulation calibrated to match
IBM Heron r2 error rates. The plot and JSON results label this explicitly as
"IBM-level noise" — not "IBM hardware". Real hardware results would require an IBM account
with sufficient Open Plan quota (~10 minutes for 50 steps × 2 circuits × individual jobs,
or a paid Session plan for batched execution).

**Models tested (4):** HAR-RV · QRC single+ent (noiseless) · QRC dual-pauli (noiseless) ·
QRC dual-pauli (IBM-noise).

**Note on dual+ent+regime under noise:** Entanglement spectrum features (Schmidt eigenvalues
of the half-chain reduced density matrix) are highly sensitive to depolarizing noise —
noise scrambles the Schmidt decomposition, making those features unusable. Only Pauli
expectation values (single-qubit marginals) survive noise robustly. `dual+ent+regime` is
the main competition model in **noiseless simulation**; `dual_pauli` is the hardware proxy.
Entanglement features on hardware require classical shadows tomography (Phase 3 extension).

```bash
cd proposed-architecture
.venv/bin/python notebook/run_aer_noise.py
```

### Run main notebook

```bash
cd proposed-architecture
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebook/QuantumEdge_QRC_Financial.ipynb
```

---

## Notebook 2 — MNIST Expressivity Benchmark

### What it does
Runs the same Ising reservoir (J=0.8, h=0.5, depth=4) on MNIST digit classification across
n ∈ {5, 10, 15} qubits to demonstrate that reservoir expressivity scales with qubit count.

- Encoding: PCA-n on training images → MinMaxScaler → [-π/2, π/2] angles (no leakage)
- Readout: Ridge classifier on Pauli + entanglement features
- Classical baseline: PCA-n + Ridge directly (no reservoir) — same information budget

### Results (10 000 train / 2 000 test, stratified)

| N qubits | PCA variance | Classical PCA+Ridge | QRC Pauli+Ent | Scaling gain |
|---|---|---|---|---|
| 5 | 33.0% | 57.3% | 51.8% | — |
| 10 | 48.6% | 72.9% | 63.8% | +12.1pp vs n=5 |
| 15 | 57.8% | 75.5% | 68.3% | +16.5pp vs n=5 |

### Honest interpretation

**QRC does not beat classical PCA+Ridge on MNIST with these fixed reservoir parameters.**

The gap exists because:
1. PCA already extracts the dominant linear structure from images. A Ridge classifier on those components is highly competitive.
2. The Ising reservoir (J=0.8, h=0.5) is fixed and not tuned for image structure — it applies the same chaotic dynamics regardless of task.
3. Fixed random reservoirs are a general-purpose non-linear feature extractor; they do not adapt to task-specific geometry the way trained networks do.

**What the results do show:**
- QRC accuracy scales clearly with qubit count: 51.8% → 63.8% → 68.3% (+16.5pp from n=5 to n=15)
- This scaling is the competition's cross-team comparison metric — it demonstrates that more Hilbert space dimensionality (2ⁿ) enables richer non-linear feature interactions
- For task-specific forecasting (financial), the reservoir IS tuned to the HAR feature structure and the results are strong

**The main competition claim lives in the financial forecasting notebook, not here.**

### Run

```bash
cd proposed-architecture
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebook/QuantumEdge_QRC_MNIST.ipynb
```

---

## Environment

Uses the `uv` environment in `proposed-architecture/.venv/` (Python 3.11).  
Does **not** touch system Python.

```bash
cd proposed-architecture
uv sync   # creates .venv if not present
```

Key packages: `pennylane==0.43.0`, `scikit-learn`, `hmmlearn`, `yfinance`, `matplotlib`.
