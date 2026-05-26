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

# Same classical benchmark through the shared benchmark config
python scripts/run_classical_benchmark.py --no-write-plots

# Quantum QRC pipeline
python -m quantumedge.pipelines.quantum_qrc \
  --datasets mackey_glass lorenz \
  --backend statevector

# Optional MNIST expressivity benchmark for report appendix / Phase 3 evidence
python scripts/run_mnist_benchmark.py \
  --qubits 5 10 15 \
  --n-train 10000 \
  --n-test 2000

# Tiny smoke check for the MNIST benchmark code path
python scripts/run_mnist_benchmark.py --quick-smoke
```

---

## Hardware Path

`proposed-architecture/notebook/qrc_financial_hardware.py` provides a hardware-compatible
companion using Pauli-only observables (EstimatorV2) for IBM Quantum.

Noise test: Aer simulation with IBM Kingston depolarizing profile (p=0.003/gate, γ=0.0002).

---

## Generate The Report Figure

One-command Phase II result script. This executes methods and builds the
comparison figure from the metrics produced by the run:

```bash
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/quantum.txt
python phaseII_result.py
```

Quick environment check. This runs only a tiny QRC job and still writes the
same PNG/PDF/CSV artifact types:

```bash
python phaseII_result.py --quick-smoke
```

Outputs:

```text
artifacts/results/figures/phaseII_result.png
artifacts/results/figures/phaseII_result.pdf
artifacts/results/figures/phaseII_result_data.csv
```

The default Phase II run uses the notebook reservoir configuration:

```text
dataset: sp500_rv
short reservoir: n=9, J=0.3, h=1.0, depth=2
long reservoir : n=9, J=1.2, h=0.4, depth=6
fairness       : QRC uses the same temporal train/test split as classical rows
```

For `sp500_rv` and `oxford_man_rv`, `phaseII_result.py` runs the notebook-style
financial QRC variants:

```text
QRC Dual
QRC Dual+Ent
QRC Dual+Ent+Regime
QRC Single+Ent
QRC Dual+Ent Regime-Blend
QRC Dual+Ent *
QRC Dual+Ent+Regime *
```

The `*` variants use the QLIKE-optimised readout from the notebook. Classical
rows come from the same classical benchmark pipeline.

Use existing classical metrics but rerun the proposed QRC methods:

```bash
python phaseII_result.py \
  --no-run-classical \
  --classical-csv artifacts/results/metrics.csv \
  --basename phaseII_result_reuse_classical
```

Run all supported datasets and write consolidated method-by-dataset results:

```bash
python phaseII_result.py \
  --all-datasets \
  --basename phaseII_all
```

All-dataset outputs:

```text
artifacts/results/phaseII_all_all_datasets_data.csv
artifacts/results/phaseII_all_split_audit.csv
artifacts/results/phaseII_all_coverage.csv
artifacts/results/figures/phaseII_all_<dataset>_<target_type>.png
artifacts/results/figures/phaseII_all_<dataset>_<target_type>.pdf
artifacts/results/figures/phaseII_all_<dataset>_<target_type>_data.csv
```

Rows are comparable within the same `dataset` and `target_type`. The split
audit CSV records the train/test pair used by each comparison group.

Fast actual-run check. This runs QRC only on a tiny Mackey-Glass split and
generates a figure from the metrics produced by that run:

```bash
python scripts/make_report_figure.py \
  --dataset mackey_glass \
  --no-run-classical \
  --quantum-n-qubits 3 \
  --quantum-n-train 8 \
  --quantum-n-test 4 \
  --device-name default.qubit \
  --progress-every 0 \
  --basename actual_qrc_smoke
```

Full 800-day report-style run for S&P 500 RV. This runs the classical pipeline
and the QRC variants, then builds the figure from actual run metrics. By
default the script forces QRC to use the same temporal train/test split as the
classical benchmark when both are included.

```bash
python scripts/make_report_figure.py \
  --dataset sp500_rv \
  --quantum-n-qubits 9 \
  --progress-every 250 \
  --basename actual_sp500_800day
```

Outputs:

```text
artifacts/results/figures/<basename>.png
artifacts/results/figures/<basename>.pdf
artifacts/results/figures/<basename>_data.csv
```

Use existing classical metrics but rerun QRC:

```bash
python scripts/make_report_figure.py \
  --dataset sp500_rv \
  --no-run-classical \
  --classical-csv artifacts/results/metrics.csv \
  --quantum-n-qubits 5 \
  --basename actual_sp500_qrc_5q
```

Run a smaller non-comparable QRC-only figure for debugging:

```bash
python scripts/make_report_figure.py \
  --dataset sp500_rv \
  --no-run-classical \
  --no-match-classical-split \
  --quantum-n-qubits 5 \
  --quantum-n-train 500 \
  --quantum-n-test 100 \
  --basename debug_sp500_qrc_5q
```

Only recreate the old fixed-value presentation figure:

```bash
python scripts/make_report_figure.py \
  --reported-only \
  --basename reported_summary_figure
```

## Quantum Variants From The Notebooks

The statevector command runs these notebook variants in one execution:

```text
QRC dual Pauli
QRC dual Pauli+Ent
QRC single long Pauli+Ent
```

Run the exact Phase 2 notebook ablation:

```bash
python -m quantumedge.pipelines.quantum_qrc \
  --datasets mackey_glass lorenz \
  --backend statevector \
  --n-qubits 9 \
  --short-j 0.3 \
  --short-h 1.0 \
  --short-depth 2 \
  --long-j 1.2 \
  --long-h 0.4 \
  --long-depth 6 \
  --n-train 1000 \
  --n-test 500
```

Run a smaller version of the same ablation:

```bash
python -m quantumedge.pipelines.quantum_qrc \
  --datasets mackey_glass lorenz \
  --backend statevector \
  --n-qubits 5 \
  --n-train 200 \
  --n-test 100 \
  --output-prefix qrc_5q_small
```

Run the hardware-compatible notebook variants:

```text
QRC dual Pauli
QRC single long Pauli
```

Aer hardware-shaped run:

```bash
python -m quantumedge.pipelines.quantum_qrc \
  --backend aer \
  --datasets mackey_glass \
  --n-qubits 9 \
  --n-train 40 \
  --n-test 10 \
  --progress-every 10 \
  --output-prefix qrc_aer_hw_shape
```

IBM hardware-shaped run:

```bash
export QE_IBM_TOKEN="your_token"
export QE_IBM_INSTANCE="your_instance"

python -m quantumedge.pipelines.quantum_qrc \
  --backend ibm \
  --datasets mackey_glass \
  --n-qubits 9 \
  --n-train 40 \
  --n-test 10 \
  --progress-every 10 \
  --output-prefix qrc_ibm_hw_shape
```

Qubit-count sweep from the Phase 2 notes:

```bash
for n in 5 8 10 12 15; do
  python -m quantumedge.pipelines.quantum_qrc \
    --datasets mackey_glass lorenz \
    --backend statevector \
    --n-qubits "$n" \
    --n-train 200 \
    --n-test 100 \
    --output-prefix "qrc_${n}q"
done
```

Trotter-depth sweep from the Phase 2 notes:

```bash
for p in 2 4 6; do
  python -m quantumedge.pipelines.quantum_qrc \
    --datasets mackey_glass lorenz \
    --backend statevector \
    --n-qubits 5 \
    --short-depth "$p" \
    --long-depth "$p" \
    --n-train 200 \
    --n-test 100 \
    --output-prefix "qrc_depth_${p}"
done
```

Memory-separation sweep:

```bash
python -m quantumedge.pipelines.quantum_qrc \
  --datasets mackey_glass lorenz \
  --backend statevector \
  --n-qubits 5 \
  --short-j 0.1 \
  --short-h 1.0 \
  --short-depth 2 \
  --long-j 1.5 \
  --long-h 0.3 \
  --long-depth 6 \
  --n-train 200 \
  --n-test 100 \
  --output-prefix qrc_memory_separation
```

Finance extension variants:

```bash
python -m quantumedge.pipelines.quantum_qrc \
  --datasets sp500_rv \
  --backend statevector \
  --n-qubits 5 \
  --n-train 500 \
  --n-test 100 \
  --output-prefix qrc_sp500_rv
```

```bash
python -m quantumedge.pipelines.quantum_qrc \
  --datasets vix \
  --backend statevector \
  --n-qubits 5 \
  --n-train 500 \
  --n-test 100 \
  --output-prefix qrc_vix
```

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
