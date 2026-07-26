# QuantumEdge — GIC 2026 Phase 3

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_black.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/vantnprof/QuantumEdge_GIC2026.git)


**Team:** QuantumEdge  
**Track:** Track A — Financial Volatility Prediction  
**Challenge:** qBraid / MITRE / JonesTrading Global Industry Challenge 2026

**Team members:**
- Van Tien Nguyen — Team lead
- Judith Sarjeant — Project Manager / Financial-domain lead
- Shema Nourine — Technical lead
  - Hasarindu Perera — Data and machine-learning lead

---

## Project

**Dual-Timescale Quantum Reservoir Computing for Financial Volatility Forecasting**

A fixed-Ising quantum reservoir with short- and long-timescale feature channels, a classical ridge readout, volatility-regime classification, Oxford-Man realized-variance validation, shot-budget and noise studies, and targeted IBM hardware validation.

---

## Recommended judge workflow

1. Click **Launch on qBraid** above.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Run notebooks in order:

| Step | Notebook | Role |
|---|---|---|
| 1 | `notebooks/01_QuantumEdge_Reproducibility.ipynb` | Main financial-volatility results |
| 2 | `notebooks/02_QuantumEdge_Graphical_Results.ipynb` | Figures from generated evidence |
| 3 | `notebooks/03_QuantumEdge_Package_Verification.ipynb` | File, metric, and credential checks |

> **IBM hardware**: keep `RUN_HARDWARE_NOW = False` (default). Pre-validated IBM evidence is loaded automatically from `results/hardware_validation.json`.  
> **Run profile**: use `RUN_PROFILE = 'QUICK'` for a smoke test, `'FULL'` for complete reproduction.

### Supplementary notebook (optional)

`notebooks/supplementary/04_QuantumEdge_MNIST_Supplementary.ipynb` — expressivity and cross-domain validation. Not part of the Track A financial benchmark. Run after Notebooks 01–03 if desired.

---

## Setup

```bash
python -m pip install -r requirements.txt
```

The notebooks also install missing core dependencies at runtime.

---

## Inputs

| File | Description |
|---|---|
| `data/market_data.csv` | S&P 500 OHLCV (Yahoo Finance, 2010–2025) |
| `data/oxman_spx.csv` | Oxford-Man realized variance |
| `data/qrc_pauli_features.npy` | Pre-computed QRC reservoir features |
| `data/qrc_pauli_features.meta.json` | Feature metadata |
| `data/mnist.npz` | Downloaded automatically by Notebook 4 if absent |
| `results/hardware_validation.json` | Pre-validated IBM hardware evidence |
| `results/hardware_observables.csv` | IBM hardware observable pairs |

---

## Outputs

Notebooks write to `results/`, `figures/`, and `dashboard_figures/` at the repository root.

Key output files:

- `results/headline_metrics.csv` — primary benchmark (RMSE, QLIKE, Sharpe, regime accuracy)
- `results/oxford_man_metrics.csv` — Oxford-Man validation
- `results/forecast_predictions.csv` / `results/forecast_significance.csv`
- `results/transition_metrics.csv`
- `results/noise_zne.csv` / `results/amplitude_damping.csv`
- `results/ablation.csv` / `results/reservoir_scaling.csv`
- `results/final_audit.csv` — all checks must pass before ZIP is frozen

---

## Completed reference run

`completed_reference_runs/full_run/` contains a fully executed run including IBM hardware validation:

- `executed_notebooks/` — notebooks with all outputs saved
- `results/` — all generated CSVs and JSON artifacts
- `figures/` / `dashboard_figures/` — all generated figures

The clean notebooks in `notebooks/` must generate fresh outputs independently and must not load from this folder.

---

## Limitations

- Results demonstrate empirical forecasting lift from fixed quantum-generated features, not asymptotic quantum advantage.
- IBM validation uses representative hardware-native reservoir observables, not the full forecasting pipeline on QPU.
- Oxford-Man comparison is competitive with HAR-RV; statistical superiority is not claimed beyond the reported significance tests.
- Notebook 04 (MNIST) is a supplementary expressivity check, not a state-of-the-art image-classification result.
- Hardware outputs vary with device calibration; stored job evidence is retained for auditability.
