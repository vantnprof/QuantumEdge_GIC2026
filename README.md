# QuantumEdge GIC 2026

Project code for classical volatility baselines and the proposed dual-timescale
Quantum Reservoir Computing method.

## Setup

Run from the repository root:

```bash
cd QuantumEdge_GIC2026
```

Use your existing conda environment:

```bash
conda activate quantumedge_env
python -m pip install -r requirements/dev.txt
python -m pip install -e . --no-deps
```

Or create a fresh virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/dev.txt
python -m pip install -e . --no-deps
```

Verify the package is importable:

```bash
python -c "import quantumedge; print(quantumedge.__file__)"
```

Run tests:

```bash
pytest -q
```

If you see `ModuleNotFoundError: No module named 'quantumedge'`, run:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Run Classical Baselines

This runs data loading, feature engineering, all classical models, metrics, and
plots:

```bash
python -m quantumedge.pipelines.classical_baselines
```

Run without plots:

```bash
python -m quantumedge.pipelines.classical_baselines --skip-plots
```

Legacy wrapper:

```bash
python classical-baseline/main.py
```

Classical outputs:

```text
artifacts/results/metrics.csv
artifacts/results/plots/
```

## Run Proposed Quantum Method

Full proposed simulator QRC on Mackey-Glass and Lorenz:

```bash
python -m quantumedge.pipelines.quantum_qrc \
  --datasets mackey_glass lorenz \
  --backend statevector
```

Fast smoke test:

```bash
python -m quantumedge.pipelines.quantum_qrc \
  --datasets mackey_glass \
  --backend statevector \
  --n-qubits 3 \
  --n-train 8 \
  --n-test 4 \
  --device-name default.qubit \
  --progress-every 0 \
  --skip-plots
```

Small Aer hardware-shaped run:

```bash
python -m quantumedge.pipelines.quantum_qrc \
  --backend aer \
  --datasets mackey_glass \
  --n-train 40 \
  --n-test 10 \
  --progress-every 10
```

IBM hardware spot-check:

```bash
export QE_IBM_TOKEN="your_token"
export QE_IBM_INSTANCE="your_instance"

python -m quantumedge.pipelines.quantum_qrc \
  --backend ibm \
  --datasets mackey_glass \
  --n-train 40 \
  --n-test 10 \
  --progress-every 10
```

Quantum outputs:

```text
artifacts/results/quantum_qrc_metrics.csv
artifacts/results/quantum_qrc_paper_metrics.csv
artifacts/results/quantum_qrc_predictions.npz
artifacts/results/plots/quantum_qrc_forecasts.png
```

## Docker

Build and run the default stack:

```bash
docker compose up --build
```

Run the classical jobs:

```bash
docker compose run --rm data
docker compose run --rm pipeline
```

Run proposed QRC simulator:

```bash
docker compose --profile quantum run --rm quantum-sim
```

Run Aer QRC check:

```bash
docker compose --profile quantum run --rm quantum-aer
```

Run IBM hardware QRC check:

```bash
QE_IBM_TOKEN="your_token" QE_IBM_INSTANCE="your_instance" \
docker compose --profile quantum-hardware run --rm quantum-hardware
```

Start API:

```bash
docker compose up --build api
```

API checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/quantum/metrics
curl http://localhost:8000/plots
```

## What Is Implemented

Classical baselines:

```text
GARCH, HAR-RV, ARIMA, Gaussian HMM regimes, ESN, LSTM, XGBoost
```

Proposed quantum method:

```text
Dual-timescale QRC with two fixed transverse-field Ising reservoirs.
Short reservoir: J=0.3, h=1.0, p=2.
Long reservoir:  J=1.2, h=0.4, p=6.
Statevector backend: Pauli/ZZ features plus entanglement-spectrum features.
Aer/IBM backend: Pauli/ZZ expectation features only.
Readout: StandardScaler + RidgeCV.
```

## Repository Structure

```text
src/quantumedge/data/           data loading and caching
src/quantumedge/features/       feature engineering and splits
src/quantumedge/models/         classical model modules
src/quantumedge/quantum/        QRC reservoirs, encoders, features, readouts
src/quantumedge/evaluation/     metrics
src/quantumedge/visualization/  plots
src/quantumedge/pipelines/      runnable workflows
src/quantumedge/services/       data/API/quantum service entry points
docs/architecture.md            architecture notes
requirements/                   dependency files
tests/                          smoke tests
```
