# QuantumEdge Architecture

## Data Flow

1. `quantumedge.services.data_service`
   - Downloads S&P 500 and VIX market data.
   - Attempts Oxford-Man realized volatility, then falls back to Garman-Klass RV.
   - Generates Mackey-Glass and Lorenz synthetic series.
   - Writes caches under `QE_DATA_DIR` or `artifacts/data`.

2. `quantumedge.features.builders`
   - Builds financial RV, VIX, and chaotic-system feature matrices.
   - Applies temporal 80/20 splits.
   - Fits scalers on training windows only.

3. `quantumedge.models`
   - Contains model-specific method modules: GARCH, HAR-RV, ARIMA, HMM regimes,
     ESN, LSTM, and XGBoost.
   - Each module exposes small `run_*` functions used by the pipeline.

4. `quantumedge.quantum`
   - Implements the proposed dual-timescale Quantum Reservoir Computing
     architecture from the Phase 2 draft.
   - Provides scalar angle encoders, fixed Ising reservoir circuits,
     statevector feature extraction, hardware-compatible Pauli feature
     extraction, and ridge readouts.

5. `quantumedge.pipelines.classical_baselines`
   - Orchestrates load -> feature build -> train/evaluate -> plot.
   - Writes `metrics.csv` and plot PNGs under `QE_RESULTS_DIR` or
     `artifacts/results`.

6. `quantumedge.pipelines.quantum_qrc`
   - Runs the proposed QRC method on Mackey-Glass, Lorenz, and optionally
     finance scalar series.
   - Writes `quantum_qrc_metrics.csv`, `quantum_qrc_paper_metrics.csv`,
     feature caches, prediction arrays, and QRC forecast plots.

7. `quantumedge.services.api`
   - Exposes health, architecture, metrics, plot listing, plot downloads, and a
     background trigger for the classical baseline run.

## Services

| Service | Type | Command | Responsibility |
| --- | --- | --- | --- |
| `data` | one-shot job | `python -m quantumedge.services.data_service` | Build/cache datasets and write a manifest. |
| `pipeline` | one-shot job | `python -m quantumedge.pipelines.classical_baselines` | Train/evaluate all baseline methods and write artifacts. |
| `api` | long-running HTTP service | `uvicorn quantumedge.services.api:app --host 0.0.0.0 --port 8000` | Read and expose generated artifacts. |
| `quantum-sim` | one-shot job, profile `quantum` | `python -m quantumedge.pipelines.quantum_qrc --backend statevector` | Run simulator QRC with Pauli plus entanglement features. |
| `quantum-aer` | one-shot job, profile `quantum` | `python -m quantumedge.pipelines.quantum_qrc --backend aer` | Run small hardware-shaped Aer QRC validation. |
| `quantum-hardware` | one-shot job, profile `quantum-hardware` | `python -m quantumedge.pipelines.quantum_qrc --backend ibm` | Run small IBM hardware spot-check using `QE_IBM_TOKEN`/`QE_IBM_INSTANCE`. |

The services communicate through mounted artifact volumes. That keeps model
training as an explicit batch workflow while still exposing a lightweight API
for downstream consumers.

## Quantum Method

The proposed method is a dual-timescale QRC:

| Component | Short Reservoir | Long Reservoir |
| --- | ---: | ---: |
| Qubits | 9 default | 9 default |
| Coupling `J` | `0.3` | `1.2` |
| Transverse field `h` | `1.0` | `0.4` |
| Trotter depth `p` | `2` | `6` |

Each scalar input is scaled on the training segment only, clipped to
`[-pi/2, pi/2]`, and injected through `RY(x_t)` before every Trotter layer.
The statevector path extracts `<Z_i>`, nearest-neighbor `<Z_i Z_{i+1}>`, and
half-chain entanglement energies. The Aer/IBM path drops entanglement features
because they require tomography and are not viable for short hardware runs.

## Runtime Paths

The defaults are local-development friendly:

| Variable | Default |
| --- | --- |
| `QE_ARTIFACTS_DIR` | `<repo>/artifacts` |
| `QE_DATA_DIR` | `<repo>/artifacts/data` |
| `QE_RESULTS_DIR` | `<repo>/artifacts/results` |

Docker Compose sets `QE_ARTIFACTS_DIR=/app/artifacts` and mounts named volumes
for data and result persistence.
