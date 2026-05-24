"""FastAPI service for inspecting pipeline artifacts and triggering batch runs."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

from quantumedge.config import ensure_runtime_dirs
from quantumedge.pipelines.classical_baselines import run_pipeline

app = FastAPI(
    title="QuantumEdge GIC 2026 API",
    version="0.1.0",
    description="Artifact API for the QuantumEdge volatility forecasting pipeline.",
)

_RUN_LOCK = threading.Lock()


def _metrics_path() -> Path:
    return ensure_runtime_dirs().results_dir / "metrics.csv"


def _quantum_metrics_path() -> Path:
    return ensure_runtime_dirs().results_dir / "quantum_qrc_metrics.csv"


def _read_metrics() -> pd.DataFrame:
    path = _metrics_path()
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.csv not found. Run the pipeline service first.",
        )
    return pd.read_csv(path)


def _read_quantum_metrics() -> pd.DataFrame:
    path = _quantum_metrics_path()
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="quantum_qrc_metrics.csv not found. Run the QRC pipeline first.",
        )
    return pd.read_csv(path)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.replace({np.nan: None})
    return clean.to_dict(orient="records")


def _run_pipeline_once(skip_plots: bool) -> None:
    if not _RUN_LOCK.acquire(blocking=False):
        print("[api] pipeline run skipped because another run is active")
        return
    try:
        run_pipeline(write_plots=not skip_plots)
    finally:
        _RUN_LOCK.release()


@app.get("/health")
def health() -> dict[str, Any]:
    settings = ensure_runtime_dirs()
    metrics_exists = (settings.results_dir / "metrics.csv").exists()
    return {
        "status": "ok",
        "data_dir": str(settings.data_dir),
        "results_dir": str(settings.results_dir),
        "metrics_ready": metrics_exists,
        "quantum_metrics_ready": (settings.results_dir / "quantum_qrc_metrics.csv").exists(),
    }


@app.get("/architecture")
def architecture() -> dict[str, Any]:
    return {
        "flow": [
            "data-service caches market and synthetic datasets",
            "feature builders create temporal splits and model-ready matrices",
            "pipeline service trains/evaluates classical baseline models",
            "quantum pipeline runs the proposed dual-timescale QRC method",
            "visualization module writes report plots",
            "api service exposes metrics and plot artifacts",
        ],
        "services": ["data", "pipeline", "api", "quantum-sim", "quantum-aer", "quantum-hardware"],
    }


@app.get("/metrics")
def metrics() -> list[dict[str, Any]]:
    return _records(_read_metrics())


@app.get("/metrics/{dataset}")
def metrics_for_dataset(dataset: str) -> list[dict[str, Any]]:
    df = _read_metrics()
    subset = df[df["dataset"] == dataset]
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"dataset not found: {dataset}")
    return _records(subset)


@app.get("/quantum/metrics")
def quantum_metrics() -> list[dict[str, Any]]:
    return _records(_read_quantum_metrics())


@app.get("/quantum/metrics/{dataset}")
def quantum_metrics_for_dataset(dataset: str) -> list[dict[str, Any]]:
    df = _read_quantum_metrics()
    subset = df[df["dataset"] == dataset]
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"quantum dataset not found: {dataset}")
    return _records(subset)


@app.get("/plots")
def plots() -> dict[str, Any]:
    plots_dir = ensure_runtime_dirs().plots_dir
    files = sorted(path.name for path in plots_dir.glob("*.png"))
    return {"count": len(files), "plots": files}


@app.get("/plots/{filename}")
def plot_file(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="invalid plot filename")
    path = ensure_runtime_dirs().plots_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"plot not found: {filename}")
    return FileResponse(path, media_type="image/png")


@app.post("/runs/classical-baseline", status_code=202)
def start_classical_baseline_run(
    background_tasks: BackgroundTasks,
    skip_plots: bool = False,
) -> dict[str, str]:
    if _RUN_LOCK.locked():
        raise HTTPException(status_code=409, detail="pipeline run already active")
    background_tasks.add_task(_run_pipeline_once, skip_plots)
    return {"status": "accepted"}
