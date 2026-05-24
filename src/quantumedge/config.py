"""Runtime configuration for local and containerized services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    artifacts_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    results_dir: Path
    plots_dir: Path


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = _resolve_path(
        os.getenv("QE_PROJECT_ROOT", Path(__file__).resolve().parents[2])
    )
    artifacts_dir = _resolve_path(
        os.getenv("QE_ARTIFACTS_DIR", project_root / "artifacts")
    )
    data_dir = _resolve_path(os.getenv("QE_DATA_DIR", artifacts_dir / "data"))
    results_dir = _resolve_path(
        os.getenv("QE_RESULTS_DIR", artifacts_dir / "results")
    )

    return Settings(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        processed_data_dir=data_dir / "processed",
        results_dir=results_dir,
        plots_dir=results_dir / "plots",
    )


def ensure_runtime_dirs(settings: Settings | None = None) -> Settings:
    settings = settings or get_settings()
    for path in (
        settings.artifacts_dir,
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.results_dir,
        settings.plots_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return settings
