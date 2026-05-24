"""Run the classical benchmark using the shared project configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantumedge.config import ensure_runtime_dirs
from quantumedge.experiments.benchmark_config import CLASSICAL_BENCHMARK
from quantumedge.pipelines.classical_baselines import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QuantumEdge classical benchmark.")
    parser.add_argument(
        "--write-plots",
        action=argparse.BooleanOptionalAction,
        default=CLASSICAL_BENCHMARK.write_plots,
        help="Write all classical diagnostic plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ensure_runtime_dirs()
    metrics = run_pipeline(write_plots=args.write_plots)
    path = settings.results_dir / CLASSICAL_BENCHMARK.metrics_filename
    print(f"Classical benchmark metrics: {path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
