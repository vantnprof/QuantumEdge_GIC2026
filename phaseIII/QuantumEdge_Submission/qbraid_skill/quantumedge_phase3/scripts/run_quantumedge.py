#!/usr/bin/env python3
"""Execute the QuantumEdge Phase 3 workflow from the repository root."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOKS = [
    ROOT / "notebooks" / "01_QuantumEdge_Reproducibility.ipynb",
    ROOT / "notebooks" / "04_QuantumEdge_MNIST_QRC_Benchmark.ipynb",
    ROOT / "notebooks" / "02_QuantumEdge_Graphical_Results.ipynb",
    ROOT / "notebooks" / "03_QuantumEdge_Package_Verification.ipynb",
]
OUTPUT_DIR = ROOT / "executed_notebooks"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [str(path.relative_to(ROOT)) for path in NOTEBOOKS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing notebooks: {missing}")

    for notebook in NOTEBOOKS:
        print(f"\nExecuting {notebook.relative_to(ROOT)}")
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(notebook),
            "--output-dir",
            str(OUTPUT_DIR),
            "--ExecutePreprocessor.timeout=-1",
        ]
        subprocess.run(command, cwd=ROOT, check=True)

    summary_path = ROOT / "results" / "package_verification_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Strict verification summary was not created.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("fail", 1)) != 0:
        raise RuntimeError(f"Package verification failed: {summary}")

    print("\nQuantumEdge Phase 3 reproduction completed with zero verification failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
