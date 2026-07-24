#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p executed_notebooks results figures dashboard_figures

python -m jupyter nbconvert --to notebook --execute   notebooks/01_QuantumEdge_Reproducibility.ipynb   --output-dir executed_notebooks   --ExecutePreprocessor.timeout=-1

python -m jupyter nbconvert --to notebook --execute   notebooks/04_QuantumEdge_MNIST_QRC_Benchmark.ipynb   --output-dir executed_notebooks   --ExecutePreprocessor.timeout=-1

python -m jupyter nbconvert --to notebook --execute   notebooks/02_QuantumEdge_Graphical_Results.ipynb   --output-dir executed_notebooks   --ExecutePreprocessor.timeout=-1

python -m jupyter nbconvert --to notebook --execute   notebooks/03_QuantumEdge_Package_Verification.ipynb   --output-dir executed_notebooks   --ExecutePreprocessor.timeout=-1
