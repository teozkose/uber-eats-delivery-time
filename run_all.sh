#!/usr/bin/env bash
# Reproduces every number and figure in the README from scratch.
set -euo pipefail
cd "$(dirname "$0")"
python data/download.py
python -m src.prepare
python -m src.model
python -m src.explain
python -m src.pca
python -m src.report
jupyter nbconvert --to notebook --execute --inplace notebooks/01_analysis.ipynb --ExecutePreprocessor.timeout=600
echo "done: see README.md and results/"
