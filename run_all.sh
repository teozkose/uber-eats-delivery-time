#!/usr/bin/env bash
# Reproduces every number and figure from scratch; the same steps as `make all`.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
python -m eta.pipeline data
python -m eta.pipeline features
python -m eta.pipeline run
python -m eta.pipeline report
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_analysis.ipynb --ExecutePreprocessor.timeout=600
echo "done: see README.md, docs/index.html and results/"
