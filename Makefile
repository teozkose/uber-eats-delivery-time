PY ?= python
export PYTHONPATH := src

.PHONY: data features run report notebook test lint all clean

data:
	$(PY) -m eta.pipeline data

features: data
	$(PY) -m eta.pipeline features

run: features
	$(PY) -m eta.pipeline run

report:
	$(PY) -m eta.pipeline report

notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_analysis.ipynb --ExecutePreprocessor.timeout=600

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

all: run report notebook
