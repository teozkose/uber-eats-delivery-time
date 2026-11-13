# Delivery-Time Prediction

[![CI](https://github.com/teozkose/uber-eats-delivery-time/actions/workflows/ci.yml/badge.svg)](https://github.com/teozkose/uber-eats-delivery-time/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## TL;DR

- **Question:** how long will a food delivery take, how sure can we be, and what ETA should the app promise?
- **Headline:** on 9,553 orders from the last ten days (never seen in training) gradient boosting is off by **3.07 min** on average (95% CI 3.02–3.11), 56% better than a city-and-distance lookup table; only 0.9% of orders miss by 10+ minutes.
- **Decision:** promise the model's 90th-percentile time, not its point estimate: late-order refunds fall from $475 to $20 per 1,000 orders for 4.7 extra minutes on the promised ETA.

## Problem and decision

A delivery app shows every customer an ETA. Too short and late orders turn into refunds and complaints; too long and customers cancel. The operations team needs three things: an accurate estimate, a sense of how wrong it can be for a given order, and a rule for which number to show. The deciding metrics are mean absolute error in minutes, the share of orders the estimate misses by 10+ minutes, and the refund cost per 1,000 orders under each ETA policy.

## Data

- **Source:** public Uber Eats delivery-time dataset ([liamarguedas/uber-eats-delivery-time](https://github.com/liamarguedas/uber-eats-delivery-time), MIT licence), `cleaned_data.csv`, about 9 MB. Downloaded by `make data`, never committed.
- **Size and range:** 43,862 orders, 1,320 couriers in 22 cities, 2022-02-11 to 2022-04-06 (44 order days). Mean delivery time 26.3 min (sd 9.4).
- **Fields used:** courier age and rating, restaurant and drop-off coordinates, order and pickup times, weather, traffic density, vehicle type and condition, stacked deliveries, order type, festival flag, city type. The city itself is decoded from the courier id (`INDORES13DEL02` is Indore).
- **Validation** (`src/eta/validate.py`): required columns, no nulls, unique order ids, ranges for age, rating, vehicle condition, stacked deliveries and target, valid coordinates, known category values, parseable dates. Any failure stops the run with the full list.
- **Cleaning** (counts from the last run):

| Step | Rows in | Dropped | Rows out |
|---|---|---|---|
| drop zero restaurant/drop-off coordinates | 43,862 | 3,509 | 40,353 |
| drop trips over 50 km | 40,353 | 0 | 40,353 |
| drop rows with unparseable order or pickup times | 40,353 | 0 | 40,353 |
| drop pickup delays over 60 min | 40,353 | 0 | 40,353 |

Negative coordinates are flipped to positive first (the source stores some Indian coordinates with the wrong sign). Pickups after midnight wrap around the day instead of showing a negative delay.

## Approach

```mermaid
flowchart LR
    A[download CSV] --> B[validate schema, ranges, nulls, duplicates]
    B --> C[features: haversine distance, pickup delay, hour, weekend, peak, distance band]
    C --> D{time split}
    D -->|orders to 2022-03-27| E[train: lookup median, ridge, XGBoost + random search grouped by city, quantile GBM]
    D -->|orders from 2022-03-28| F[holdout]
    E --> G[evaluate: MAE/R² with bootstrap CIs, 80% interval coverage, segment errors]
    F --> G
    G --> H[permutation importance with CIs + SHAP]
    G --> I[ETA policy cost: refunds per 1,000 orders]
    H --> J[README + report page]
    I --> J
```

- **Validation that fits the data:** the headline uses a time-ordered holdout (train on 30,800 orders up to 2022-03-27, test on 9,553 orders from 2022-03-28 to 2022-04-06). Hyperparameters are tuned with 3-fold cross-validation grouped by city on the training window only, so tuning never sees the test weeks and never rewards memorising a city.
- **Models:** a naive lookup (median minutes by city type and distance band), ridge regression on scaled and one-hot features, XGBoost with default settings, and XGBoost after a 12-draw random search over depth, learning rate, child weight, subsampling and L2 penalty.
- **Uncertainty:** 95% bootstrap intervals (1,000 resamples of holdout orders) on MAE, R² and the off-by-10 share. Per-order 80% prediction intervals from two quantile gradient-boosting models (10th and 90th percentile, pinball loss).
- **Sensitivity check:** the refund cost depends on how much lateness customers forgive, so the policy comparison is repeated for grace windows of 0, 5 and 10 minutes.

## Results

Time-ordered holdout, 9,553 orders:

| Model | MAE (min) | 95% CI | RMSE | R² | Off by 10+ min (95% CI) |
|---|---|---|---|---|---|
| Lookup median (city type × distance band) | 7.04 | 6.94–7.13 | 8.71 | 0.145 | 27.8% (26.9%–28.7%) |
| Ridge regression | 4.74 | 4.67–4.81 | 5.95 | 0.601 | 8.8% (8.2%–9.3%) |
| XGBoost, default settings | 3.10 | 3.05–3.14 | 3.85 | 0.833 | 1.0% (0.8%–1.2%) |
| XGBoost, tuned (grouped CV) | 3.07 | 3.02–3.11 | 3.81 | 0.837 | 0.9% (0.8%–1.1%) |

The tuned model reached 3.06 min in city-grouped cross-validation; on the holdout it beats default XGBoost (3.10) by a few seconds, so tuning matters far less than the move from a linear model (4.74) to trees. The relationships are strongly non-linear.

![Predicted vs actual and prediction intervals](results/pred_vs_actual.png)

**Prediction intervals.** The 80% interval averages 9.5 minutes wide and covers 77.0% of holdout orders against a 80% target, between 75.5% and 77.8% across traffic levels: slightly too narrow, evenly so.

**Does the original random split overstate accuracy?** The first version of this project scored on a random 80/20 split. Re-scored three ways:

| Validation | MAE (min) | R² | Folds |
|---|---|---|---|
| Random 5-fold (original write-up) | 3.06 ± 0.03 | 0.836 | 5 |
| 5-fold grouped by city | 3.08 ± 0.04 | 0.832 | 5 |
| Time-ordered holdout | 3.10 | 0.833 | 1 |

The gap is small: the time-ordered holdout is 1.4% worse than random k-fold. Every courier works in a single city (53 per city on average) and the features describe conditions (traffic, weather, distance, courier profile) rather than places, so what the model learns carries across cities and across weeks. The random-split number was not inflated, but the time-ordered one is the honest figure to quote.

![Validation schemes](results/validation_schemes.png)

**What a wrong ETA costs.** An order is late when it arrives more than 5 minutes after the promise; each late order earns a $5 credit.

| ETA promise | Avg promise (min) | Late rate | Refunds / 1,000 orders | 95% CI |
|---|---|---|---|---|
| Lookup median | 25.9 | 27.0% | $1,349 | $1,304–$1,390 |
| Model point ETA | 26.3 | 9.5% | $475 | $445–$505 |
| Model 90th-percentile ETA | 31.0 | 0.4% | $20 | $14–$27 |

A point estimate is late about half the time by construction, so with no grace window it is no better than the lookup. Sensitivity to the grace window (late rate / refunds per 1,000 orders):

| ETA promise | grace 0 min | grace 5 min | grace 10 min |
|---|---|---|---|
| Lookup median | 46.4% / $2,319 | 27.0% / $1,349 | 12.9% / $646 |
| Model point ETA | 48.5% / $2,426 | 9.5% / $475 | 0.7% / $36 |
| Model 90th-percentile ETA | 11.4% / $570 | 0.4% / $20 | 0.0% / $2 |

The 90th-percentile promise stays cheap at every grace window; the point estimate only wins when customers forgive a few minutes. The cost of the longer promise (cancellations at checkout) is not in this data and is the number to measure before rolling it out.

![ETA policy cost](results/eta_policy_cost.png)

## What drives it / error analysis

Errors by segment on the holdout (bias = predicted minus actual):

| Segment | Level | Orders | MAE | 95% CI | Bias | Off by 10+ |
|---|---|---|---|---|---|---|
| traffic | Jam | 2,989 | 3.47 | 3.38–3.56 | -0.06 | 1.6% |
| traffic | High | 997 | 3.39 | 3.21–3.56 | -0.06 | 2.3% |
| traffic | Medium | 2,317 | 3.00 | 2.91–3.08 | +0.12 | 0.8% |
| traffic | Low | 3,250 | 2.65 | 2.58–2.71 | +0.06 | 0.0% |
| weather | Windy | 1,636 | 3.37 | 3.25–3.50 | -0.14 | 1.9% |
| weather | Sandstorms | 1,599 | 3.30 | 3.17–3.43 | +0.05 | 1.4% |
| weather | Stormy | 1,582 | 3.23 | 3.12–3.33 | -0.00 | 1.4% |
| weather | Sunny | 1,541 | 3.10 | 2.98–3.22 | +0.19 | 0.6% |
| weather | Fog | 1,642 | 2.73 | 2.63–2.84 | -0.04 | 0.2% |
| weather | Cloudy | 1,553 | 2.68 | 2.58–2.78 | +0.10 | 0.1% |
| distance band | 12+ km | 3,425 | 3.26 | 3.18–3.33 | +0.09 | 0.7% |
| distance band | 9-12 km | 1,738 | 3.08 | 2.97–3.18 | -0.10 | 0.9% |
| distance band | 6-9 km | 1,741 | 3.02 | 2.91–3.12 | +0.07 | 1.4% |
| distance band | 3-6 km | 1,702 | 2.88 | 2.77–2.99 | -0.09 | 1.3% |
| distance band | 0-3 km | 947 | 2.79 | 2.67–2.92 | +0.10 | 0.5% |

Jammed and high traffic, windy and sandstorm weather, and the longest trips are where the model is least accurate; bias stays within a quarter of a minute everywhere, so no segment is systematically under-promised.

![Error by segment](results/error_by_segment.png)

**Feature importance.** Permutation importance (rise in MAE when one feature is shuffled, 10 repeats, 95% intervals) ranks weather, traffic, distance_km, courier_age, courier_rating; shuffling weather alone adds 2.10 minutes (CI 2.08–2.12). SHAP ranks courier_age, traffic, distance_km, courier_rating, vehicle_condition; 4 of the top five agree. The weakest features are order_type, is_weekend, pickup_delay_min.

![Feature importance](results/feature_importance.png)

## Limitations and next steps

- **Pickup-time ETA.** Pickup delay is known only after the courier collects the order, so this is the ETA shown at pickup. A checkout-time model would drop that feature and lose some accuracy.
- **Interval calibration.** Coverage is 77.0% against 80%; a conformal adjustment fitted on one calibration week would fix it.
- **Same couriers.** 100% of holdout orders come from couriers seen in training, so accuracy for newly hired couriers is untested; a courier-grouped split is the next check.
- **Short window.** Eight weeks of data, one season. Weather and festival effects need a longer history.
- **Assumed costs.** The $5 credit and 5-minute grace window are assumptions; refund cost scales linearly with the credit, and the cancellation cost of a longer promise is not measured.
- **Library note:** `shap` is used beyond the standard stack because it gives exact per-feature contributions for tree models, a second view next to permutation importance.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
make all        # data -> features -> run -> report -> notebook (about 2 minutes)
make test       # 8+ tests on a synthetic sample, no download
```

Every number above is read from `results/metrics.json`, which `make run` regenerates; `make report` re-renders this README and `docs/index.html`.

## Project structure

```
├── config.yaml                 paths, cleaning thresholds, model and cost parameters
├── Makefile / run_all.sh       make data | features | run | report | test | all
├── data/download.py            fetch the CSV (data files are not committed)
├── src/eta/
│   ├── data.py                 download and load raw records
│   ├── validate.py             schema, range, null and duplicate checks
│   ├── features.py             cleaning with row counts, distance, delays, time features, time split
│   ├── models.py               lookup baseline, ridge, XGBoost + random search, quantile GBM
│   ├── evaluate.py             metrics, bootstrap CIs, CV schemes, coverage, segments, ETA cost, importance
│   ├── figures.py              charts in results/
│   ├── report.py               README.md and docs/index.html from metrics.json
│   └── pipeline.py             command-line entry point
├── tests/                      pytest suite on synthetic data
├── notebooks/01_analysis.ipynb the story, importing from src/eta
├── results/                    metrics.json, segment_errors.csv, figures
└── docs/index.html             one-page report
```
