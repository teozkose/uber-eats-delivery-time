# Delivery-Time Prediction

Predicting how many minutes a food delivery will take, and finding out which factors move that number most.

**Data:** {{TOTAL_ROWS}} delivery records from a public Uber Eats delivery dataset (courier age and rating, restaurant and drop-off coordinates, order and pickup times, weather, traffic, vehicle, order type, city, festival flag). Average delivery takes {{TARGET_MEAN}} min with a standard deviation of {{TARGET_STD}} min.

**Question:** can route conditions and courier characteristics predict delivery time well enough to be useful, and which of them matter most?

## Results

Held-out test set ({{TEST_ROWS}} deliveries, 20% split, {{N_FEATURES}} features):

| Model | R² | MAE (min) | RMSE (min) |
|---|---|---|---|
{{RESULTS_TABLE}}

Gradient boosting explains {{XGB_R2}} of the variance and is off by {{XGB_MAE}} minutes on average, against {{LIN_R2}} for a linear model: the relationships are clearly non-linear. Compressing the features to {{PCA_N}} principal components (95% of variance) made the model worse ({{PCA_R2}}), so the full feature set is kept.

![Predicted vs actual](results/pred_vs_actual.png)

## What drives delivery time

Two independent methods rank the features. SHAP measures each feature's average contribution to a prediction; permutation importance measures how much R² drops when a feature is shuffled. {{OVERLAP}} of the top 5 features agree between the two.

SHAP (mean |contribution|):
{{SHAP_LIST}}

Permutation (drop in R²):
{{PERM_LIST}}

![Feature importance](results/feature_importance.png)

Takeaway: delivery time is driven by a mix of route conditions (distance, traffic) and courier factors (age, rating, vehicle condition). Weather, stacked deliveries and city add a little on top; order type and festival days barely register.

## How it works

```
data/download.py     fetch the raw CSV (not committed)
src/prepare.py       clean, compute haversine distance, pickup delay, time features; one-hot encode
src/model.py         80/20 split; linear regression, random forest, XGBoost; metrics + scatter plot
src/explain.py       SHAP and permutation importance; agreement check
src/pca.py           PCA to 95% variance, retrain XGBoost, compare
src/report.py        render this README from results/metrics.json
notebooks/01_analysis.ipynb   the same steps with charts, executed end to end
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

About two minutes on a laptop. Every number above is read from `results/metrics.json`, which the run regenerates.

Data: [uber-eats-delivery-time](https://github.com/liamarguedas/uber-eats-delivery-time) (MIT).
