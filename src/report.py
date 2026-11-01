"""Render README.md from README.template.md and results/metrics.json so every number in the
write-up comes from the last run."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = json.load(open(os.path.join(ROOT, "results", "metrics.json")))

models = m["models"]
LABELS = {"linear_regression": "Linear regression", "random_forest": "Random forest", "xgboost": "XGBoost (gradient boosting)"}
rows = "\n".join(
    f"| {LABELS.get(name, name)} | {v['r2']:.3f} | {v['mae']:.2f} | {v['rmse']:.2f} |"
    for name, v in models.items()
)
pca = m["pca"]
rows += (f"\n| XGBoost on {pca['n_components_95pct']} principal components | "
         f"{pca['xgboost_on_components']['r2']:.3f} | {pca['xgboost_on_components']['mae']:.2f} | "
         f"{pca['xgboost_on_components']['rmse']:.2f} |")

imp = m["importance"]
shap_list = "\n".join(f"{i+1}. `{x['feature']}` ({x['mean_abs_shap_min']:.2f} min)" for i, x in enumerate(imp["shap_top"][:6]))
perm_list = "\n".join(f"{i+1}. `{x['feature']}` (ΔR² {x['r2_drop']:.3f})" for i, x in enumerate(imp["permutation_top"][:6]))

fill = {
    "TRAIN_ROWS": f"{m['dataset']['train_rows']:,}",
    "TEST_ROWS": f"{m['dataset']['test_rows']:,}",
    "TOTAL_ROWS": f"{m['dataset']['train_rows'] + m['dataset']['test_rows']:,}",
    "N_FEATURES": str(m["dataset"]["n_features"]),
    "TARGET_MEAN": f"{m['dataset']['target_mean_min']:.1f}",
    "TARGET_STD": f"{m['dataset']['target_std_min']:.1f}",
    "RESULTS_TABLE": rows,
    "XGB_R2": f"{models['xgboost']['r2']:.3f}",
    "XGB_MAE": f"{models['xgboost']['mae']:.2f}",
    "LIN_R2": f"{models['linear_regression']['r2']:.3f}",
    "PCA_N": str(pca["n_components_95pct"]),
    "PCA_R2": f"{pca['xgboost_on_components']['r2']:.3f}",
    "SHAP_LIST": shap_list,
    "PERM_LIST": perm_list,
    "OVERLAP": str(imp["top5_overlap"]),
}
text = open(os.path.join(ROOT, "README.template.md")).read()
for k, v in fill.items():
    text = text.replace("{{" + k + "}}", v)
open(os.path.join(ROOT, "README.md"), "w").write(text)
print("README.md rendered")
