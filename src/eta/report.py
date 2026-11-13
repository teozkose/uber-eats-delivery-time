"""Render README.md (from README.template.md) and docs/index.html from results/metrics.json,
so every number in the write-up comes from the last run."""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any

from eta.config import ROOT, resolve

MODEL_LABELS = {"median_baseline": "Lookup median (city type × distance band)",
                "ridge": "Ridge regression", "xgboost_default": "XGBoost, default settings",
                "xgboost_tuned": "XGBoost, tuned (grouped CV)"}
SCHEME_TEXT = {"random_kfold": "Random 5-fold (original write-up)",
               "city_grouped_kfold": "5-fold grouped by city", "time_holdout": "Time-ordered holdout"}
POLICY_TEXT = {"baseline_median": "Lookup median", "model_point": "Model point ETA",
               "model_p90": "Model 90th-percentile ETA"}


def pct(x: float, digits: int = 1) -> str:
    """0.1234 -> '12.3%'."""
    return f"{100 * x:.{digits}f}%"


def ci(pair: list[float], digits: int = 2) -> str:
    """[1.0, 2.0] -> '1.00–2.00'."""
    return f"{pair[0]:.{digits}f}–{pair[1]:.{digits}f}"


def money(x: float) -> str:
    return f"${x:,.0f}"


def model_rows(m: dict[str, Any]) -> list[list[str]]:
    rows = []
    for key, label in MODEL_LABELS.items():
        r = m["models"][key]
        rows.append([label, f"{r['mae']:.2f}", ci(r["mae_ci"]), f"{r['rmse']:.2f}", f"{r['r2']:.3f}",
                     f"{pct(r['off_by_10'])} ({pct(r['off_by_10_ci'][0])}–{pct(r['off_by_10_ci'][1])})"])
    return rows


def scheme_rows(m: dict[str, Any]) -> list[list[str]]:
    rows = []
    for key, label in SCHEME_TEXT.items():
        r = m["validation_schemes"][key]
        spread = f" ± {r['mae_std']:.2f}" if r["folds"] > 1 else ""
        rows.append([label, f"{r['mae_mean']:.2f}{spread}", f"{r['r2_mean']:.3f}", str(r["folds"])])
    return rows


def segment_rows(m: dict[str, Any]) -> list[list[str]]:
    rows = []
    for r in m["segments"]:
        rows.append([r["segment"].replace("_", " "), r["level"], f"{r['n']:,}", f"{r['mae']:.2f}",
                     f"{r['mae_lo']:.2f}–{r['mae_hi']:.2f}", f"{r['bias']:+.2f}", pct(r["off_by_10"])])
    return rows


def policy_rows(m: dict[str, Any]) -> list[list[str]]:
    rows = []
    for key, label in POLICY_TEXT.items():
        r = m["business"]["policies"][key]
        rows.append([label, f"{r['mean_promise_min']:.1f}", pct(r["late_rate"]),
                     money(r["cost_per_1000"]), f"{money(r['cost_ci'][0])}–{money(r['cost_ci'][1])}"])
    return rows


def sensitivity_rows(m: dict[str, Any]) -> list[list[str]]:
    by: dict[str, dict[int, str]] = {}
    for r in m["business"]["sensitivity"]:
        by.setdefault(r["policy"], {})[r["grace_min"]] = f"{pct(r['late_rate'])} / {money(r['cost_per_1000'])}"
    graces = sorted({r["grace_min"] for r in m["business"]["sensitivity"]})
    return [[POLICY_TEXT[p]] + [by[p][g] for g in graces] for p in POLICY_TEXT]


def cleaning_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[s["step"], f"{s['rows_before']:,}", f"{s['dropped']:,}", f"{s['rows_after']:,}"]
            for s in m["data"]["cleaning"]]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Rows of strings -> a GitHub markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def html_table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


TABLES = {
    "MODEL_TABLE": (["Model", "MAE (min)", "95% CI", "RMSE", "R²", "Off by 10+ min (95% CI)"], model_rows),
    "SCHEME_TABLE": (["Validation", "MAE (min)", "R²", "Folds"], scheme_rows),
    "SEGMENT_TABLE": (["Segment", "Level", "Orders", "MAE", "95% CI", "Bias", "Off by 10+"], segment_rows),
    "POLICY_TABLE": (["ETA promise", "Avg promise (min)", "Late rate", "Refunds / 1,000 orders", "95% CI"],
                     policy_rows),
    "SENSITIVITY_TABLE": (["ETA promise", "grace 0 min", "grace 5 min", "grace 10 min"], sensitivity_rows),
    "CLEANING_TABLE": (["Step", "Rows in", "Dropped", "Rows out"], cleaning_rows),
}


def values(m: dict[str, Any]) -> dict[str, str]:
    """Scalar placeholders used by README.template.md and the HTML page."""
    h, s, iv, biz = m["headline"], m["split"], m["intervals"], m["business"]
    pol = biz["policies"]
    sch = m["validation_schemes"]
    perm = m["importance"]["permutation"]
    shap = m["importance"]["shap"]
    return {
        "RAW_ROWS": f"{m['data']['rows']:,}",
        "ROWS": f"{m['data']['modelling_rows']:,}",
        "DATE_MIN": m["data"]["date_min"], "DATE_MAX": m["data"]["date_max"],
        "ORDER_DAYS": str(m["data"]["order_days"]),
        "COURIERS": f"{m['data']['couriers']:,}", "CITIES": str(s["cities"]),
        "TRAIN_ROWS": f"{s['train_rows']:,}", "TEST_ROWS": f"{s['test_rows']:,}",
        "TRAIN_END": s["train_dates"][1], "TEST_START": s["test_dates"][0], "TEST_END": s["test_dates"][1],
        "TARGET_MEAN": f"{s['target_mean_min']:.1f}", "TARGET_STD": f"{s['target_std_min']:.1f}",
        "MAE": f"{h['mae']:.2f}", "MAE_CI": ci(h["mae_ci"]), "R2": f"{h['r2']:.2f}",
        "OFF10": pct(h["off_by_10"]), "OFF10_CI": f"{pct(h['off_by_10_ci'][0])}–{pct(h['off_by_10_ci'][1])}",
        "BASE_MAE": f"{h['baseline_mae']:.2f}", "MAE_CUT": pct(h["mae_reduction_vs_baseline"], 0),
        "RIDGE_MAE": f"{m['models']['ridge']['mae']:.2f}",
        "RANDOM_MAE": f"{sch['random_kfold']['mae_mean']:.2f}",
        "GROUP_MAE": f"{sch['city_grouped_kfold']['mae_mean']:.2f}",
        "TIME_MAE": f"{sch['time_holdout']['mae_mean']:.2f}",
        "SCHEME_GAP": pct(sch["time_holdout"]["mae_mean"] / sch["random_kfold"]["mae_mean"] - 1),
        "COURIER_OVERLAP": pct(s["courier_overlap_time_holdout"], 0),
        "COURIERS_PER_CITY": f"{s['couriers_per_city']:.0f}",
        "TUNED_CV_MAE": f"{m['tuning']['grouped_cv_mae']:.2f}", "N_ITER": str(m["tuning"]["n_iter"]),
        "DEFAULT_MAE": f"{m['models']['xgboost_default']['mae']:.2f}",
        "COVERAGE": pct(iv["coverage"]), "NOMINAL": pct(iv["nominal"], 0),
        "WIDTH": f"{iv['mean_width_min']:.1f}",
        "COV_MIN": pct(min(iv["by_traffic"].values())), "COV_MAX": pct(max(iv["by_traffic"].values())),
        "REFUND": money(biz["refund"]), "GRACE": str(biz["grace_min"]),
        "COST_BASE": money(pol["baseline_median"]["cost_per_1000"]),
        "COST_POINT": money(pol["model_point"]["cost_per_1000"]),
        "COST_P90": money(pol["model_p90"]["cost_per_1000"]),
        "LATE_BASE": pct(pol["baseline_median"]["late_rate"]),
        "LATE_POINT": pct(pol["model_point"]["late_rate"]),
        "LATE_P90": pct(pol["model_p90"]["late_rate"]),
        "P90_EXTRA_MIN": f"{pol['model_p90']['mean_promise_min'] - pol['model_point']['mean_promise_min']:.1f}",
        "PERM_TOP": ", ".join(p["feature"] for p in perm[:5]),
        "SHAP_TOP": ", ".join(p["feature"] for p in shap[:5]),
        "OVERLAP": str(m["importance"]["top5_overlap"]),
        "PERM_1": perm[0]["feature"], "PERM_1_VAL": f"{perm[0]['mae_increase']:.2f}",
        "PERM_1_CI": ci(perm[0]["ci"]),
        "PERM_LAST": ", ".join(p["feature"] for p in perm[-3:]),
    }


def fill(template: str, m: dict[str, Any], table_fn: Any) -> str:
    """Replace {{KEY}} placeholders; tables are rendered with `table_fn(header, rows)`."""
    vals = values(m)
    for key, (header, rows_fn) in TABLES.items():
        vals[key] = table_fn(header, rows_fn(m))
    out = re.sub(r"\{\{(\w+)\}\}", lambda mt: vals[mt.group(1)], template)
    return out


def _img(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}"></figure>'


def render_html(m: dict[str, Any], results: Path) -> str:
    """One self-contained page: TL;DR, figures, tables, method, limitations."""
    v = values(m)
    t = {k: html_table(h, fn(m)) for k, (h, fn) in TABLES.items()}
    figs = {name: _img(results / f"{name}.png", alt) for name, alt in [
        ("pred_vs_actual", "Predicted against actual minutes and 80% intervals"),
        ("validation_schemes", "MAE under three validation schemes"),
        ("error_by_segment", "MAE by traffic, weather and distance band"),
        ("feature_importance", "Permutation importance and SHAP"),
        ("eta_policy_cost", "Late rate against average promised ETA")]}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Delivery-Time Prediction</title>
<style>
:root {{ --fg:#1a202c; --muted:#4a5568; --bg:#ffffff; --line:#e2e8f0; --accent:#2b6cb0; --card:#f7fafc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e2e8f0; --muted:#a0aec0; --bg:#171923;
  --line:#2d3748; --accent:#63b3ed; --card:#1a202c; }} }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.6 system-ui,-apple-system,sans-serif;
  max-width:920px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ margin-bottom:4px; }} h2 {{ margin-top:40px; border-bottom:1px solid var(--line); padding-bottom:4px; }}
.sub {{ color:var(--muted); margin-top:0; }}
.tldr {{ background:var(--card); border-left:4px solid var(--accent); padding:12px 16px; border-radius:4px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:12px; }}
.kpi b {{ display:block; font-size:24px; color:var(--accent); }}
.kpi span {{ color:var(--muted); font-size:14px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:12px 0; display:block; overflow-x:auto; }}
th,td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; white-space:nowrap; }}
figure {{ margin:16px 0; }} img {{ max-width:100%; height:auto; background:#fff; border-radius:4px; }}
</style></head><body>
<h1>Delivery-Time Prediction</h1>
<p class="sub">How long will this food delivery take, how sure are we, and what does a wrong ETA cost?</p>
<div class="tldr"><b>TL;DR.</b> On {v['TEST_ROWS']} orders from the last ten days (never seen in training),
gradient boosting predicts delivery time within <b>{v['MAE']} min</b> on average (95% CI {v['MAE_CI']}),
{v['MAE_CUT']} better than a city-and-distance lookup table. Only {v['OFF10']} of orders are off by 10+ minutes.
Promising the model's 90th-percentile time instead of its point estimate cuts late-order refunds from
{v['COST_POINT']} to {v['COST_P90']} per 1,000 orders for {v['P90_EXTRA_MIN']} extra minutes on the promise.</div>
<div class="kpis">
<div class="kpi"><b>{v['MAE']} min</b><span>holdout MAE (lookup: {v['BASE_MAE']})</span></div>
<div class="kpi"><b>{v['R2']}</b><span>holdout R²</span></div>
<div class="kpi"><b>{v['OFF10']}</b><span>orders off by 10+ min</span></div>
<div class="kpi"><b>{v['COVERAGE']}</b><span>coverage of the {v['NOMINAL']} interval</span></div>
</div>
<h2>Results</h2>
{t['MODEL_TABLE']}
{figs['pred_vs_actual']}
<h2>Does the original random split overstate accuracy?</h2>
<p>Barely. Random k-fold gives {v['RANDOM_MAE']} min, folds grouped by city {v['GROUP_MAE']}, and the
time-ordered holdout {v['TIME_MAE']} ({v['SCHEME_GAP']} worse than random). Every courier works in one city
({v['COURIERS_PER_CITY']} per city on average), and the features describe conditions rather than places,
so the model carries across cities and across weeks.</p>
{t['SCHEME_TABLE']}
{figs['validation_schemes']}
<h2>Where it is wrong</h2>
{t['SEGMENT_TABLE']}
{figs['error_by_segment']}
<h2>What drives it</h2>
<p>Permutation importance top five: {v['PERM_TOP']}. SHAP top five: {v['SHAP_TOP']}
({v['OVERLAP']} of 5 shared).</p>
{figs['feature_importance']}
<h2>What a wrong ETA costs</h2>
<p>An order counts as late when it arrives more than {v['GRACE']} minutes after the promise, and each late
order earns a {v['REFUND']} credit.</p>
{t['POLICY_TABLE']}
<p>Sensitivity to the grace window (late rate / refunds per 1,000 orders):</p>
{t['SENSITIVITY_TABLE']}
{figs['eta_policy_cost']}
<h2>Data</h2>
<p>{v['RAW_ROWS']} delivery records from {v['CITIES']} cities, {v['DATE_MIN']} to {v['DATE_MAX']}
({v['ORDER_DAYS']} order days, {v['COURIERS']} couriers). Public Uber Eats delivery-time dataset, MIT licence.</p>
{t['CLEANING_TABLE']}
<h2>Method</h2>
<p>Train on orders up to {v['TRAIN_END']}, test on {v['TEST_START']} to {v['TEST_END']}. Models: a lookup
median by city type and distance band, ridge regression, and XGBoost with a {v['N_ITER']}-draw random search
scored by city-grouped CV. 80% intervals come from two quantile gradient-boosting models (10th and 90th
percentiles). Uncertainty on every headline number is a 1,000-draw bootstrap over holdout orders.</p>
<h2>Limitations</h2>
<ul>
<li>Pickup delay is only known once the courier collects the order, so this is the ETA shown at pickup,
not at checkout.</li>
<li>The interval covers {v['COVERAGE']} of orders against a {v['NOMINAL']} target; a conformal
adjustment on a calibration week would close the gap.</li>
<li>All test couriers also appear in training ({v['COURIER_OVERLAP']}), so accuracy for new couriers is
untested.</li>
<li>The refund amount and grace window are assumptions; costs scale linearly with the refund.</li>
</ul>
</body></html>
"""


def render_all(cfg: dict[str, Any]) -> None:
    """Write README.md and docs/index.html."""
    results = resolve(cfg["paths"]["results"])
    m = json.loads((results / "metrics.json").read_text())
    template = (ROOT / "README.template.md").read_text()
    (ROOT / "README.md").write_text(fill(template, m, md_table))
    docs = resolve(cfg["paths"]["docs"])
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(render_html(m, results))
    print("rendered README.md and docs/index.html")
