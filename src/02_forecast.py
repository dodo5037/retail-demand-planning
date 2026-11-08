"""Step 2: forecast weekly demand for the top products and compare methods.

Which products: the 20 highest-revenue products that sold in at least 40 of the 52 weeks
(very intermittent items need different methods, e.g. Croston, and are out of scope here).

Methods (all one-week-ahead):
  - Naive: next week = this week
  - 4-week moving average
  - Simple exponential smoothing (SES): weighted average, recent weeks count more; alpha fitted on training data
  - Holt (damped trend): SES plus a trend that fades out

Backtest: rolling origin over the last 12 weeks. For each test week, fit on all earlier weeks only,
forecast that week, then move forward one week. No future data leaks into a forecast.

Error metrics:
  - WAPE = sum(|actual - forecast|) / sum(actual). Used instead of MAPE because MAPE divides by each
    week's actual and blows up on zero-sales weeks.
  - Bias = sum(forecast - actual) / sum(actual). Positive = over-forecasting (excess stock),
    negative = under-forecasting (stock-outs).
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
T = ROOT / "outputs" / "tables"
T.mkdir(parents=True, exist_ok=True)

weekly = pd.read_csv(P / "weekly_demand.csv", parse_dates=["Week"], dtype={"StockCode": str})
names = pd.read_csv(P / "product_names.csv", dtype={"StockCode": str}).set_index("StockCode")["Description"]

stats = weekly.groupby("StockCode").agg(revenue=("revenue", "sum"), weeks_sold=("units", lambda s: (s > 0).sum()))
top = stats[stats["weeks_sold"] >= 40].sort_values("revenue", ascending=False).head(20).index.tolist()
TEST_WEEKS = 12

def forecast_next(history, method):
    h = np.asarray(history, dtype=float)
    if method == "Naive":
        return h[-1]
    if method == "4-week MA":
        return h[-4:].mean()
    if method == "SES":
        return SimpleExpSmoothing(h, initialization_method="estimated").fit().forecast(1)[0]
    if method == "Holt (damped)":
        return ExponentialSmoothing(h, trend="add", damped_trend=True, initialization_method="estimated").fit().forecast(1)[0]
    raise ValueError(method)

METHODS = ["Naive", "4-week MA", "SES", "Holt (damped)"]
rows = []
for code in top:
    y = weekly[weekly["StockCode"] == code].sort_values("Week")
    vals, weeks = y["units"].values, y["Week"].values
    for i in range(len(vals) - TEST_WEEKS, len(vals)):
        for m in METHODS:
            f = max(0.0, float(forecast_next(vals[:i], m)))
            rows.append({"StockCode": code, "Week": weeks[i], "method": m, "actual": vals[i], "forecast": f})

bt = pd.DataFrame(rows)
bt["abs_err"] = (bt["actual"] - bt["forecast"]).abs()
bt["err"] = bt["forecast"] - bt["actual"]
bt.to_csv(T / "backtest_detail.csv", index=False)

def summarise(g):
    return pd.Series({"WAPE": g["abs_err"].sum() / g["actual"].sum(), "Bias": g["err"].sum() / g["actual"].sum()})

overall = bt.groupby("method").apply(summarise).loc[METHODS]
per_product = bt.groupby(["StockCode", "method"]).apply(summarise).reset_index()
best = per_product.loc[per_product.groupby("StockCode")["WAPE"].idxmin()].set_index("StockCode")
best["Description"] = names.reindex(best.index)
best["revenue"] = stats.loc[best.index, "revenue"]

overall.to_csv(T / "method_comparison.csv")
best.sort_values("revenue", ascending=False).to_csv(T / "best_method_per_product.csv")

print(f"Products forecast: {len(top)} | test weeks: {TEST_WEEKS}\n")
print("Overall (all 20 products, last 12 weeks):")
print((overall * 100).round(1).astype(str).add("%").to_string())
print("\nWinning method count per product:")
print(best["method"].value_counts().to_string())
