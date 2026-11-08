"""Step 3: classify products (ABC-XYZ) and set safety stock + reorder points for the forecast products.

ABC (by revenue, all products): A = top 80% of revenue, B = next 15%, C = last 5%.
XYZ (by demand variability, coefficient of variation of weekly units):
    X = CV < 0.5 (steady), Y = 0.5-1.0 (variable), Z = > 1.0 (erratic).

Safety stock for the 20 forecast products:
    SS  = z * sigma_error * sqrt(L)
    ROP = forecast_per_week * L + SS
  sigma_error = standard deviation of the best method's one-week-ahead errors in the backtest
  L = lead time in weeks. ASSUMPTION: 2 weeks (the dataset has no supplier lead times).
  z from the target service level: A = 98%, B = 95%, C = 90%.
Note: errors were measured on only 12 weeks, so sigma is a rough estimate.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statistics import NormalDist
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
P, T = ROOT / "data" / "processed", ROOT / "outputs" / "tables"
LEAD_TIME_WEEKS = 2
SERVICE = {"A": 0.98, "B": 0.95, "C": 0.90}

weekly = pd.read_csv(P / "weekly_demand.csv", parse_dates=["Week"], dtype={"StockCode": str})
names = pd.read_csv(P / "product_names.csv", dtype={"StockCode": str}).set_index("StockCode")["Description"]
bt = pd.read_csv(T / "backtest_detail.csv", dtype={"StockCode": str})
best = pd.read_csv(T / "best_method_per_product.csv", dtype={"StockCode": str}).set_index("StockCode")

# ---- ABC-XYZ on all products
g = weekly.groupby("StockCode")
cls = pd.DataFrame({"revenue": g["revenue"].sum(), "mean_units": g["units"].mean(), "std_units": g["units"].std()})
cls = cls[cls["revenue"] > 0].sort_values("revenue", ascending=False)
cls["cum_share"] = cls["revenue"].cumsum() / cls["revenue"].sum()
cls["ABC"] = np.where(cls["cum_share"] <= 0.80, "A", np.where(cls["cum_share"] <= 0.95, "B", "C"))
cls["CV"] = cls["std_units"] / cls["mean_units"]
cls["XYZ"] = np.where(cls["CV"] < 0.5, "X", np.where(cls["CV"] <= 1.0, "Y", "Z"))
cls["Description"] = names.reindex(cls.index)
cls.to_csv(T / "abc_xyz_all_products.csv")

matrix = pd.crosstab(cls["ABC"], cls["XYZ"])
rev_share = cls.groupby("ABC")["revenue"].sum() / cls["revenue"].sum()
matrix.to_csv(T / "abc_xyz_matrix.csv")

# ---- Safety stock and reorder point for the 20 forecast products
def next_week_forecast(h, method):
    h = np.asarray(h, dtype=float)
    if method == "Naive":
        return h[-1]
    if method == "4-week MA":
        return h[-4:].mean()
    if method == "SES":
        return SimpleExpSmoothing(h, initialization_method="estimated").fit().forecast(1)[0]
    return ExponentialSmoothing(h, trend="add", damped_trend=True, initialization_method="estimated").fit().forecast(1)[0]

rows = []
for code, b in best.iterrows():
    hist = weekly[weekly["StockCode"] == code].sort_values("Week")["units"].values
    errs = bt[(bt["StockCode"] == code) & (bt["method"] == b["method"])]["err"]
    sigma = errs.std(ddof=1)
    abc = cls.loc[code, "ABC"]
    z = NormalDist().inv_cdf(SERVICE[abc])
    fc = max(0.0, next_week_forecast(hist, b["method"]))
    ss = z * sigma * np.sqrt(LEAD_TIME_WEEKS)
    rows.append({
        "StockCode": code, "Description": names.get(code, ""), "ABC": abc, "XYZ": cls.loc[code, "XYZ"],
        "method": b["method"], "WAPE": round(b["WAPE"], 3), "forecast_per_week": round(fc),
        "error_std": round(sigma), "service_level": SERVICE[abc], "z": round(z, 2),
        "safety_stock": round(ss), "reorder_point": round(fc * LEAD_TIME_WEEKS + ss),
    })
pol = pd.DataFrame(rows).sort_values("reorder_point", ascending=False)
pol.to_csv(T / "inventory_policy_top20.csv", index=False)

print("ABC share of products and revenue:")
for k in "ABC":
    print(f"  {k}: {(cls['ABC'] == k).sum():,} products ({(cls['ABC'] == k).mean():.0%}) -> {rev_share[k]:.0%} of revenue")
print("\nABC-XYZ matrix (product counts):")
print(matrix.to_string())
print(f"\nInventory policy (lead time {LEAD_TIME_WEEKS} weeks, assumed):")
print(pol[["Description", "ABC", "XYZ", "method", "forecast_per_week", "safety_stock", "reorder_point"]].head(10).to_string(index=False))
