"""Step 4: charts for the README."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P, T, C = ROOT / "data" / "processed", ROOT / "outputs" / "tables", ROOT / "outputs" / "charts"
C.mkdir(parents=True, exist_ok=True)
NAVY, TEAL, PINK, GREY = "#1f3864", "#10b981", "#f43f5e", "#9aa5b1"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

# 1. Pareto / ABC curve
cls = pd.read_csv(T / "abc_xyz_all_products.csv", dtype={"StockCode": str})
x = np.arange(1, len(cls) + 1) / len(cls)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x * 100, cls["cum_share"] * 100, color=NAVY, lw=2)
a_cut = (cls["ABC"] == "A").mean() * 100
ax.axvline(a_cut, color=TEAL, ls="--"); ax.axhline(80, color=TEAL, ls="--")
ax.text(a_cut + 1, 40, f"A items: {a_cut:.0f}% of products\n= 80% of revenue", color=NAVY)
ax.set_xlabel("% of products (ranked by revenue)"); ax.set_ylabel("% of total revenue")
ax.set_title("ABC analysis: a small share of products drives most revenue")
fig.tight_layout(); fig.savefig(C / "01_pareto_abc.png"); plt.close(fig)

# 2. Method comparison
mc = pd.read_csv(T / "method_comparison.csv", index_col=0)
fig, ax = plt.subplots(figsize=(8, 4.5))
cols = [GREY, GREY, TEAL, TEAL]
bars = ax.bar(mc.index, mc["WAPE"] * 100, color=cols)
for b, w, bias in zip(bars, mc["WAPE"], mc["Bias"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{w:.1%}\nbias {bias:+.0%}", ha="center", fontsize=9)
ax.set_ylabel("WAPE (lower is better)"); ax.set_ylim(0, 70)
ax.set_title("Forecast error, 20 top products, rolling 12-week backtest")
fig.tight_layout(); fig.savefig(C / "02_method_comparison.png"); plt.close(fig)

# 3. Forecast vs actual for one product
bt = pd.read_csv(T / "backtest_detail.csv", parse_dates=["Week"], dtype={"StockCode": str})
weekly = pd.read_csv(P / "weekly_demand.csv", parse_dates=["Week"], dtype={"StockCode": str})
names = pd.read_csv(P / "product_names.csv", dtype={"StockCode": str}).set_index("StockCode")["Description"]
code = "85099B"  # JUMBO BAG RED RETROSPOT
hist = weekly[weekly["StockCode"] == code].sort_values("Week")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(hist["Week"], hist["units"], color=NAVY, lw=1.8, label="Actual weekly units")
for m, col in [("Naive", GREY), ("Holt (damped)", TEAL)]:
    s = bt[(bt["StockCode"] == code) & (bt["method"] == m)]
    ax.plot(s["Week"], s["forecast"], color=col, lw=2, ls="--", marker="o", ms=3, label=f"{m} forecast")
ax.axvspan(bt["Week"].min(), bt["Week"].max(), color=TEAL, alpha=0.06, label="Backtest window")
ax.set_title(f"{names.get(code, code).title()}: forecast vs actual"); ax.set_ylabel("Units per week")
ax.legend(frameon=False, fontsize=9); fig.autofmt_xdate()
fig.tight_layout(); fig.savefig(C / "03_forecast_vs_actual.png"); plt.close(fig)

# 4. ABC-XYZ heatmap
mx = pd.read_csv(T / "abc_xyz_matrix.csv", index_col=0).reindex(columns=["X", "Y", "Z"], fill_value=0)
fig, ax = plt.subplots(figsize=(5.5, 4))
im = ax.imshow(mx.values, cmap="GnBu")
ax.set_xticks(range(3), ["X (steady)", "Y (variable)", "Z (erratic)"]); ax.set_yticks(range(3), ["A", "B", "C"])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{mx.values[i, j]:,}", ha="center", va="center", color="black")
ax.set_title("ABC-XYZ: product counts")
fig.tight_layout(); fig.savefig(C / "04_abc_xyz.png"); plt.close(fig)
print("charts:", sorted(p.name for p in C.glob("*.png")))
