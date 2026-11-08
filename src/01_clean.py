"""Step 1: load the UCI Online Retail data, clean it, and build weekly demand per product.

Cleaning rules (each one is printed with how many rows it removed):
  1. Cancelled invoices (InvoiceNo starts with "C") are not demand.
  2. Orders that were later cancelled (same customer, product and quantity) are removed too,
     otherwise a cancelled 80,995-unit order would look like real demand.
  3. Zero/negative quantities and prices are data errors or adjustments.
  4. Non-product codes (postage, fees, manual adjustments, samples) are not inventory.
  5. Only full weeks are kept (the first and last weeks of the file are partial).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Online Retail.xlsx"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

print("Reading Excel file (takes about a minute)...")
df = pd.read_excel(RAW, dtype={"InvoiceNo": str, "StockCode": str})
print(f"Raw rows: {len(df):,}")

log = []
def step(name, mask_keep):
    global df
    before = len(df)
    df = df[mask_keep].copy()
    log.append((name, before - len(df)))
    print(f"  - {name}: removed {before - len(df):,} rows, {len(df):,} left")

# 1 + 2: cancellations and the orders they cancel
cancel = df[df["InvoiceNo"].str.startswith("C")].copy()
cancel["Quantity"] = cancel["Quantity"].abs()
key = ["CustomerID", "StockCode", "Quantity"]
cancelled_keys = set(map(tuple, cancel[key].dropna().values.tolist()))
is_cancel = df["InvoiceNo"].str.startswith("C")
step("cancellation invoices", ~is_cancel)
matched = df[key].apply(tuple, axis=1).isin(cancelled_keys) & df["CustomerID"].notna()
step("orders later cancelled (matched by customer, product, quantity)", ~matched)

# 3: bad quantities / prices
step("zero or negative quantity or price", (df["Quantity"] > 0) & (df["UnitPrice"] > 0))

# 4: keep real products only: codes are 5 digits, optionally followed by letters
step("non-product codes (postage, fees, adjustments)", df["StockCode"].str.match(r"^\d{5}[A-Za-z]*$", na=False))

df["Revenue"] = df["Quantity"] * df["UnitPrice"]
df["Week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time  # weeks start Monday

# 5: full weeks only
weeks = sorted(df["Week"].unique())
first, last = weeks[0], weeks[-1]
step("partial first and last week", (df["Week"] != first) & (df["Week"] != last))

df.to_pickle(OUT / "clean_transactions.pkl")

# Weekly demand per product; weeks with no sales are filled with 0
weekly = df.groupby(["StockCode", "Week"]).agg(units=("Quantity", "sum"), revenue=("Revenue", "sum")).reset_index()
all_weeks = pd.date_range(weekly["Week"].min(), weekly["Week"].max(), freq="W-MON")
idx = pd.MultiIndex.from_product([weekly["StockCode"].unique(), all_weeks], names=["StockCode", "Week"])
weekly = weekly.set_index(["StockCode", "Week"]).reindex(idx, fill_value=0).reset_index()
weekly.to_csv(OUT / "weekly_demand.csv", index=False)

names = df.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
names.to_csv(OUT / "product_names.csv")

pd.DataFrame(log, columns=["rule", "rows_removed"]).to_csv(OUT / "cleaning_log.csv", index=False)
print(f"\nClean rows: {len(df):,} | products: {df['StockCode'].nunique():,} | full weeks: {len(all_weeks)} "
      f"({all_weeks[0].date()} to {all_weeks[-1].date()}) | revenue: £{df['Revenue'].sum():,.0f}")
