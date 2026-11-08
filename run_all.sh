#!/usr/bin/env bash
# Downloads the dataset (if missing) and runs the full pipeline.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw
if [ ! -f "data/raw/Online Retail.xlsx" ]; then
  curl -sSL -o data/raw/online_retail.zip "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
  unzip -o -q data/raw/online_retail.zip -d data/raw
fi
python src/01_clean.py
python src/02_forecast.py
python src/03_inventory.py
python src/04_charts.py
