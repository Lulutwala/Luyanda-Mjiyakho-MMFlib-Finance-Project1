import pandas as pd
import json
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STOCK_FILE = os.path.join(PROJECT_ROOT, "data", "Economy", "mmflib_stock_features.csv")
NEWS_FILE = os.path.join(PROJECT_ROOT, "data", "News", "mmflib_raw_news_data.json")

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "Economy", "mmflib_stock_features_with_news.csv")

print("Loading stock dataset:", STOCK_FILE)
df_stock = pd.read_csv(STOCK_FILE, parse_dates=["date"])

print("Loading news dataset:", NEWS_FILE)
rows = []
with open(NEWS_FILE, "r", encoding="utf8") as f:
    for line in f:
        rows.append(json.loads(line))

df_news = pd.DataFrame(rows)

# Normalize dates
df_news["date"] = pd.to_datetime(df_news["time_published"]).dt.date

# Group news by (date, ticker_main)
df_grouped = df_news.groupby(["date", "ticker_main"]).apply(
    lambda x: " ".join(x["headline"] + ". " + x["summary"])
).reset_index(name="news_text")

df_stock["date"] = df_stock["date"].dt.date

# Merge stock features with grouped news
df_merged = df_stock.merge(
    df_grouped,
    left_on=["date", "ticker"],     # from stock file
    right_on=["date", "ticker_main"],  # from news
    how="left"
)

df_merged["news_text"] = df_merged["news_text"].fillna("")

# Drop redundant column
if "ticker_main" in df_merged.columns:
    df_merged = df_merged.drop(columns=["ticker_main"])

df_merged.to_csv(OUTPUT_FILE, index=False)

print("\n====================================")
print("SUCCESS! Multimodal dataset created:")
print(OUTPUT_FILE)
print("====================================\n")
