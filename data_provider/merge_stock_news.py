import pandas as pd
import os

# ===== PATHS =====
BASE_DIR = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1"

STOCK_PATH = os.path.join(BASE_DIR, "data", "Economy", "mmflib_stock_features.csv")
NEWS_PATH = os.path.join(BASE_DIR, "data", "News", "mmflib_raw_news_data.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "merged")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mmflib_merged_stock_news.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== LOAD STOCK DATA =====
df_stock = pd.read_csv(STOCK_PATH)

# normalize date column
df_stock["date"] = pd.to_datetime(df_stock["date"], errors="coerce").dt.strftime("%Y-%m-%d")

print("Stock rows:", len(df_stock))
print("Stock date range:", df_stock["date"].min(), "→", df_stock["date"].max())

# ===== LOAD NEWS JSON =====
df_news = pd.read_json(NEWS_PATH, lines=True)

print("Raw news rows:", len(df_news))

# ===== CLEAN NEWS =====
df_news = df_news.rename(columns={"ticker_main": "ticker"})

# extract YYYYMMDD
df_news["date"] = df_news["time_published"].str[:8]

# convert to YYYY-MM-DD
df_news["date"] = pd.to_datetime(df_news["date"], format="%Y%m%d", errors="coerce") \
                      .dt.strftime("%Y-%m-%d")

# keep only MMFLIB-required text columns
df_news = df_news[["ticker", "date", "headline", "summary", "source"]]

print("\nCleaned news sample:")
print(df_news.head())

# ===== MERGE =====
df_merged = df_stock.merge(
    df_news,
    on=["ticker", "date"],
    how="left"   # KEEP ALL STOCK DATA
)

# Fill empty news (important for training)
df_merged["headline"] = df_merged["headline"].fillna("")
df_merged["summary"] = df_merged["summary"].fillna("")
df_merged["source"] = df_merged["source"].fillna("")

# ===== SAVE FINAL MERGED DATASET =====
df_merged.to_csv(OUTPUT_FILE, index=False)

print("\n✅ MERGE COMPLETE!")
print("Saved to:", OUTPUT_FILE)
print("Final shape:", df_merged.shape)
print("\nSample merged rows:")
print(df_merged[["ticker", "date", "headline"]].head())
