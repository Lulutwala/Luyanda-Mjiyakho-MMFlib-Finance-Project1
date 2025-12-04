import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE  = os.path.join(PROJECT_ROOT, "data", "Economy", "mmflib_stock_features_with_news.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "Economy", "mmflib_top5_multimodal.csv")

TOP5 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

df = pd.read_csv(INPUT_FILE)

df_filtered = df[df["ticker"].isin(TOP5)].reset_index(drop=True)

df_filtered.to_csv(OUTPUT_FILE, index=False)

print("Saved filtered dataset:", OUTPUT_FILE)
