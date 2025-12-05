import pandas as pd
import numpy as np
import os

BASE = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1"
STOCK_PATH = os.path.join(BASE, "data", "Economy", "mmflib_stock_features.csv")

df = pd.read_csv(STOCK_PATH)

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMN TYPES ===")
print(df.dtypes)

print("\n=== COUNT NaNs PER COLUMN ===")
print(df.isna().sum())

print("\n=== COLUMNS WITH NON-NUMERIC VALUES ===")
bad_cols = []
for col in df.columns:
    try:
        pd.to_numeric(df[col], errors="raise")
    except:
        bad_cols.append(col)

print(bad_cols)

print("\n=== SAMPLE BAD CELLS ===")
for col in bad_cols:
    print(f"\n---- {col} ----")
    print(df[col].unique()[:20])
