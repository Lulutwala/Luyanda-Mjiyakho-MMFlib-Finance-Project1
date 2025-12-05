import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
import matplotlib.pyplot as plt

# ======================================================
# PATHS
# ======================================================

BASE_DIR = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1"

STOCK_PATH = os.path.join(BASE_DIR, "data", "Economy", "mmflib_stock_features.csv")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "mmflib_gpt2_model.pt")
FEATURES_PATH = os.path.join(BASE_DIR, "selected_features.pkl")
NEWS_EMB_PATH = os.path.join(BASE_DIR, "data", "News", "gpt2_news_embedding.npy")

SAVE_DIR = os.path.join(BASE_DIR, "plots_by_ticker")
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# LOAD DATA, SCALER, FEATURE LIST
# ======================================================

df = pd.read_csv(STOCK_PATH)
df = df.sort_values(["ticker", "date"])

# Make sure target exists
df["close_used"] = pd.to_numeric(df["close_used"], errors="coerce")
df["target"] = df.groupby("ticker")["close_used"].shift(-1) / df["close_used"] - 1
df = df.dropna(subset=["target"])

# Load feature list
num_cols = pickle.load(open(FEATURES_PATH, "rb"))
print("Numeric feature columns used:", len(num_cols))

# Load global news embedding
news_emb = np.load(NEWS_EMB_PATH)

# Numeric data
num_df = df[num_cols]
num_df = num_df.replace([np.inf, -np.inf], np.nan)
num_df = num_df.fillna(method="ffill").fillna(method="bfill")

X_num = num_df.to_numpy()

# Combine news embedding
X = np.hstack([X_num, np.tile(news_emb, (len(num_df), 1))])
y = df["target"].to_numpy()

# Load scaler + apply
scaler = pickle.load(open(SCALER_PATH, "rb"))
X_scaled = scaler.transform(X)

# ======================================================
# LOAD MODEL
# ======================================================

model = nn.Sequential(
    nn.Linear(X_scaled.shape[1], 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Predict
with torch.no_grad():
    preds = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().reshape(-1)

df["pred"] = preds

# ======================================================
# PLOT EACH TICKER
# ======================================================

tickers = df["ticker"].unique()

for t in tickers:
    sub = df[df["ticker"] == t]

    actual = sub["target"].values
    pred = sub["pred"].values

    # direction accuracy
    actual_dir = np.sign(actual)
    pred_dir = np.sign(pred)
    dir_acc = (actual_dir == pred_dir).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Next-Day Return")
    plt.plot(pred, label="Predicted Return")
    plt.title(f"{t} – Prediction vs Actual (Dir Acc={dir_acc:.2%})")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(SAVE_DIR, f"{t}_prediction.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot for {t}: {save_path}")

print("\n🎉 All ticker plots generated!")
