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
# LOAD STOCK DATA AND PREP TARGET
# ======================================================

df = pd.read_csv(STOCK_PATH)
df = df.sort_values(["ticker", "date"])

df["close_used"] = pd.to_numeric(df["close_used"], errors="coerce")

# Next-day % return prediction target
df["target"] = df.groupby("ticker")["close_used"].shift(-1) / df["close_used"] - 1
df = df.dropna(subset=["target"])  # remove last rows per ticker


# ======================================================
# LOAD FEATURE LIST, NEWS EMBEDDING, SCALER
# ======================================================

num_cols = pickle.load(open(FEATURES_PATH, "rb"))
print("Numeric feature columns used:", len(num_cols))

news_emb = np.load(NEWS_EMB_PATH)
scaler = pickle.load(open(SCALER_PATH, "rb"))


# ======================================================
# PREPARE INPUT FEATURES
# ======================================================

num_df = df[num_cols]
num_df = num_df.replace([np.inf, -np.inf], np.nan)
num_df = num_df.fillna(method="ffill").fillna(method="bfill")

X_num = num_df.to_numpy()

# Append same news embedding to every row
X = np.hstack([X_num, np.tile(news_emb, (len(num_df), 1))])
X_scaled = scaler.transform(X)

y = df["target"].to_numpy()


# ======================================================
# LOAD TRAINED MODEL
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

    # Direction accuracy: % of correct up/down predictions
    dir_acc = (np.sign(actual) == np.sign(pred)).mean()

    plt.figure(figsize=(13, 6))

    # Plot lines
    plt.plot(actual, label="Actual Next-Day Return", linewidth=2)
    plt.plot(pred, label="Predicted Return", linestyle="--")

    # Labels + formatting
    plt.title(f"{t} – Predicted vs Actual Returns\nDirection Accuracy = {dir_acc:.2%}")
    plt.xlabel("Days (Time Index)")       # Requested axis label
    plt.ylabel("Next-Day Return (%)")     # Requested axis label
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path = os.path.join(SAVE_DIR, f"{t}_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot for {t}: {save_path}")

print("\n🎉 All ticker plots generated!")
