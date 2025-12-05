import os
import json
import numpy as np
import pandas as pd

import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle

from transformers import GPT2Tokenizer, GPT2Model

# ======================================================
# PATHS
# ======================================================

BASE_DIR = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1"

STOCK_PATH = os.path.join(BASE_DIR, "data", "Economy", "mmflib_stock_features.csv")
NEWS_JSON_PATH = os.path.join(BASE_DIR, "data", "News", "mmflib_raw_news_data.json")
NEWS_EMB_PATH = os.path.join(BASE_DIR, "data", "News", "gpt2_news_embedding.npy")

MODEL_PATH = os.path.join(BASE_DIR, "mmflib_gpt2_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "selected_features.pkl")

# ======================================================
# 1. LOAD & CLEAN STOCK DATA
# ======================================================

print("📌 Loading stock data...")
df = pd.read_csv(STOCK_PATH)
print("Stock rows:", df.shape)

df = df.sort_values(["ticker", "date"])
df["close_used"] = pd.to_numeric(df["close_used"], errors="coerce")

df["target"] = df.groupby("ticker")["close_used"].shift(-1) / df["close_used"] - 1
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["target"])

print("Rows after target dropna:", df.shape)

# ======================================================
# 2. LOAD OR BUILD GLOBAL NEWS EMBEDDING
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_global_news_embedding():
    print("📌 Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained("gpt2").to(device)
    model.eval()

    print("📌 Loading raw news JSON...")
    news_rows = []
    with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                news_rows.append(json.loads(line))
            except:
                continue

    if len(news_rows) == 0:
        print("⚠ No news found — using zero embedding.")
        return np.zeros(768, dtype=np.float32)

    texts = []
    for n in news_rows:
        h = str(n.get("headline", "") or "")
        s = str(n.get("summary", "") or "")
        txt = (h + " " + s).strip()
        if txt:
            texts.append(txt)

    all_embs = []
    batch_size = 16
    print("📌 Embedding news (global sentiment)...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        padding=True, max_length=64).to(device)
        with torch.no_grad():
            out = model(**enc)
            emb = out.last_hidden_state.mean(dim=1)
        all_embs.append(emb.cpu().numpy())

    all_embs = np.vstack(all_embs)
    return all_embs.mean(axis=0).astype(np.float32)


if os.path.exists(NEWS_EMB_PATH):
    print("📌 Loading precomputed news embedding...")
    global_news_emb = np.load(NEWS_EMB_PATH)
else:
    global_news_emb = build_global_news_embedding()
    np.save(NEWS_EMB_PATH, global_news_emb)
    print("📌 Saved news embedding.")

print("News embedding shape:", global_news_emb.shape)

# ======================================================
# 3. CLEAN NUMERICAL FEATURE MATRIX
# ======================================================

num_cols = [
    c for c in df.columns
    if df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
    and c != "target"
]

X_raw = df[num_cols].to_numpy()
X_raw = np.where(np.isfinite(X_raw), X_raw, np.nan)

col_nan = np.all(np.isnan(X_raw), axis=0)
col_std = np.nanstd(X_raw, axis=0)
col_const = (col_std == 0) | np.isnan(col_std)

drop_mask = col_nan | col_const
kept_indices = np.where(~drop_mask)[0]
kept_cols = [num_cols[i] for i in kept_indices]

print("\nDropped columns (all-NaN or constant):")
for i, name in enumerate(num_cols):
    if drop_mask[i]:
        print("  -", name)

print("\nKept numeric columns:", len(kept_cols))

X_num = X_raw[:, kept_indices]

# Fill NaNs with column means
means = np.nanmean(X_num, axis=0)
inds = np.where(np.isnan(X_num))
X_num[inds] = np.take(means, inds[1])

X_num = np.where(np.isfinite(X_num), X_num, 0.0)

# SAVE selected features list
with open(FEATURES_PATH, "wb") as f:
    pickle.dump(kept_cols, f)
print(f"📌 Saved {len(kept_cols)} selected features to {FEATURES_PATH}")

# ======================================================
# 4. BUILD FINAL X
# ======================================================

news_block = np.tile(global_news_emb, (X_num.shape[0], 1))
X = np.hstack([X_num, news_block])
y = df["target"].to_numpy()

rowmask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
X = X[rowmask]
y = y[rowmask]

print("\nFinal feature shape:", X.shape)
print("Final target length:", y.shape)

# ======================================================
# 5. TRAIN / TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=False
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# 6. MODEL
# ======================================================

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

print("\n📌 TRAINING...")
batch = 256
epochs = 10

for ep in range(1, epochs + 1):
    perm = torch.randperm(X_train_t.size(0))
    losses = []
    for i in range(0, len(perm), batch):
        idx = perm[i:i + batch]
        xb = X_train_t[idx]
        yb = y_train_t[idx]

        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"Epoch {ep}/{epochs} Loss={np.mean(losses):.6f}")

# ======================================================
# 7. EVALUATION
# ======================================================

X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    preds = model(X_test_t).cpu().numpy().reshape(-1)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("\n📊 Test RMSE:", rmse)

torch.save(model.state_dict(), MODEL_PATH)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Training complete.")
print("Model saved to:", MODEL_PATH)
print("Scaler saved to:", SCALER_PATH)
