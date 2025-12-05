import os
import json
import numpy as np
import pandas as pd

import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

# ======================================================
# 1. LOAD & CLEAN STOCK DATA
# ======================================================

print("📌 Loading stock data...")
df = pd.read_csv(STOCK_PATH)
print("Stock rows:", df.shape)

# Compute target: next-day return per ticker
df = df.sort_values(["ticker", "date"])
df["close_used"] = pd.to_numeric(df["close_used"], errors="coerce")

df["target"] = df.groupby("ticker")["close_used"].shift(-1) / df["close_used"] - 1

# Remove rows with invalid target
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["target"])
print("Rows after target dropna:", df.shape)

# ======================================================
# 2. LOAD / BUILD GLOBAL NEWS EMBEDDING
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_global_news_embedding():
    print("📌 Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Use EOS token as pad to avoid padding error
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained("gpt2").to(device)
    model.eval()

    print("📌 Loading raw news JSON...")
    news_rows = []
    with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                news_rows.append(obj)
            except json.JSONDecodeError:
                continue

    print("Total news articles:", len(news_rows))
    if len(news_rows) == 0:
        print("⚠ No news found; using zero embedding.")
        return np.zeros(768, dtype=np.float32)

    texts = []
    for a in news_rows:
        h = a.get("headline", "") or ""
        s = a.get("summary", "") or ""
        txt = (str(h) + " " + str(s)).strip()
        if txt:
            texts.append(txt)

    if not texts:
        print("⚠ No valid text in news; using zero embedding.")
        return np.zeros(768, dtype=np.float32)

    # Batch encode all texts and average all token embeddings
    all_embs = []
    batch_size = 16

    print("📌 Embedding news (global market sentiment)…")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        ).to(device)

        with torch.no_grad():
            out = model(**enc)
            # mean over sequence length -> (batch, 768)
            emb = out.last_hidden_state.mean(dim=1)

        all_embs.append(emb.cpu().numpy())

        if (i // batch_size) % 20 == 0:
            print(f" → Embedded {min(i+batch_size, len(texts))} / {len(texts)}")

    all_embs = np.vstack(all_embs)  # (N_articles, 768)
    global_emb = all_embs.mean(axis=0)  # (768,)

    return global_emb.astype(np.float32)


if os.path.exists(NEWS_EMB_PATH):
    print("📌 Loading precomputed news embedding...")
    global_news_emb = np.load(NEWS_EMB_PATH)
else:
    global_news_emb = build_global_news_embedding()
    os.makedirs(os.path.dirname(NEWS_EMB_PATH), exist_ok=True)
    np.save(NEWS_EMB_PATH, global_news_emb)
    print("📌 Saved embedding:", NEWS_EMB_PATH)

print("News embedding shape:", global_news_emb.shape)

# ======================================================
# 3. BUILD NUMERICAL FEATURE MATRIX (CLEAN!)
# ======================================================

# Keep only numeric columns (float / int)
num_cols = [
    c
    for c in df.columns
    if df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
    and c not in ["target"]  # exclude target itself from inputs
]

X_num_raw = df[num_cols].to_numpy()

# Replace +/-inf with NaN
X_num_raw = np.where(np.isfinite(X_num_raw), X_num_raw, np.nan)

# Drop columns that are all NaN
col_all_nan = np.all(np.isnan(X_num_raw), axis=0)

# Drop constant columns (std == 0 or NaN)
col_std = np.nanstd(X_num_raw, axis=0)
col_const = (col_std == 0) | np.isnan(col_std)

drop_mask = col_all_nan | col_const

kept_indices = np.where(~drop_mask)[0]
kept_cols = [num_cols[i] for i in kept_indices]

print("\nDropped columns (all NaN or constant):")
for i, c in enumerate(num_cols):
    if drop_mask[i]:
        print("  -", c)

print("\nKept numeric columns:", len(kept_cols))

X_num = X_num_raw[:, kept_indices]

# Impute remaining NaNs with column mean
col_means = np.nanmean(X_num, axis=0)
inds = np.where(np.isnan(X_num))
X_num[inds] = np.take(col_means, inds[1])

# Final safety: X_num must be finite
X_num = np.where(np.isfinite(X_num), X_num, 0.0)

# ======================================================
# 4. BUILD FINAL MULTIMODAL FEATURES
# ======================================================

# Repeat global news embedding for each row
news_block = np.tile(global_news_emb, (X_num.shape[0], 1))

X = np.hstack([X_num, news_block])

y = df["target"].to_numpy()

# Final row-wise mask: keep rows where X and y are finite
row_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
X = X[row_mask]
y = y[row_mask]

print("\nFinal feature shape:", X.shape)
print("Final target length:", y.shape)

# ======================================================
# 5. TRAIN / TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=False
)

# Scale numeric + news together
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# 6. SIMPLE MLP MODEL
# ======================================================

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

batch_size = 256
epochs = 10

print("\n📌 TRAINING...")
for epoch in range(1, epochs + 1):
    model.train()
    # shuffle indices
    perm = torch.randperm(X_train_t.size(0))
    epoch_losses = []

    for i in range(0, X_train_t.size(0), batch_size):
        idx = perm[i : i + batch_size]
        xb = X_train_t[idx]
        yb = y_train_t[idx]

        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    print(f"Epoch {epoch}/{epochs}  Loss={np.mean(epoch_losses):.6f}")

# ======================================================
# 7. EVALUATION
# ======================================================

print("\n📌 Evaluating...")
model.eval()
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    preds = model(X_test_t).cpu().numpy().reshape(-1)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("📊 Test RMSE:", rmse)

# ======================================================
# 8. SAVE MODEL + SCALER
# ======================================================

torch.save(model.state_dict(), MODEL_PATH)
import pickle

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Training complete.")
print("Model saved to:", MODEL_PATH)
print("Scaler saved to:", SCALER_PATH)
