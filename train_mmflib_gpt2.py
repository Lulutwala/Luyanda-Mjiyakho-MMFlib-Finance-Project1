import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1"
DATA_PATH = os.path.join(BASE_DIR, "data", "merged", "mmflib_merged_stock_news.csv")

print("📌 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

# ============================================================
# DATA FIXING (Critical)
# ============================================================

# Replace bad values with NaN
df = df.replace(["", " ", "  ", None, "NA", "N/A", "null"], np.nan)

# Convert ALL non-text columns to numeric
text_cols = ["ticker", "date", "headline", "summary", "source"]

for col in df.columns:
    if col not in text_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fix missing text fields
for col in ["headline", "summary", "source"]:
    df[col] = df[col].fillna("").astype(str)

# Remove rows where close_used is invalid
df = df[df["close_used"].notna()]

# Replace Inf with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Fill numeric NaNs with column medians
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# ============================================================
# TARGET VARIABLE
# Predict NEXT DAY RETURN
# ============================================================

df["target"] = df.groupby("ticker")["close_used"].shift(-1) / df["close_used"] - 1
df = df.dropna(subset=["target"])

print("After cleaning:", df.shape)

# ============================================================
# GPT-2 MODEL + TOKENIZER
# ============================================================

print("\n📌 Loading GPT-2 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load tokenizer ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# FIX: GPT2 has no pad token → assign EOS as padding token
tokenizer.pad_token = tokenizer.eos_token

# --- Load GPT-2 model ---
gpt2 = GPT2Model.from_pretrained("gpt2")

# Resize embeddings to include the new pad token
gpt2.resize_token_embeddings(len(tokenizer))

gpt2 = gpt2.to(device)
gpt2.eval()  # inference mode


# ============================================================
# TEXT EMBEDDING FUNCTION (Safe)
# ============================================================

def embed_text_batch(text_list):
    """Batch-encode text for speed."""
    if len(text_list) == 0:
        return np.zeros((0, 768))

    enc = tokenizer(
        text_list,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=40
    ).to(device)

    with torch.no_grad():
        out = gpt2(**enc).last_hidden_state.mean(dim=1)

    return out.cpu().numpy()

print("\n📌 Embedding headlines & summaries…")

text_data = (df["headline"] + " " + df["summary"]).tolist()

BATCH = 64
embeddings = []

for i in range(0, len(text_data), BATCH):
    batch = text_data[i:i+BATCH]
    emb = embed_text_batch(batch)
    embeddings.append(emb)
    print(f" → Embedded {i+len(batch)} / {len(text_data)}")

df["text_embedding"] = list(np.vstack(embeddings))

# ============================================================
# FEATURE MATRIX
# ============================================================

exclude = ["ticker", "date", "headline", "summary", "source", "text_embedding", "target"]
num_features = [c for c in df.columns if c not in exclude and df[c].dtype != "object"]

X_num = df[num_features].values
X_text = np.vstack(df["text_embedding"].values)
y = df["target"].values

# Standardize numeric features
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

# Combine numeric + text
X = np.hstack([X_num, X_text])

print("\nFinal feature matrix:", X.shape)

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=False
)

# ============================================================
# MODEL
# ============================================================

model = nn.Sequential(
    nn.Linear(X.shape[1], 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ============================================================
# TRAINING
# ============================================================

epochs = 10
batch_size = 256

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

print("\n📌 Starting training...")

for epoch in range(epochs):

    permutation = torch.randperm(X_train_t.size()[0])
    losses = []

    for i in range(0, X_train_t.size()[0], batch_size):
        idx = permutation[i:i+batch_size]
        batch_x = X_train_t[idx]
        batch_y = y_train_t[idx]

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch+1}/{epochs} | Loss: {np.mean(losses):.6f}")

# ============================================================
# EVALUATION
# ============================================================

print("\n📌 Evaluating...")

model.eval()
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    preds = model(X_test_t).cpu().numpy()

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("\n🎯 TEST RMSE:", rmse)
# ==========================
# BUILD FINAL FEATURE MATRIX
# ==========================

exclude_cols = ["ticker", "date", "headline", "summary", "source", "text_embedding", "target"]

# Step 1 — keep only numeric columns
num_df = df.drop(columns=exclude_cols)

# Step 2 — drop columns that are all NaN
num_df = num_df.dropna(axis=1, how="all")

# Step 3 — drop columns with zero variance (constant values)
num_df = num_df.loc[:, num_df.nunique() > 1]

print("Numeric columns AFTER cleaning:", num_df.shape[1])

# Final usable numerical features
X_num = num_df.values

# Text embeddings
X_text = np.vstack(df["text_embedding"].values)

# Target
y = df["target"].values

# ============================================================
# SAVE MODEL + SCALER
# ============================================================

torch.save(model.state_dict(), "mmflib_gpt2_model.pt")

import pickle
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\n🎉 TRAINING COMPLETE! Model saved as mmflib_gpt2_model.pt")
