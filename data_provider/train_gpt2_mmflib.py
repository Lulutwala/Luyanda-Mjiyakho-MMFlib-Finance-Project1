from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer


DEFAULT_DATA_PATH = "/home/mjiyakho/MM-TSFlib/data/merged/merged_stock_news_dataset.csv"

DATE_COL = "date"
TICKER_COL = "ticker"
TARGET_COL = "target_next_return"
CLASS_TARGET_COL = "target_up_down"
TEXT_COLS = ["all_headlines", "all_summaries", "all_news_text"]

EXCLUDE_COLS = {
    DATE_COL,
    TICKER_COL,
    "source_list",
    "all_headlines",
    "all_summaries",
    "all_news_text",
    "latest_news_time",
    "earliest_news_time",
    "target_next_close",
    "target_next_return",
    "target_up_down",
    "Month",
    "start_date",
    "end_date",
}


@dataclass
class SplitData:
    train_idx: np.ndarray
    test_idx: np.ndarray
    split_date: pd.Timestamp


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if max_rows is not None and max_rows > 0:
        df = df.iloc[:max_rows].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if CLASS_TARGET_COL in df.columns:
        df[CLASS_TARGET_COL] = pd.to_numeric(df[CLASS_TARGET_COL], errors="coerce")
    else:
        # Fallback for datasets that do not already include a binary class label.
        df[CLASS_TARGET_COL] = (df[TARGET_COL] >= 0).astype(np.float32)
    df = df.dropna(subset=[DATE_COL, TARGET_COL, CLASS_TARGET_COL]).copy()
    df[CLASS_TARGET_COL] = (df[CLASS_TARGET_COL] > 0).astype(np.float32)
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> SplitData:
    unique_dates = sorted(df[DATE_COL].dropna().unique())
    split_idx = int(len(unique_dates) * train_ratio)
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    split_date = pd.to_datetime(unique_dates[split_idx])
    train_idx = np.where(df[DATE_COL] < split_date)[0]
    test_idx = np.where(df[DATE_COL] >= split_date)[0]
    return SplitData(train_idx=train_idx, test_idx=test_idx, split_date=split_date)


def compose_text(df: pd.DataFrame) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in TEXT_COLS:
            value = row.get(col, "")
            if pd.notna(value):
                value = str(value).strip()
                if value and value.lower() != "nan":
                    parts.append(value)
        texts.append(" ".join(parts).strip())
    return texts


def embedding_cache_paths(output_dir: str, n_rows: int) -> tuple[str, str]:
    emb_path = os.path.join(output_dir, "gpt2_text_embeddings.npy")
    meta_path = os.path.join(output_dir, "gpt2_text_embeddings.meta.json")
    return emb_path, meta_path


def load_cached_embeddings(output_dir: str, n_rows: int) -> np.ndarray | None:
    emb_path, meta_path = embedding_cache_paths(output_dir, n_rows)
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("n_rows") != n_rows:
            return None
        return np.load(emb_path)
    except Exception:
        return None


def save_cached_embeddings(output_dir: str, embs: np.ndarray) -> None:
    emb_path, meta_path = embedding_cache_paths(output_dir, embs.shape[0])
    np.save(emb_path, embs)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"n_rows": int(embs.shape[0]), "dim": int(embs.shape[1])}, f, indent=2)


def build_gpt2_embeddings(
    texts: list[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained("gpt2").to(device)
    model.eval()

    all_embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        cleaned = [t if t else "<|endoftext|>" for t in batch_texts]
        enc = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
            emb = out.last_hidden_state.mean(dim=1)
        all_embs.append(emb.cpu().numpy())

        if (i // batch_size + 1) % 50 == 0:
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} rows")

    return np.vstack(all_embs).astype(np.float32)


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in EXCLUDE_COLS]


class Regressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_nn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[dict, Classifier]:
    model = Classifier(X_train.shape[1]).to(device)

    pos_count = float(np.sum(y_train > 0.5))
    neg_count = float(len(y_train) - pos_count)
    if pos_count > 0:
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(X_train_t.size(0), device=device)
        losses = []
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        print(f"  Epoch {epoch:02d}/{epochs} | cls_train_loss={np.mean(losses):.6f}")

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    y_true = y_test.astype(int)

    cls_metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if np.unique(y_true).size > 1 else float("nan"),
    }
    return cls_metrics, model


def train_nn_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[dict, Regressor]:
    model = Regressor(X_train.shape[1]).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(X_train_t.size(0), device=device)
        losses = []
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        print(f"  Epoch {epoch:02d}/{epochs} | train_loss={np.mean(losses):.6f}")

    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().reshape(-1)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "directional_acc": float(np.mean((preds >= 0) == (y_test >= 0))),
    }
    return metrics, model


def main():
    parser = argparse.ArgumentParser("Train GPT-2 and MMF-Lib models on merged stock-news CSV")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "merged", "training_outputs"))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_batch_size", type=int, default=16)
    parser.add_argument("--embed_max_length", type=int, default=96)
    parser.add_argument("--max_rows", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    max_rows = None if args.max_rows <= 0 else args.max_rows
    df = load_dataset(args.data_path, max_rows=max_rows)
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")

    split = time_split(df, train_ratio=args.train_ratio)
    print(f"Train rows: {len(split.train_idx):,}")
    print(f"Test rows: {len(split.test_idx):,}")
    print(f"Split date: {split.split_date.date()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    texts = compose_text(df)
    cached_embs = load_cached_embeddings(args.output_dir, len(df))
    if cached_embs is not None:
        print("Using cached GPT-2 embeddings.")
        text_embs = cached_embs
    else:
        print("Building GPT-2 text embeddings...")
        text_embs = build_gpt2_embeddings(
            texts=texts,
            device=device,
            batch_size=args.embed_batch_size,
            max_length=args.embed_max_length,
        )
        save_cached_embeddings(args.output_dir, text_embs)
        print("Saved GPT-2 embeddings cache.")

    y_reg = df[TARGET_COL].astype(float).to_numpy()
    y_cls = df[CLASS_TARGET_COL].astype(np.float32).to_numpy()

    print("\nTraining model: gpt2_only (classification)")
    X_text_train = text_embs[split.train_idx]
    X_text_test = text_embs[split.test_idx]
    y_cls_train = y_cls[split.train_idx]
    y_cls_test = y_cls[split.test_idx]
    y_reg_train = y_reg[split.train_idx]
    y_reg_test = y_reg[split.test_idx]

    gpt2_cls_metrics, gpt2_cls_model = train_nn_classifier(
        X_train=X_text_train,
        y_train=y_cls_train,
        X_test=X_text_test,
        y_test=y_cls_test,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print("\nTraining model: gpt2_only (regression secondary)")
    gpt2_metrics, gpt2_model = train_nn_regressor(
        X_train=X_text_train,
        y_train=y_reg_train,
        X_test=X_text_test,
        y_test=y_reg_test,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    numeric_features = get_numeric_features(df)
    X_num = df[numeric_features].apply(pd.to_numeric, errors="coerce")
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    X_num = X_num.fillna(X_num.median(numeric_only=True))
    X_num = X_num.fillna(0.0)
    X_num = X_num.to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_num[split.train_idx])
    X_num_test = scaler.transform(X_num[split.test_idx])

    X_mmf_train = np.hstack([X_num_train, X_text_train]).astype(np.float32)
    X_mmf_test = np.hstack([X_num_test, X_text_test]).astype(np.float32)

    print("\nTraining model: mmflib_fusion (classification)")
    mmf_cls_metrics, mmf_cls_model = train_nn_classifier(
        X_train=X_mmf_train,
        y_train=y_cls_train,
        X_test=X_mmf_test,
        y_test=y_cls_test,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print("\nTraining model: mmflib_fusion (regression secondary)")
    mmf_metrics, mmf_model = train_nn_regressor(
        X_train=X_mmf_train,
        y_train=y_reg_train,
        X_test=X_mmf_test,
        y_test=y_reg_test,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    metrics_df = pd.DataFrame(
        [
            {"model": "gpt2_only", **{f"cls_{k}": v for k, v in gpt2_cls_metrics.items()}, **{f"reg_{k}": v for k, v in gpt2_metrics.items()}},
            {"model": "mmflib_fusion", **{f"cls_{k}": v for k, v in mmf_cls_metrics.items()}, **{f"reg_{k}": v for k, v in mmf_metrics.items()}},
        ]
    ).sort_values(["cls_roc_auc", "cls_f1"], ascending=False)

    cls_df = pd.DataFrame(
        [
            {"model": "gpt2_only", **gpt2_cls_metrics},
            {"model": "mmflib_fusion", **mmf_cls_metrics},
        ]
    ).sort_values(["roc_auc", "f1"], ascending=False)

    reg_df = pd.DataFrame(
        [
            {"model": "gpt2_only", **gpt2_metrics},
            {"model": "mmflib_fusion", **mmf_metrics},
        ]
    ).sort_values("rmse")

    metrics_path = os.path.join(args.output_dir, "gpt2_mmflib_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    cls_metrics_path = os.path.join(args.output_dir, "gpt2_mmflib_classification_metrics.csv")
    reg_metrics_path = os.path.join(args.output_dir, "gpt2_mmflib_regression_metrics.csv")
    cls_df.to_csv(cls_metrics_path, index=False)
    reg_df.to_csv(reg_metrics_path, index=False)

    torch.save(gpt2_cls_model.state_dict(), os.path.join(args.output_dir, "gpt2_only_classifier_model.pt"))
    torch.save(mmf_cls_model.state_dict(), os.path.join(args.output_dir, "mmflib_fusion_classifier_model.pt"))
    torch.save(gpt2_model.state_dict(), os.path.join(args.output_dir, "gpt2_only_model.pt"))
    torch.save(mmf_model.state_dict(), os.path.join(args.output_dir, "mmflib_fusion_model.pt"))
    np.save(os.path.join(args.output_dir, "target_test.npy"), y_reg_test)
    np.save(os.path.join(args.output_dir, "target_up_down_test.npy"), y_cls_test)
    with open(os.path.join(args.output_dir, "numeric_features.json"), "w", encoding="utf-8") as f:
        json.dump(numeric_features, f, indent=2)

    print("\nResults:")
    print("\nClassification (primary):")
    print(cls_df.to_string(index=False))
    print("\nRegression (secondary):")
    print(reg_df.to_string(index=False))
    print("\nCombined summary:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved classification metrics: {cls_metrics_path}")
    print(f"Saved regression metrics: {reg_metrics_path}")
    print(f"Saved artifacts in: {args.output_dir}")


if __name__ == "__main__":
    main()
