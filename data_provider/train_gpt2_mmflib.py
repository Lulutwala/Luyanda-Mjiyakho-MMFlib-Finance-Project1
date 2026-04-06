from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from transformers import AutoModel, AutoTokenizer, GPT2Model, GPT2Tokenizer


DEFAULT_DATA_PATH = "/home/mjiyakho/MM-TSFlib/data/merged/merged_stock_news_dataset.csv"

DATE_COL = "date"
TICKER_COL = "ticker"
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
class FoldSplit:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in s).strip("_")


def load_dataset(path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if max_rows is not None and max_rows > 0:
        df = df.iloc[:max_rows].copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if CLASS_TARGET_COL in df.columns:
        df[CLASS_TARGET_COL] = pd.to_numeric(df[CLASS_TARGET_COL], errors="coerce")
    else:
        raise ValueError(f"Missing required classification target column: {CLASS_TARGET_COL}")

    df = df.dropna(subset=[DATE_COL, CLASS_TARGET_COL]).copy()
    df[CLASS_TARGET_COL] = (df[CLASS_TARGET_COL] > 0).astype(np.float32)
    df[TICKER_COL] = df[TICKER_COL].astype(str).str.upper().str.strip()
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    return df


def build_walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int,
    min_train_ratio: float,
) -> list[FoldSplit]:
    unique_dates = sorted(df[DATE_COL].dropna().unique())
    n_dates = len(unique_dates)
    if n_dates < 4:
        raise ValueError("Not enough unique dates for walk-forward validation.")

    min_train_dates = max(2, int(n_dates * min_train_ratio))
    if min_train_dates >= n_dates - 1:
        min_train_dates = max(2, n_dates - 2)

    remaining = n_dates - min_train_dates
    effective_splits = max(1, min(n_splits, remaining))
    fold_size = max(1, remaining // effective_splits)

    folds: list[FoldSplit] = []
    train_end = min_train_dates
    for fold in range(1, effective_splits + 1):
        test_start = train_end
        if test_start >= n_dates - 1:
            break
        if fold == effective_splits:
            test_end = n_dates
        else:
            test_end = min(n_dates, test_start + fold_size)
        if test_end <= test_start:
            continue

        train_start_date = pd.to_datetime(unique_dates[0])
        train_end_date = pd.to_datetime(unique_dates[test_start - 1])
        test_start_date = pd.to_datetime(unique_dates[test_start])
        test_end_date = pd.to_datetime(unique_dates[test_end - 1])

        train_idx = np.where(df[DATE_COL] <= train_end_date)[0]
        test_idx = np.where((df[DATE_COL] >= test_start_date) & (df[DATE_COL] <= test_end_date))[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        _ = train_start_date  # kept for clarity in debug prints
        folds.append(
            FoldSplit(
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                train_end_date=train_end_date,
                test_start_date=test_start_date,
                test_end_date=test_end_date,
            )
        )
        train_end = test_end

    if not folds:
        raise ValueError("Unable to build valid walk-forward splits.")
    return folds


def compose_text(df: pd.DataFrame) -> list[str]:
    texts: list[str] = []
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


def embedding_cache_paths(
    output_dir: str,
    n_rows: int,
    encoder_key: str,
    max_length: int,
) -> tuple[str, str]:
    stem = f"{safe_slug(encoder_key)}_L{max_length}_N{n_rows}"
    emb_path = os.path.join(output_dir, f"text_embeddings_{stem}.npy")
    meta_path = os.path.join(output_dir, f"text_embeddings_{stem}.meta.json")
    return emb_path, meta_path


def load_cached_embeddings(output_dir: str, n_rows: int, encoder_key: str, max_length: int) -> np.ndarray | None:
    emb_path, meta_path = embedding_cache_paths(output_dir, n_rows, encoder_key, max_length)
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


def save_cached_embeddings(
    output_dir: str,
    encoder_key: str,
    max_length: int,
    embs: np.ndarray,
) -> None:
    emb_path, meta_path = embedding_cache_paths(output_dir, embs.shape[0], encoder_key, max_length)
    np.save(emb_path, embs)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_rows": int(embs.shape[0]),
                "dim": int(embs.shape[1]),
                "encoder_key": encoder_key,
                "max_length": int(max_length),
            },
            f,
            indent=2,
        )


def embed_text_with_model(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
    tag: str,
) -> np.ndarray:
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
            print(f"  [{tag}] Embedded {min(i + batch_size, len(texts))}/{len(texts)} rows")
    return np.vstack(all_embs).astype(np.float32)


def build_news_embeddings(
    texts: list[str],
    device: str,
    batch_size: int,
    max_length: int,
    news_encoder: str,
    finbert_model_name: str,
    local_files_only: bool,
    output_dir: str,
) -> np.ndarray:
    os.makedirs(output_dir, exist_ok=True)

    gpt2_key = "gpt2"
    finbert_key = safe_slug(finbert_model_name)

    def get_gpt2_embs() -> np.ndarray:
        cached = load_cached_embeddings(output_dir, len(texts), gpt2_key, max_length)
        if cached is not None:
            print("Using cached GPT-2 embeddings.")
            return cached
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2Model.from_pretrained("gpt2", local_files_only=local_files_only).to(device)
        embs = embed_text_with_model(texts, tokenizer, model, device, batch_size, max_length, "gpt2")
        save_cached_embeddings(output_dir, gpt2_key, max_length, embs)
        return embs

    def get_finbert_embs() -> np.ndarray:
        cached = load_cached_embeddings(output_dir, len(texts), finbert_key, max_length)
        if cached is not None:
            print(f"Using cached {finbert_model_name} embeddings.")
            return cached
        tokenizer = AutoTokenizer.from_pretrained(finbert_model_name, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(finbert_model_name, local_files_only=local_files_only).to(device)
        embs = embed_text_with_model(texts, tokenizer, model, device, batch_size, max_length, "finbert")
        save_cached_embeddings(output_dir, finbert_key, max_length, embs)
        return embs

    if news_encoder == "gpt2":
        return get_gpt2_embs()
    if news_encoder == "finbert":
        return get_finbert_embs()
    if news_encoder == "both":
        gpt2_embs = get_gpt2_embs()
        finbert_embs = get_finbert_embs()
        return np.hstack([gpt2_embs, finbert_embs]).astype(np.float32)
    raise ValueError(f"Unsupported news_encoder={news_encoder}")


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in EXCLUDE_COLS]


def qcut_labels(values: pd.Series, n_bins: int = 3) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.nunique(dropna=True) <= 1:
        return pd.Series(["mid"] * len(values), index=values.index, dtype="object")

    ranks = values.rank(method="first")
    n_quantiles = int(min(n_bins, max(2, values.nunique(dropna=True))))
    if n_quantiles <= 1:
        return pd.Series(["mid"] * len(values), index=values.index, dtype="object")

    label_map = {
        2: ["small", "large"],
        3: ["small", "mid", "large"],
    }
    labels = label_map.get(n_quantiles, [f"q{i+1}" for i in range(n_quantiles)])

    try:
        bins = pd.qcut(ranks, q=n_quantiles, labels=labels, duplicates="drop")
        return bins.astype(str)
    except ValueError:
        return pd.Series(["mid"] * len(values), index=values.index, dtype="object")


def build_identity_features(df: pd.DataFrame, ticker_meta_path: str | None) -> tuple[np.ndarray, dict]:
    ticker_map = {t: i for i, t in enumerate(sorted(df[TICKER_COL].dropna().unique()))}
    ticker_id = df[TICKER_COL].map(ticker_map).fillna(0).astype(int)

    if ticker_meta_path and os.path.exists(ticker_meta_path):
        meta = pd.read_csv(ticker_meta_path)
        meta["ticker"] = meta["ticker"].astype(str).str.upper().str.strip()
        sector_series = meta.set_index("ticker").get("sector", pd.Series(dtype=object))
        market_cap_series = pd.to_numeric(meta.set_index("ticker").get("market_cap", pd.Series(dtype=float)), errors="coerce")
        sector = df[TICKER_COL].map(sector_series).fillna("UNKNOWN").astype(str)
        if market_cap_series.empty:
            market_cap = pd.Series(np.nan, index=df.index)
        else:
            market_cap = df[TICKER_COL].map(market_cap_series)
    else:
        sector = pd.Series(["UNKNOWN"] * len(df), index=df.index)
        market_cap = pd.Series(np.nan, index=df.index)

    if market_cap.isna().all():
        close_col = "close_used" if "close_used" in df.columns else "close"
        proxy = (pd.to_numeric(df.get(close_col, 0), errors="coerce").fillna(0.0) *
                 pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0))
        proxy_by_ticker = proxy.groupby(df[TICKER_COL]).median()
        market_bucket_map = qcut_labels(proxy_by_ticker, n_bins=3).to_dict()
        market_bucket = df[TICKER_COL].map(market_bucket_map).fillna("mid")
    else:
        by_ticker = market_cap.groupby(df[TICKER_COL]).median().fillna(market_cap.median())
        market_bucket_map = qcut_labels(by_ticker, n_bins=3).to_dict()
        market_bucket = df[TICKER_COL].map(market_bucket_map).fillna("mid")

    sector_values = sorted(sector.astype(str).unique())
    sector_map = {s: i for i, s in enumerate(sector_values)}
    sector_id = sector.astype(str).map(sector_map).fillna(0).astype(int)

    mcap_values = sorted(market_bucket.astype(str).unique())
    mcap_map = {b: i for i, b in enumerate(mcap_values)}
    mcap_id = market_bucket.astype(str).map(mcap_map).fillna(0).astype(int)

    cat_matrix = np.vstack([ticker_id.to_numpy(), sector_id.to_numpy(), mcap_id.to_numpy()]).T.astype(np.int64)
    vocab = {
        "ticker": ticker_map,
        "sector": sector_map,
        "market_cap_bucket": mcap_map,
    }
    return cat_matrix, vocab


class TabularClassifier(nn.Module):
    def __init__(self, cont_dim: int, cat_cardinalities: list[int]):
        super().__init__()
        self.emb_layers = nn.ModuleList()
        embed_total = 0
        for card in cat_cardinalities:
            dim = max(2, min(16, int(np.ceil(np.sqrt(max(2, card))))))
            self.emb_layers.append(nn.Embedding(card, dim))
            embed_total += dim

        in_dim = cont_dim + embed_total
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        chunks = [x_cont]
        if x_cat is not None and len(self.emb_layers) > 0:
            for i, emb in enumerate(self.emb_layers):
                chunks.append(emb(x_cat[:, i]))
        x = torch.cat(chunks, dim=1)
        return self.net(x)


def best_threshold_by_f1(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true_int = y_true.astype(int)
    best_t, best_f1, best_bal = 0.5, -1.0, -1.0
    for t in np.linspace(0.2, 0.8, 25):
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true_int, pred, zero_division=0)
        bal = balanced_accuracy_score(y_true_int, pred)
        if f1 > best_f1 or (f1 == best_f1 and bal > best_bal):
            best_t, best_f1, best_bal = float(t), float(f1), float(bal)
    return best_t


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y_true_int = y_true.astype(int)
    pred = (probs >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true_int, pred)),
        "f1": float(f1_score(y_true_int, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_int, pred)),
        "threshold": float(threshold),
    }
    out["roc_auc"] = float(roc_auc_score(y_true_int, probs)) if np.unique(y_true_int).size > 1 else float("nan")
    return out


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_cat: np.ndarray | None,
    test_cat: np.ndarray | None,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    cat_cardinalities: list[int] = []
    if train_cat is not None:
        for i in range(train_cat.shape[1]):
            cat_cardinalities.append(int(max(train_cat[:, i].max(), test_cat[:, i].max()) + 1))

    model = TabularClassifier(cont_dim=X_train.shape[1], cat_cardinalities=cat_cardinalities).to(device)

    pos_count = float(np.sum(y_train > 0.5))
    neg_count = float(len(y_train) - pos_count)
    if pos_count > 0:
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    train_cat_t = torch.tensor(train_cat, dtype=torch.long).to(device) if train_cat is not None else None
    test_cat_t = torch.tensor(test_cat, dtype=torch.long).to(device) if test_cat is not None else None

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(X_train_t.size(0), device=device)
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            cb = train_cat_t[idx] if train_cat_t is not None else None

            opt.zero_grad()
            logits = model(xb, cb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t, test_cat_t).cpu().numpy().reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    threshold = best_threshold_by_f1(y_test, probs)
    metrics = classification_metrics(y_test, probs, threshold)
    metrics["train_pos_rate"] = float(np.mean(y_train))
    metrics["test_pos_rate"] = float(np.mean(y_test))
    return metrics


def parse_ablations(raw: str) -> list[str]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    allowed = {"numeric_only", "news_only", "fusion"}
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"Unsupported ablation(s): {bad}. Allowed: {sorted(allowed)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser("Walk-forward classification for stock up/down with news + numeric fusion")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "merged", "training_outputs"))
    parser.add_argument("--max_rows", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--walk_splits", type=int, default=5)
    parser.add_argument("--min_train_ratio", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)

    parser.add_argument("--news_encoder", type=str, default="both", choices=["gpt2", "finbert", "both"])
    parser.add_argument("--finbert_model_name", type=str, default="ProsusAI/finbert")
    parser.add_argument("--embed_batch_size", type=int, default=16)
    parser.add_argument("--embed_max_length", type=int, default=96)
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--ticker_meta_path", type=str, default="")
    parser.add_argument("--use_identity_features", action="store_true", default=True)
    parser.add_argument("--no_identity_features", dest="use_identity_features", action="store_false")
    parser.add_argument("--ablations", type=str, default="numeric_only,news_only,fusion")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    max_rows = None if args.max_rows <= 0 else args.max_rows
    df = load_dataset(args.data_path, max_rows=max_rows)
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
    print(f"Tickers: {df[TICKER_COL].nunique()}")

    folds = build_walk_forward_splits(df, n_splits=args.walk_splits, min_train_ratio=args.min_train_ratio)
    print(f"Walk-forward folds: {len(folds)}")

    print("Preparing text embeddings...")
    texts = compose_text(df)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    text_embs = build_news_embeddings(
        texts=texts,
        device=device,
        batch_size=args.embed_batch_size,
        max_length=args.embed_max_length,
        news_encoder=args.news_encoder,
        finbert_model_name=args.finbert_model_name,
        local_files_only=args.local_files_only,
        output_dir=args.output_dir,
    )
    print(f"Text embedding dim: {text_embs.shape[1]}")

    numeric_features = get_numeric_features(df)
    X_num_raw = df[numeric_features].apply(pd.to_numeric, errors="coerce")
    X_num_raw = X_num_raw.replace([np.inf, -np.inf], np.nan)
    X_num_raw = X_num_raw.fillna(X_num_raw.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=np.float32)
    y = df[CLASS_TARGET_COL].astype(np.float32).to_numpy()

    cat_matrix = None
    identity_vocab = {}
    if args.use_identity_features:
        cat_matrix, identity_vocab = build_identity_features(df, args.ticker_meta_path or None)
        print(
            "Identity features enabled: "
            f"ticker={len(identity_vocab['ticker'])}, "
            f"sector={len(identity_vocab['sector'])}, "
            f"mcap_bucket={len(identity_vocab['market_cap_bucket'])}"
        )
    else:
        print("Identity features disabled.")

    ablations = parse_ablations(args.ablations)
    rows: list[dict] = []

    for fold in folds:
        print(
            f"\nFold {fold.fold}: train<= {fold.train_end_date.date()} | "
            f"test= {fold.test_start_date.date()}..{fold.test_end_date.date()}"
        )
        scaler = StandardScaler()
        X_num_train = scaler.fit_transform(X_num_raw[fold.train_idx])
        X_num_test = scaler.transform(X_num_raw[fold.test_idx])

        X_text_train = text_embs[fold.train_idx]
        X_text_test = text_embs[fold.test_idx]

        y_train = y[fold.train_idx]
        y_test = y[fold.test_idx]

        train_cat = cat_matrix[fold.train_idx] if cat_matrix is not None else None
        test_cat = cat_matrix[fold.test_idx] if cat_matrix is not None else None

        for ablation in ablations:
            if ablation == "numeric_only":
                X_train, X_test = X_num_train, X_num_test
            elif ablation == "news_only":
                X_train, X_test = X_text_train, X_text_test
            else:
                X_train = np.hstack([X_num_train, X_text_train]).astype(np.float32)
                X_test = np.hstack([X_num_test, X_text_test]).astype(np.float32)

            print(f"  Training ablation: {ablation}")
            metrics = train_classifier(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_cat=train_cat,
                test_cat=test_cat,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )

            row = {
                "fold": fold.fold,
                "ablation": ablation,
                "train_end_date": fold.train_end_date.date().isoformat(),
                "test_start_date": fold.test_start_date.date().isoformat(),
                "test_end_date": fold.test_end_date.date().isoformat(),
                "n_train": int(len(fold.train_idx)),
                "n_test": int(len(fold.test_idx)),
                **metrics,
            }
            rows.append(row)
            print(
                "    "
                f"f1={metrics['f1']:.4f} "
                f"auc={metrics['roc_auc']:.4f} "
                f"bal_acc={metrics['balanced_accuracy']:.4f} "
                f"acc={metrics['accuracy']:.4f}"
            )

    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby("ablation", as_index=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .sort_values(["f1_mean", "roc_auc_mean", "balanced_accuracy_mean"], ascending=False)
    )

    fold_path = os.path.join(args.output_dir, "gpt2_mmflib_classification_fold_metrics.csv")
    summary_path = os.path.join(args.output_dir, "gpt2_mmflib_metrics.csv")
    config_path = os.path.join(args.output_dir, "gpt2_mmflib_run_config.json")
    features_path = os.path.join(args.output_dir, "numeric_features.json")
    vocab_path = os.path.join(args.output_dir, "identity_vocab.json")

    fold_df.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(numeric_features, f, indent=2)
    if identity_vocab:
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(identity_vocab, f, indent=2)

    print("\nAblation summary (walk-forward):")
    print(summary.to_string(index=False))
    print(f"\nSaved fold metrics: {fold_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
