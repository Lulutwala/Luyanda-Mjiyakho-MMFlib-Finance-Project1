import os
import re
import numpy as np
import pandas as pd

PROJECT_ROOT = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1"

STOCK_FILE = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1\data\Economy\mmflib_stock_features.csv"
NEWS_FILE = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1\data\News\financial_news_monthly_2015_to_current.csv"
OUTPUT_FILE = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1\data\merged_stock_news_dataset.csv"

STOCK_DATE_COL = "date"
STOCK_TICKER_COL = "ticker"
STOCK_OPEN_COL = "open"
STOCK_HIGH_COL = "high"
STOCK_LOW_COL = "low"
STOCK_CLOSE_COL = "close"
STOCK_VOLUME_COL = "volume"

NEWS_DATE_COL = "published_at"
NEWS_TICKER_COL = "ticker"


def normalize_ticker(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def load_and_clean_stock_data(stock_file: str) -> pd.DataFrame:
    stock = pd.read_csv(stock_file)

    required_cols = [
        STOCK_DATE_COL,
        STOCK_TICKER_COL,
        STOCK_OPEN_COL,
        STOCK_HIGH_COL,
        STOCK_LOW_COL,
        STOCK_CLOSE_COL,
        STOCK_VOLUME_COL,
    ]
    missing = [c for c in required_cols if c not in stock.columns]
    if missing:
        raise ValueError(
            f"Missing required stock columns: {missing}\n"
            f"Available columns are: {list(stock.columns)}"
        )

    stock = stock.copy()

    stock[STOCK_TICKER_COL] = normalize_ticker(stock[STOCK_TICKER_COL])
    stock[STOCK_DATE_COL] = pd.to_datetime(stock[STOCK_DATE_COL], errors="coerce").dt.date

    stock = safe_numeric(
        stock,
        [STOCK_OPEN_COL, STOCK_HIGH_COL, STOCK_LOW_COL, STOCK_CLOSE_COL, STOCK_VOLUME_COL]
    )

    stock = stock.dropna(subset=[STOCK_DATE_COL, STOCK_TICKER_COL, STOCK_CLOSE_COL])
    stock = stock.drop_duplicates(subset=[STOCK_TICKER_COL, STOCK_DATE_COL])
    stock = stock.sort_values([STOCK_TICKER_COL, STOCK_DATE_COL]).reset_index(drop=True)

    stock["return_1d"] = stock.groupby(STOCK_TICKER_COL)[STOCK_CLOSE_COL].pct_change(1)
    stock["return_5d"] = stock.groupby(STOCK_TICKER_COL)[STOCK_CLOSE_COL].pct_change(5)
    stock["ma_5"] = stock.groupby(STOCK_TICKER_COL)[STOCK_CLOSE_COL].transform(
        lambda s: s.rolling(5).mean()
    )
    stock["ma_20"] = stock.groupby(STOCK_TICKER_COL)[STOCK_CLOSE_COL].transform(
        lambda s: s.rolling(20).mean()
    )
    stock["volatility_5d"] = stock.groupby(STOCK_TICKER_COL)["return_1d"].transform(
        lambda s: s.rolling(5).std()
    )
    stock["volume_change_1d"] = stock.groupby(STOCK_TICKER_COL)[STOCK_VOLUME_COL].pct_change(1)
    stock["hl_spread"] = (stock[STOCK_HIGH_COL] - stock[STOCK_LOW_COL]) / stock[STOCK_CLOSE_COL]

    return stock


def load_and_clean_news_data(news_file: str) -> pd.DataFrame:
    news = pd.read_csv(news_file)

    required_cols = [NEWS_DATE_COL, NEWS_TICKER_COL, "headline"]
    missing = [c for c in required_cols if c not in news.columns]
    if missing:
        raise ValueError(
            f"Missing required news columns: {missing}\n"
            f"Available columns are: {list(news.columns)}"
        )

    news = news.copy()

    news[NEWS_TICKER_COL] = normalize_ticker(news[NEWS_TICKER_COL])
    news[NEWS_DATE_COL] = pd.to_datetime(news[NEWS_DATE_COL], errors="coerce", utc=True)
    news["date"] = news[NEWS_DATE_COL].dt.date

    for col in ["headline", "summary", "author", "source", "url", "content"]:
        if col in news.columns:
            news[col] = news[col].apply(clean_text)
        else:
            news[col] = ""

    if "article_id" not in news.columns:
        news["article_id"] = np.arange(len(news)).astype(str)

    news = news.dropna(subset=[NEWS_TICKER_COL, "date"])
    news = news.drop_duplicates(subset=["article_id", NEWS_TICKER_COL])

    news["news_text"] = (
        news["headline"].fillna("") + ". " +
        news["summary"].fillna("") + ". " +
        news["content"].fillna("")
    ).str.strip()

    return news

def aggregate_news_per_ticker_day(news: pd.DataFrame) -> pd.DataFrame:
    def combine_text(series: pd.Series) -> str:
        texts = [clean_text(x) for x in series if clean_text(x)]
        return " ".join(texts)

    def unique_join(series: pd.Series) -> str:
        vals = sorted(set(v for v in series if clean_text(v)))
        return " | ".join(vals)

    grouped = (
        news.groupby([NEWS_TICKER_COL, "date"], as_index=False)
        .agg(
            news_count=("article_id", "count"),
            unique_sources=("source", lambda x: x.nunique()),
            source_list=("source", unique_join),
            author_count=("author", lambda x: x.nunique()),
            all_headlines=("headline", combine_text),
            all_summaries=("summary", combine_text),
            all_news_text=("news_text", combine_text),
            latest_news_time=(NEWS_DATE_COL, "max"),
            earliest_news_time=(NEWS_DATE_COL, "min"),
        )
    )

    grouped = grouped.rename(columns={NEWS_TICKER_COL: "ticker"})
    grouped["headline_char_len"] = grouped["all_headlines"].str.len()
    grouped["news_text_char_len"] = grouped["all_news_text"].str.len()

    return grouped

def merge_stock_and_news(stock: pd.DataFrame, news_daily: pd.DataFrame) -> pd.DataFrame:
    merged = stock.merge(
        news_daily,
        left_on=[STOCK_TICKER_COL, STOCK_DATE_COL],
        right_on=["ticker", "date"],
        how="left"
    )

    fill_zero_cols = [
        "news_count", "unique_sources", "author_count",
        "headline_char_len", "news_text_char_len"
    ]
    for col in fill_zero_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    fill_text_cols = ["source_list", "all_headlines", "all_summaries", "all_news_text"]
    for col in fill_text_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")

    merged["has_news"] = (merged["news_count"] > 0).astype(int)

    return merged

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([STOCK_TICKER_COL, STOCK_DATE_COL]).reset_index(drop=True)

    df["target_next_close"] = df.groupby(STOCK_TICKER_COL)[STOCK_CLOSE_COL].shift(-1)
    df["target_next_return"] = (
        df.groupby(STOCK_TICKER_COL)[STOCK_CLOSE_COL].shift(-1) / df[STOCK_CLOSE_COL]
    ) - 1
    df["target_up_down"] = (df["target_next_return"] > 0).astype(int)

    return df


def build_merged_dataset():
    print("Loading stock data...")
    stock = load_and_clean_stock_data(STOCK_FILE)
    print(f"Stock rows: {len(stock):,}")

    print("Loading news data...")
    news = load_and_clean_news_data(NEWS_FILE)
    print(f"News rows: {len(news):,}")

    print("Aggregating news per ticker-day...")
    news_daily = aggregate_news_per_ticker_day(news)
    print(f"Ticker-day news rows: {len(news_daily):,}")

    print("Merging stock and news...")
    merged = merge_stock_and_news(stock, news_daily)
    print(f"Merged rows: {len(merged):,}")

    print("Creating targets...")
    merged = create_targets(merged)

    merged = merged.dropna(subset=["target_next_close", "target_next_return"])

    preferred_cols = [
        STOCK_TICKER_COL,
        STOCK_DATE_COL,
        STOCK_OPEN_COL,
        STOCK_HIGH_COL,
        STOCK_LOW_COL,
        STOCK_CLOSE_COL,
        STOCK_VOLUME_COL,
        "return_1d",
        "return_5d",
        "ma_5",
        "ma_20",
        "volatility_5d",
        "volume_change_1d",
        "hl_spread",
        "news_count",
        "unique_sources",
        "author_count",
        "has_news",
        "headline_char_len",
        "news_text_char_len",
        "source_list",
        "all_headlines",
        "all_summaries",
        "all_news_text",
        "latest_news_time",
        "earliest_news_time",
        "target_next_close",
        "target_next_return",
        "target_up_down",
    ]

    existing_cols = [c for c in preferred_cols if c in merged.columns]
    remaining_cols = [c for c in merged.columns if c not in existing_cols]
    merged = merged[existing_cols + remaining_cols]

    print(f"Saving merged dataset to:\n{OUTPUT_FILE}")
    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(merged.head())
    print("\nFinal shape:", merged.shape)


if __name__ == "__main__":
    build_merged_dataset()