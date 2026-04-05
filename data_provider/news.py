import argparse
import os
import csv
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================
ENV_PATH = "/home/mjiyakho/MM-TSFlib/.env"
load_dotenv(ENV_PATH)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("API_KEY and API_SECRET are missing in your .env file.")

# =========================
# PATHS
# =========================
PROJECT_ROOT = "/home/mjiyakho/MM-TSFlib"
SYMBOL_CSV_PATH = os.path.join(PROJECT_ROOT, "nasdaq-listed-symbols.csv")
SAVE_FOLDER = os.path.join(PROJECT_ROOT, "data", "News")

OUTPUT_CSV = os.path.join(SAVE_FOLDER, "financial_news_monthly_2015_to_current.csv")
CHECKPOINT_FILE = os.path.join(SAVE_FOLDER, "financial_news_monthly_checkpoint.json")
SEEN_FILE = os.path.join(SAVE_FOLDER, "financial_news_seen_keys.txt")

# =========================
# API CONFIG
# =========================
BASE_URL = "https://data.alpaca.markets/v1beta1/news"
HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
}

START_DATE = "2015-01-01T00:00:00Z"
REQUEST_LIMIT = 50
REQUEST_TIMEOUT = 60
SYMBOL_CHUNK_SIZE = 80
MAX_WORKERS = 4
RETRY_COUNT = 3
RETRY_SLEEP_SECONDS = 2

CSV_COLUMNS = [
    "article_id",
    "ticker",
    "headline",
    "summary",
    "author",
    "published_at",
    "updated_at",
    "source",
    "url",
    "content",
]


# =========================
# HELPERS
# =========================
def ensure_folder() -> None:
    os.makedirs(SAVE_FOLDER, exist_ok=True)


def reset_backfill_state() -> None:
    for path in (OUTPUT_CSV, CHECKPOINT_FILE, SEEN_FILE):
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed: {path}")


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def load_symbols_from_csv(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Symbol CSV not found: {path}")

    symbols = set()

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        possible_cols = ["symbol", "ticker", "Symbol", "Ticker", "symbols"]

        chosen_col = next((col for col in possible_cols if col in fieldnames), None)

        if chosen_col:
            for row in reader:
                value = row.get(chosen_col)
                if value:
                    symbols.add(value.strip().upper())
        else:
            f.seek(0)
            raw_reader = csv.reader(f)
            next(raw_reader, None)
            for row in raw_reader:
                if row and row[0]:
                    symbols.add(row[0].strip().upper())

    cleaned = sorted(sym for sym in symbols if sym and sym not in {"N/A", "NULL", "NONE"})
    return cleaned


def chunked(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def make_key(article_id: str, ticker: str) -> str:
    return f"{article_id}::{ticker}"


def load_seen_keys() -> set[str]:
    if not os.path.exists(SEEN_FILE):
        return set()

    with open(SEEN_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_seen_keys(keys: list[str]) -> None:
    if not keys:
        return

    with open(SEEN_FILE, "a", encoding="utf-8") as f:
        for key in keys:
            f.write(key + "\n")


def ensure_output_csv() -> None:
    if os.path.exists(OUTPUT_CSV):
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_rows(rows: list[dict]) -> None:
    if not rows:
        return

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerows(rows)


def load_checkpoint() -> dict:
    if not os.path.exists(CHECKPOINT_FILE):
        return {"last_completed_window_end": None}

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(last_completed_window_end: str) -> None:
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_completed_window_end": last_completed_window_end}, f, indent=2)


def month_windows(start_iso: str, end_iso: str) -> list[tuple[str, str]]:
    start_dt = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    windows = []
    current = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)

    while current < end_dt:
        if current.month == 12:
            next_month = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_month = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)

        win_start = max(current, start_dt)
        win_end = min(next_month, end_dt)

        windows.append((
            win_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            win_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ))
        current = next_month

    return windows


def article_to_rows(article: dict, allowed_symbols: set[str]) -> list[dict]:
    article_id = normalize_text(article.get("id"))
    headline = normalize_text(article.get("headline"))
    summary = normalize_text(article.get("summary"))
    author = normalize_text(article.get("author"))
    published_at = normalize_text(article.get("created_at"))
    updated_at = normalize_text(article.get("updated_at"))
    source = normalize_text(article.get("source"))
    url = normalize_text(article.get("url"))
    content = normalize_text(article.get("content"))

    symbols = article.get("symbols", [])
    if not isinstance(symbols, list):
        symbols = [symbols] if symbols else []

    rows = []
    for sym in symbols:
        ticker = normalize_text(sym).upper()
        if ticker in allowed_symbols:
            rows.append({
                "article_id": article_id,
                "ticker": ticker,
                "headline": headline,
                "summary": summary,
                "author": author,
                "published_at": published_at,
                "updated_at": updated_at,
                "source": source,
                "url": url,
                "content": content,
            })

    return rows


# =========================
# API FETCH
# =========================
def fetch_news_page(session: requests.Session, symbols: list[str], start_iso: str, end_iso: str, page_token: str | None = None) -> dict:
    params = {
        "symbols": ",".join(symbols),
        "start": start_iso,
        "end": end_iso,
        "limit": REQUEST_LIMIT,
        "sort": "asc",
    }

    if page_token:
        params["page_token"] = page_token

    last_error = None

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            response = session.get(BASE_URL, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                return response.json()

            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = RuntimeError(f"Retryable error {response.status_code}: {response.text}")
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
                continue

            raise RuntimeError(f"Request failed: {response.status_code}\n{response.text}")

        except requests.RequestException as e:
            last_error = e
            time.sleep(RETRY_SLEEP_SECONDS * attempt)

    raise RuntimeError(f"Failed after retries: {last_error}")


def fetch_all_news_for_chunk(symbol_chunk: list[str], start_iso: str, end_iso: str) -> tuple[list[dict], int]:
    session = requests.Session()
    articles = []
    page_token = None
    page_count = 0

    while True:
        data = fetch_news_page(session, symbol_chunk, start_iso, end_iso, page_token)
        batch = data.get("news", [])
        articles.extend(batch)
        page_count += 1

        page_token = data.get("next_page_token")
        if not page_token:
            break

    return articles, page_count


def process_chunk(symbol_chunk: list[str], start_iso: str, end_iso: str) -> dict:
    allowed_symbols = set(symbol_chunk)
    articles, page_count = fetch_all_news_for_chunk(symbol_chunk, start_iso, end_iso)

    rows = []
    for article in articles:
        rows.extend(article_to_rows(article, allowed_symbols))

    return {
        "symbols": len(symbol_chunk),
        "articles": len(articles),
        "pages": page_count,
        "rows": rows,
    }


# =========================
# MAIN
# =========================
def run_optimized_backfill(from_scratch: bool = False) -> None:
    ensure_folder()

    if from_scratch:
        print("Starting from scratch. Clearing output, checkpoint, and seen keys...")
        reset_backfill_state()

    ensure_output_csv()

    symbols = load_symbols_from_csv(SYMBOL_CSV_PATH)
    if not symbols:
        raise ValueError("No symbols found in the symbol CSV.")

    print(f"Loaded {len(symbols)} symbols from:\n{SYMBOL_CSV_PATH}")

    checkpoint = load_checkpoint()
    seen_keys = load_seen_keys()

    start_iso = checkpoint["last_completed_window_end"] or START_DATE
    end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    windows = month_windows(start_iso, end_iso)
    if not windows:
        print("Nothing left to fetch.")
        return

    print(f"Monthly windows to fetch: {len(windows)}")
    print(f"Using {MAX_WORKERS} parallel workers")

    symbol_chunks = list(chunked(symbols, SYMBOL_CHUNK_SIZE))
    print(f"Symbol chunks: {len(symbol_chunks)}")

    for idx, (win_start, win_end) in enumerate(windows, start=1):
        print(f"\n[{idx}/{len(windows)}] Window: {win_start} -> {win_end}")

        all_new_rows = []
        all_new_keys = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_chunk, chunk, win_start, win_end): i
                for i, chunk in enumerate(symbol_chunks, start=1)
            }

            for future in as_completed(futures):
                chunk_no = futures[future]

                try:
                    result = future.result()
                    rows = result["rows"]

                    deduped_rows = []
                    deduped_keys = []

                    for row in rows:
                        key = make_key(row["article_id"], row["ticker"])
                        if key not in seen_keys:
                            seen_keys.add(key)
                            deduped_rows.append(row)
                            deduped_keys.append(key)

                    all_new_rows.extend(deduped_rows)
                    all_new_keys.extend(deduped_keys)

                    print(
                        f"  Chunk {chunk_no}: "
                        f"symbols={result['symbols']}, "
                        f"pages={result['pages']}, "
                        f"articles={result['articles']}, "
                        f"new_rows={len(deduped_rows)}"
                    )

                except Exception as e:
                    print(f"  Chunk {chunk_no} failed: {e}")

        append_rows(all_new_rows)
        append_seen_keys(all_new_keys)
        save_checkpoint(win_end)

        print(f"Window complete. Rows written: {len(all_new_rows)}")

    print("\nDone.")
    print(f"Saved file:\n{OUTPUT_CSV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Alpaca news data by monthly windows.")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Delete output/checkpoint/seen files and backfill again from START_DATE.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_optimized_backfill(from_scratch=args.from_scratch)
