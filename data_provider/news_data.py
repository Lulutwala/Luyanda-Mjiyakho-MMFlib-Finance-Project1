import requests
import pandas as pd
import time
import os
from typing import List, Dict, Any
from dotenv import load_dotenv


# =========================================================
# LOAD .env FROM PROJECT ROOT (one directory above script)
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


# =========================================================
# CONFIGURATION
# =========================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "News"
)

OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "mmflib_raw_news_data.json")

BASE_URL = "https://www.alphavantage.co/query"
WAIT_TIME_SECONDS = 12
MAX_ARTICLES_PER_CALL = 200
MAX_RETRIES = 2


# =========================================================
# FETCH S&P 500 TICKERS
# =========================================================

def get_sp500_tickers() -> List[str]:
    """
    Fetch S&P 500 tickers from GitHub. Falls back to a short list on failure.
    """
    print("Fetching S&P 500 tickers...")

    url = "https://raw.githubusercontent.com/datasets/s-p-500-companies/master/data/constituents.csv"

    try:
        df = pd.read_csv(url)
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"✔ Loaded {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"❌ Failed to load tickers: {e}")
        print("⚠ Using fallback list.")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "V"]


# =========================================================
# FETCH NEWS FOR SINGLE TICKER WITH RETRIES
# =========================================================

def fetch_news_for_ticker(ticker: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch news for a single ticker using Alpha Vantage.
    Includes retry and rate-limit handling.
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "sort": "LATEST",
        "limit": MAX_ARTICLES_PER_CALL,
        "apikey": api_key,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=20)
            data = response.json()

            # Rate-limit reached
            if "Note" in data:
                wait = 60 * attempt  # exponential backoff
                print(f"⏳ Rate limit hit for {ticker}. Waiting {wait}s...")
                time.sleep(wait)
                continue

            # API error
            if "Error Message" in data:
                print(f"❌ API error for {ticker}: {data['Error Message']}")
                return []

            return data.get("feed", [])

        except requests.exceptions.RequestException as e:
            print(f"🌐 Network error for {ticker}: {e}")

        if attempt == MAX_RETRIES:
            print(f"❌ Failed after {MAX_RETRIES} attempts for {ticker}.")
            return []

    return []


# =========================================================
# MAIN NEWS COLLECTION LOOP
# =========================================================

def fetch_company_news(api_key: str, tickers: List[str]) -> pd.DataFrame:
    """
    Fetch news for a list of tickers.
    Only first 5 tickers are fetched due to Alpha Vantage free tier limits.
    """
    tickers_to_fetch = tickers[:5]
    print(f"Starting news fetch for {len(tickers_to_fetch)} tickers.\n")

    all_items = []

    for i, ticker in enumerate(tickers_to_fetch):
        print(f"=== [{i + 1}/{len(tickers_to_fetch)}] Fetching: {ticker} ===")

        feed = fetch_news_for_ticker(ticker, api_key)
        print(f"→ Retrieved {len(feed)} articles.")

        for item in feed:
            topics = item.get("topics", [])
            topic = topics[0].get("topic") if topics else "N/A"

            all_items.append({
                "ticker_main": ticker,
                "time_published": item.get("time_published"),
                "headline": item.get("title"),
                "summary": item.get("summary"),
                "source": item.get("source"),
                "topic": topic,
                "overall_sentiment": item.get("overall_sentiment_label", "N/A"),
            })

        if i < len(tickers_to_fetch) - 1:
            print(f"⏳ Waiting {WAIT_TIME_SECONDS}s for cooldown...\n")
            time.sleep(WAIT_TIME_SECONDS)

    return pd.DataFrame(all_items)


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():
    # Check API key
    if not API_KEY:
        raise ValueError(
            "Missing API key. Ensure ALPHA_VANTAGE_API_KEY is set in your .env file."
        )

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = get_sp500_tickers()
    df_news = fetch_company_news(API_KEY, tickers)

    if df_news.empty:
        print("\n❌ No news data retrieved.")
        return

    df_news.to_json(OUTPUT_FILENAME, orient="records", lines=True, date_format="iso")
    print(f"\n✅ SUCCESS: News saved → {OUTPUT_FILENAME}")
    print(f"Total articles fetched: {len(df_news)}")
    print("✔ Use 'headline' and 'summary' for NLP / GPT model training.")


if __name__ == "__main__":
    main()
