import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ======================================
# Load API KEY from .env automatically
# ======================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

API_KEY = os.getenv("NEWSAPI_KEY")  # <-- Make sure your .env has: NEWSAPI_KEY=xxxx

if not API_KEY:
    raise ValueError("❌ NEWSAPI_KEY missing in .env file!")

# ======================================
# Configuration
# ======================================
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "News")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "newsapi_recent_news.csv")

# Fetch last 30 days (fastest possible free range)
FROM_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

NEWS_URL = "https://newsapi.org/v2/everything"


def fetch_news_for_ticker(ticker: str):
    """Fetch up to 100 recent articles for a ticker."""
    params = {
        "q": ticker,
        "from": FROM_DATE,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": API_KEY,
    }

    print(f"Fetching news for: {ticker}")

    response = requests.get(NEWS_URL, params=params)
    data = response.json()

    articles = data.get("articles", [])
    rows = []

    for a in articles:
        rows.append({
            "ticker": ticker,
            "date": a["publishedAt"][:10],
            "headline": a["title"],
            "summary": a["description"],
            "source": a["source"]["name"]
        })

    return rows


def main():
    all_news = []

    for t in TICKERS:
        rows = fetch_news_for_ticker(t)
        all_news.extend(rows)

    df = pd.DataFrame(all_news)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n=====================================")
    print("✅ NEWS FETCH COMPLETE")
    print(f"📄 Saved to: {OUTPUT_FILE}")
    print(f"📰 Total articles: {len(df)}")
    print("=====================================")
    print(df.head())


if __name__ == "__main__":
    main()
