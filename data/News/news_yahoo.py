import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("NEWSDATA_API_KEY")

BASE_DIR = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1"
NEWS_DIR = os.path.join(BASE_DIR, "data", "News")
OUTPUT_FILE = os.path.join(NEWS_DIR, "news_newsdatalo.csv")

os.makedirs(NEWS_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

START_YEAR = 2015
END_YEAR = 2025

# ============================================================
# FETCH NEWS FOR ONE TICKER
# ============================================================

def get_news_for_ticker(ticker):
    all_articles = []

    print(f"\n📌 Fetching NewsData.io news for {ticker}")

    for year in range(START_YEAR, END_YEAR + 1):

        url = (
            "https://newsdata.io/api/1/news?"
            f"apikey={API_KEY}&q={ticker}&language=en"
            f"&from_date={year}-01-01&to_date={year}-12-31"
        )

        try:
            response = requests.get(url, timeout=10)
            data = response.json()
        except Exception as e:
            print(f"   ❌ Error fetching year {year}: {e}")
            continue

        articles = data.get("results", [])

        # ---- FIX: skip invalid entries (strings, None, errors) ----
        clean_articles = [a for a in articles if isinstance(a, dict)]

        print(f"   Year {year}: {len(clean_articles)} valid articles")

        # Extract fields
        for a in clean_articles:
            pub_date = a.get("pubDate", "")
            pub_date = pub_date[:10] if isinstance(pub_date, str) else ""

            entry = {
                "ticker": ticker,
                "date": pub_date,
                "headline": a.get("title", "") or "",
                "summary": a.get("description", "") or "",
                "source": a.get("source_id", "") or ""
            }

            all_articles.append(entry)

        time.sleep(1)  # free tier API rate limit

    print(f"✔ Total collected for {ticker}: {len(all_articles)}")
    return all_articles

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    all_news = []

    for ticker in TICKERS:
        news_data = get_news_for_ticker(ticker)
        all_news.extend(news_data)

    if len(all_news) == 0:
        print("\n❌ ERROR: NO NEWS FETCHED. CHECK API KEY.")
        return

    df = pd.DataFrame(all_news)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n======================================")
    print("✅ NEWS FETCHING COMPLETE")
    print("Saved to:", OUTPUT_FILE)
    print("Total rows:", len(df))
    print("======================================")
    print(df.head())

if __name__ == "__main__":
    main()
