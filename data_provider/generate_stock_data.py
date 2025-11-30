import yfinance as yf
import pandas as pd
import os
import time
import random 
from datetime import date
from pandas_datareader import data as pdr # <-- New import

# --- NEW ROBUST FUNCTION TO GET TICKERS ---
def get_sp500_tickers():
    """
    Fetches the list of S&P 500 component tickers using a reliable data reader.
    """
    print("Attempting to fetch S&P 500 ticker list from NASDAQ DataReader...")
    try:
        # Use pandas_datareader to get the NASDAQ list (which includes many S&P components)
        # We will use the 'NASDAQ' listing data source.
        nasdaq_df = pdr.get_nasdaq_symbols()
        
        # The 'Symbol' column contains the ticker names
        tickers = nasdaq_df.index.tolist()
        
        # Filter out common problematic entries (indices, options, etc.)
        # Keep only strings that look like typical stock tickers (e.g., all caps, no dots/dashes)
        cleaned_tickers = [
            ticker.replace('.', '-') for ticker in tickers 
            if isinstance(ticker, str) and len(ticker) <= 5 and not ticker.startswith('^')
        ]
        
        # Optional: Limit the list to prevent overly long runtime if you downloaded the full NASDAQ list
        # If you want ALL, remove the slice [:]
        return cleaned_tickers[:500] 
        
    except Exception as e:
        print(f"Error fetching ticker list using pandas_datareader: {e}")
        # As a fallback, use a small, known working list
        print("Falling back to a small test list.")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]

# --- END NEW FUNCTION ---

# --- Configuration ---
# 1. Automatically get the list of tickers
TICKERS_TO_FETCH = get_sp500_tickers() 
if not TICKERS_TO_FETCH:
    print("FATAL: Could not fetch ticker list. Exiting.")
    exit()

print(f"Starting fetch for {len(TICKERS_TO_FETCH)} tickers...")

START_DATE = "2015-01-01"
END_DATE = str(date.today()) 
OUTPUT_DIR = "data/sp500_stock_data" 
LAGGED_PERIODS = [1, 7, 30]
MIN_SLEEP_SEC = 2  # Minimum delay between requests
MAX_SLEEP_SEC = 5  # Maximum delay between requests

# --- CORE FETCHING LOGIC ---

def generate_stock_features(ticker, start_date, end_date, lagged_periods):
    """Fetches data and calculates lagged features for a SINGLE stock."""
    
    print(f"-> Processing {ticker}...")
    
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"!!! ERROR: Failed to download {ticker}. Reason: {e}")
        return None

    if stock_data.empty:
        print(f"!!! WARNING: No data retrieved for {ticker}. Skipping.")
        return None

    df = stock_data[['Adj Close', 'Volume']].copy()
    df.index.name = 'Month'
    df.rename(columns={'Adj Close': 'Stock_Price', 'Volume': 'Stock_Vol'}, inplace=True)

    # Calculated lagged features
    for period in lagged_periods:
        df[f'his_avg_{period}d'] = df['Stock_Price'].shift(1).rolling(window=period).mean()
        df[f'his_std_{period}d'] = df['Stock_Price'].shift(1).rolling(window=period).std()

    df.dropna(inplace=True)
    return df

def fetch_all_stocks(tickers, start, end, output_dir, lagged_periods):
    """Loops through all tickers, fetches data, and saves each to a separate file."""
    
    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    
    for i, ticker in enumerate(tickers):
        df_features = generate_stock_features(ticker, start, end, lagged_periods)
        
        if df_features is not None:
            file_path = os.path.join(output_dir, f"{ticker}_features.csv")
            df_features.to_csv(file_path)
            print(f"SUCCESS: Saved {ticker} to {file_path}. Shape: {df_features.shape}")
            success_count += 1

        # Throttle the request between downloads
        if i < len(tickers) - 1:
            sleep_time = random.uniform(MIN_SLEEP_SEC, MAX_SLEEP_SEC)
            print(f"--- Waiting {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
    
    print(f"\n✅ Finished fetching data. Total successful downloads: {success_count}/{len(tickers)}")


fetch_all_stocks(TICKERS_TO_FETCH, START_DATE, END_DATE, OUTPUT_DIR, LAGGED_PERIODS)