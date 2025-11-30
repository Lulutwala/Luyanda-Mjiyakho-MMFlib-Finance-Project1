import yfinance as yf
import pandas as pd
import os
import time 

def generate_stock_csv(ticker, start_date, end_date, output_filename, lagged_periods=[1, 7, 30]):
    """
    Fetches historical stock data, calculates basic time-series features (lags), and saves to CSV.
    """
    print(f"Fetching data for {ticker}...")

    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        time.sleep(5) 
        return

    if stock_data.empty:
        print(f"Error: Could not retrieve any data for {ticker}. Check the ticker symbol and dates.")
        return
    df = stock_data[['Adj Close', 'Volume']].copy()
    df.index.name = 'Month' 

    df.rename(columns={'Adj Close': 'Stock_Price', 'Volume': 'Stock_Vol'}, inplace=True)

    print("Calculating lagged features...")
    
    for period in lagged_periods:
        df[f'his_avg_{period}d'] = df['Stock_Price'].shift(1).rolling(window=period).mean()

        df[f'his_std_{period}d'] = df['Stock_Price'].shift(1).rolling(window=period).std()


    df.dropna(inplace=True)


    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', output_filename)
    df.to_csv(file_path)

    print(f"\nSuccessfully generated and saved data to: {file_path}")
    print(f"Final DataFrame Shape: {df.shape}")




TICKER_SYMBOL = "MSFT" 
START_DATE = "2015-01-01"
END_DATE = "2025-11-30"  
OUTPUT_FILE = f"{TICKER_SYMBOL}_stock_data.csv"

generate_stock_csv(TICKER_SYMBOL, START_DATE, END_DATE, OUTPUT_FILE)