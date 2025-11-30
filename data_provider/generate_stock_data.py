"""
Super-optimized stock + feature generator for MMFlib training
- Batch-downloads tickers with yfinance (fast)
- Normalizes outputs safely (handles MultiIndex)
- Computes technical indicators and historical windows
- Produces CSV with exact MMFlib header order first, then stock feature columns
- Saves CSV directly to the specified path
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List
import os

# --- CONFIGURATION ---
OUTPUT_DIR = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1\data\Economy"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, 'mmflib_stock_features.csv')

START_DATE = '2015-01-01'
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')
BATCH_THREADS = True
PRIOR_HISTORY_WINDOW = 30

#to match the columns in the existing economy data#
MMFLIB_COLUMNS = [
    "Month","Exports","Imports","OT","start_date","date","end_date",
    "his_avg_1","his_std_1","his_avg_2","his_std_2","his_avg_3","his_std_3",
    "his_avg_4","his_std_4","his_avg_5","his_std_5","his_avg_6","his_std_6",
    "his_avg_7","his_std_7","prior_history_avg","prior_history_std",
    "Final_Search_2","Final_Search_4","Final_Search_6"
]

# --- HELPERS ---
def get_sp500_tickers() -> List[str]:
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except Exception:
        return ["AAPL","MSFT","GOOGL","AMZN","NVDA","JPM","V"]

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

# --- MAIN PROCEDURE ---
def batch_download_and_build(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Batch downloading {len(tickers)} tickers ({start_date} → {end_date}) ...")
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        group_by="ticker",
        threads=BATCH_THREADS,
        progress=False
    )
    if raw.empty:
        raise RuntimeError("Batch download returned empty DataFrame.")

    all_frames = []
    tickers_in_df = raw.columns.levels[0] if isinstance(raw.columns, pd.MultiIndex) else [tickers[0]]

    for t in tickers_in_df:
        try:
            tdf = raw[t].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except Exception:
            continue
        if tdf.dropna(how="all").empty:
            continue
        tdf = tdf.reset_index()
        if 'Date' in tdf.columns:
            tdf.rename(columns={'Date':'date'}, inplace=True)
        elif 'date' not in tdf.columns:
            tdf['date'] = pd.to_datetime(tdf.iloc[:,0], errors='coerce')
        tdf['date'] = pd.to_datetime(tdf['date'], errors='coerce')
        tdf = tdf.dropna(subset=['date']).reset_index(drop=True)

        # Standardize columns
        colmap = {}
        for c in tdf.columns:
            lc = c.lower()
            if lc.startswith('open'): colmap[c]='open'
            elif lc.startswith('high'): colmap[c]='high'
            elif lc.startswith('low'): colmap[c]='low'
            elif lc.startswith('close'):
                colmap[c] = 'adj_close' if 'adj' in c.lower() else 'close'
            elif 'volume' in lc: colmap[c]='volume'
        tdf.rename(columns=colmap, inplace=True)

        tdf['close_used'] = tdf.get('adj_close', tdf.get('close'))
        if tdf['close_used'].isna().all(): continue

        for col in ['open','high','low','close','adj_close','close_used','volume']:
            if col in tdf.columns:
                tdf[col] = pd.to_numeric(tdf[col], errors='coerce')

        tdf = tdf.sort_values('date').reset_index(drop=True)
        tdf['log_return'] = np.log(tdf['close_used'] / tdf['close_used'].shift(1))
        tdf['pct_change'] = tdf['close_used'].pct_change()

        # Lags 1..7
        for lag in range(1,8):
            tdf[f'lag_{lag}'] = tdf['log_return'].shift(lag)
        # Rolling mean/std
        for w in range(1,8):
            tdf[f'rolling_mean_{w}'] = tdf['close_used'].shift(1).rolling(window=w, min_periods=1).mean()
            tdf[f'rolling_std_{w}'] = tdf['close_used'].shift(1).rolling(window=w, min_periods=1).std(ddof=0)
        # his_avg / his_std
        for n in range(1,8):
            tdf[f'his_avg_{n}'] = tdf['log_return'].shift(1).rolling(window=n, min_periods=1).mean()
            tdf[f'his_std_{n}'] = tdf['log_return'].shift(1).rolling(window=n, min_periods=1).std(ddof=0)
        tdf['prior_history_avg'] = tdf['log_return'].shift(1).rolling(PRIOR_HISTORY_WINDOW, min_periods=1).mean()
        tdf['prior_history_std'] = tdf['log_return'].shift(1).rolling(PRIOR_HISTORY_WINDOW, min_periods=1).std(ddof=0)

        # SMA, RSI, MACD, Bollinger, Volatility
        for span in (5,10,20): tdf[f'sma_{span}'] = tdf['close_used'].shift(1).rolling(span,min_periods=1).mean()
        tdf['rsi_14'] = rsi(tdf['close_used'],14)
        ema12, ema26 = ema(tdf['close_used'],12), ema(tdf['close_used'],26)
        tdf['macd'], tdf['macd_signal'] = ema12-ema26, ema(ema12-ema26,9)
        tdf['macd_hist'] = tdf['macd'] - tdf['macd_signal']
        tdf['bb_sma20'] = tdf['close_used'].shift(1).rolling(20,min_periods=1).mean()
        tdf['bb_std20'] = tdf['close_used'].shift(1).rolling(20,min_periods=1).std(ddof=0)
        tdf['bb_upper'] = tdf['bb_sma20'] + 2*tdf['bb_std20']
        tdf['bb_lower'] = tdf['bb_sma20'] - 2*tdf['bb_std20']
        tdf['volatility_20_ann'] = tdf['log_return'].shift(1).rolling(20,min_periods=1).std(ddof=0) * np.sqrt(252)

        # MMFlib placeholders
        tdf['Final_Search_2'] = np.nan
        tdf['Final_Search_4'] = np.nan
        tdf['Final_Search_6'] = np.nan
        tdf['Month'] = tdf['date'].dt.to_period('M').astype(str)
        tdf['Exports'] = np.nan
        tdf['Imports'] = np.nan
        tdf['OT'] = np.nan
        tdf['start_date'] = START_DATE
        tdf['end_date'] = END_DATE

        # Stock symbol column
        tdf['ticker'] = t

        tdf = tdf.loc[:, ~tdf.columns.duplicated()]
        all_frames.append(tdf)

    if not all_frames: raise RuntimeError("No valid ticker data.")
    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
    combined['Month'] = pd.to_datetime(combined['date']).dt.to_period('M').astype(str)

    # Ensure all MMFLIB_COLUMNS exist
    for c in MMFLIB_COLUMNS:
        if c not in combined.columns: combined[c]=np.nan

    # Build final column order
    stock_feature_cols = ['ticker','open','high','low','close','adj_close','close_used','volume','log_return','pct_change']
    extra_cols = sorted([c for c in combined.columns if c not in MMFLIB_COLUMNS + stock_feature_cols])
    final_cols = MMFLIB_COLUMNS + stock_feature_cols + extra_cols
    seen = set()
    final_cols_ordered = [c for c in final_cols if not (c in seen or seen.add(c))]
    combined = combined.reindex(columns=final_cols_ordered)

    return combined

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = get_sp500_tickers()
    df = batch_download_and_build(tickers, START_DATE, END_DATE)
    df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"✅ Saved features to {OUTPUT_FILENAME}")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()
