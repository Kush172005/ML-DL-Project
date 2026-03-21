"""
Download historical stock data for the Hybrid Temporal Forecaster project.
Uses yfinance to fetch SPY (S&P 500 ETF) and VIX data.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

def download_stock_data(ticker='SPY', start='2010-01-01', end='2024-12-31'):
    """Download adjusted close price data for a given ticker."""
    print(f"Downloading {ticker} data from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

def main():
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download SPY (primary target)
    spy_data = download_stock_data('SPY', start='2010-01-01', end='2024-12-31')
    spy_path = data_dir / 'SPY_daily.csv'
    spy_data.to_csv(spy_path)
    print(f"Saved SPY data to {spy_path} ({len(spy_data)} rows)")
    
    # Download VIX (volatility context feature)
    vix_data = download_stock_data('^VIX', start='2010-01-01', end='2024-12-31')
    vix_path = data_dir / 'VIX_daily.csv'
    vix_data.to_csv(vix_path)
    print(f"Saved VIX data to {vix_path} ({len(vix_data)} rows)")
    
    print("\nData download complete!")

if __name__ == '__main__':
    main()
