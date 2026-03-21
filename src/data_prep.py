"""
Data preparation module: loading, cleaning, and chronological splitting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(data_dir='data/raw'):
    """Load SPY and VIX data, merge on date."""
    data_path = Path(data_dir)
    
    spy = pd.read_csv(data_path / 'SPY_daily.csv', header=[0, 1], index_col=0, parse_dates=True)
    vix = pd.read_csv(data_path / 'VIX_daily.csv', header=[0, 1], index_col=0, parse_dates=True)
    
    # Extract the Close price columns (yfinance format)
    df = pd.DataFrame({
        'price': spy[('Close', 'SPY')],
        'vix': vix[('Close', '^VIX')]
    })
    
    df = df.sort_index()
    return df

def clean_data(df):
    """Handle missing values and duplicates."""
    # Check for duplicates
    if df.index.duplicated().any():
        print(f"Warning: {df.index.duplicated().sum()} duplicate dates found, keeping first")
        df = df[~df.index.duplicated(keep='first')]
    
    # Forward fill missing values (common for market holidays)
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"Forward-filling {missing_before} missing values")
        df = df.ffill()
    
    # Drop any remaining NaNs at the start
    df = df.dropna()
    
    return df

def add_returns(df):
    """Compute log returns for price."""
    df = df.copy()
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    return df

def chronological_split(df, train_frac=0.7, val_frac=0.15):
    """
    Split data chronologically: train / validation / test.
    Returns indices for each split.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_idx = df.index[:train_end]
    val_idx = df.index[train_end:val_end]
    test_idx = df.index[val_end:]
    
    print(f"Train: {train_idx[0].date()} to {train_idx[-1].date()} ({len(train_idx)} days)")
    print(f"Val:   {val_idx[0].date()} to {val_idx[-1].date()} ({len(val_idx)} days)")
    print(f"Test:  {test_idx[0].date()} to {test_idx[-1].date()} ({len(test_idx)} days)")
    
    return train_idx, val_idx, test_idx

def plot_splits(df, train_idx, val_idx, test_idx, save_path=None):
    """Visualize the chronological splits."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(df.loc[train_idx].index, df.loc[train_idx, 'price'], 
            label='Train', alpha=0.7)
    ax.plot(df.loc[val_idx].index, df.loc[val_idx, 'price'], 
            label='Validation', alpha=0.7)
    ax.plot(df.loc[test_idx].index, df.loc[test_idx, 'price'], 
            label='Test', alpha=0.7)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('SPY Price ($)')
    ax.set_title('Chronological Train/Val/Test Split')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved split visualization to {save_path}")
    
    return fig

def compute_regime_labels(df, window=20):
    """
    Compute realized volatility and assign regime labels (high/low vol).
    Uses quartiles on rolling volatility of returns.
    """
    df = df.copy()
    df['realized_vol'] = df['log_return'].rolling(window).std() * np.sqrt(252)
    
    # Assign regime based on quartiles
    vol_q75 = df['realized_vol'].quantile(0.75)
    df['regime'] = 'low_vol'
    df.loc[df['realized_vol'] > vol_q75, 'regime'] = 'high_vol'
    
    return df

if __name__ == '__main__':
    # Quick test
    df = load_data()
    print(f"Loaded {len(df)} rows")
    df = clean_data(df)
    df = add_returns(df)
    print(f"After cleaning: {len(df)} rows")
    print(df.head())
    
    train_idx, val_idx, test_idx = chronological_split(df)
