"""
Data preparation for ETTh1 (Electricity Transformer Temperature - hourly).
Handles loading, cleaning, and chronological splitting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_etth1(data_dir='data/raw'):
    """
    Loads the electricity transformer temperature (ETTh1) data from a CSV file.
    """
    data_path = Path(data_dir)
    file_path = data_path / 'ETTh1.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {file_path}. "
            "Please run 'python scripts/download_data.py' first to fetch the data."
        )
    
    df = pd.read_csv(file_path)
    
    # Organize data by date so it's in order
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.sort_index()
    
    return df

def clean_data(df):
    """
    Cleans the data by removing duplicates and filling in any missing spots.
    """
    if df.index.duplicated().any():
        print(f"Warning: Found some repeated time stamps, keeping only the first one.")
        df = df[~df.index.duplicated(keep='first')]
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Notice: Some data points were missing. Filling them in with the last known values.")
        df = df.ffill().bfill()
    
    return df

def chronological_split(df, train_frac=0.6, val_frac=0.2):
    """
    Splits the data into training, validation, and test sets in order.
    We don't shuffle because timing matters in this data.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_idx = df.index[:train_end]
    val_idx = df.index[train_end:val_end]
    test_idx = df.index[val_end:]
    
    print(f"Dividing data into three parts:")
    print(f"  Training: {train_idx[0]} to {train_idx[-1]}")
    print(f"  Check (Val): {val_idx[0]} to {val_idx[-1]}")
    print(f"  TFinal test: {test_idx[0]} to {test_idx[-1]}")
    
    return train_idx, val_idx, test_idx

def plot_target_and_splits(df, train_idx, val_idx, test_idx, target_col='OT', save_path=None):
    """
    Creates a graph showing how the oil temperature is divided into training and testing sections.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(df.loc[train_idx].index, df.loc[train_idx, target_col], 
            label='Training Data', alpha=0.7, linewidth=0.8)
    ax.plot(df.loc[val_idx].index, df.loc[val_idx, target_col], 
            label='Check (Val) Data', alpha=0.7, linewidth=0.8)
    ax.plot(df.loc[test_idx].index, df.loc[test_idx, target_col], 
            label='Final Test Data', alpha=0.7, linewidth=0.8)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Oil Temperature')
    ax.set_title('Visualizing how we split the data')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved graph to {save_path}")
    
    return fig

def add_time_features(df):
    """
    Adds time information like the hour and day of the week to help the model 
    understand daily and weekly cycles.
    """
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return df


if __name__ == '__main__':
    df = load_etth1()
    print(f"\nLoaded {len(df)} hourly observations")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    df = clean_data(df)
    train_idx, val_idx, test_idx = chronological_split(df)
    
    print(f"\nTarget (OT) statistics:")
    print(df['OT'].describe())
