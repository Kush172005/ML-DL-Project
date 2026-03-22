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
    Load ETTh1 dataset.
    
    Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    Target: OT (Oil Temperature)
    Covariates: HUFL, HULL, MUFL, MULL (High/Medium/Low Usage/Load)
    """
    data_path = Path(data_dir)
    df = pd.read_csv(data_path / 'ETTh1.csv')
    
    # Parse date and set as index
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.sort_index()
    
    return df

def clean_data(df):
    """Handle missing values and check for issues."""
    # Check for duplicates
    if df.index.duplicated().any():
        print(f"Warning: {df.index.duplicated().sum()} duplicate timestamps, keeping first")
        df = df[~df.index.duplicated(keep='first')]
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        print("Forward-filling missing values...")
        df = df.ffill().bfill()
    
    return df

def chronological_split(df, train_frac=0.6, val_frac=0.2):
    """
    Split data chronologically for time series: train / validation / test.
    No shuffling to prevent temporal leakage.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_idx = df.index[:train_end]
    val_idx = df.index[train_end:val_end]
    test_idx = df.index[val_end:]
    
    print(f"Chronological splits:")
    print(f"  Train: {train_idx[0]} to {train_idx[-1]} ({len(train_idx)} hours)")
    print(f"  Val:   {val_idx[0]} to {val_idx[-1]} ({len(val_idx)} hours)")
    print(f"  Test:  {test_idx[0]} to {test_idx[-1]} ({len(test_idx)} hours)")
    
    return train_idx, val_idx, test_idx

def plot_target_and_splits(df, train_idx, val_idx, test_idx, target_col='OT', save_path=None):
    """Visualize target variable with train/val/test regions."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(df.loc[train_idx].index, df.loc[train_idx, target_col], 
            label='Train', alpha=0.7, linewidth=0.8)
    ax.plot(df.loc[val_idx].index, df.loc[val_idx, target_col], 
            label='Validation', alpha=0.7, linewidth=0.8)
    ax.plot(df.loc[test_idx].index, df.loc[test_idx, target_col], 
            label='Test', alpha=0.7, linewidth=0.8)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{target_col} (Oil Temperature)')
    ax.set_title('ETTh1: Chronological Train/Val/Test Split')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved split visualization to {save_path}")
    
    return fig

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
