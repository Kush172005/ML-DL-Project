"""
Feature engineering: causal lag and rolling features for time series.
All features use only past information (no future leakage).
"""

import pandas as pd
import numpy as np

def create_lag_features(df, target_col='log_return', lags=[1, 2, 3, 5, 10]):
    """Create lagged features from target column."""
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col='log_return', windows=[5, 10, 20]):
    """
    Create rolling statistics (mean, std) using only past data.
    Window includes current point, but for prediction at t, we shift by 1.
    """
    df = df.copy()
    for window in windows:
        # Rolling mean and std - shift to ensure causality
        df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window).std()
    return df

def create_calendar_features(df):
    """Create calendar-based features (day of week, month, etc.)."""
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    return df

def create_vix_features(df, lags=[1, 5]):
    """Create lagged VIX features as volatility context."""
    df = df.copy()
    if 'vix' in df.columns:
        for lag in lags:
            df[f'vix_lag_{lag}'] = df['vix'].shift(lag)
    return df

def create_interaction_features(df):
    """
    Simple stress × momentum proxy: yesterday's return times lagged VIX.
    Lets the tree model condition on "recent move in a fearful market" without hand-tuned thresholds.
    """
    df = df.copy()
    if 'lag_1' in df.columns and 'vix_lag_1' in df.columns:
        df['lag1_x_vix1'] = df['lag_1'] * df['vix_lag_1']
    return df

def create_all_features(df, target_col='log_return', 
                       lags=[1, 2, 3, 5, 10],
                       rolling_windows=[5, 10, 20],
                       include_calendar=True,
                       include_vix=True):
    """
    Create all features for ML models.
    Returns dataframe with features and drops rows with NaN.
    """
    df = df.copy()
    
    df = create_lag_features(df, target_col, lags)
    df = create_rolling_features(df, target_col, rolling_windows)
    
    if include_calendar:
        df = create_calendar_features(df)
    
    if include_vix and 'vix' in df.columns:
        df = create_vix_features(df)

    df = create_interaction_features(df)
    
    # Drop rows with NaN (from lagging/rolling)
    initial_len = len(df)
    df = df.dropna()
    print(f"Feature engineering: {initial_len} -> {len(df)} rows (dropped {initial_len - len(df)} due to lag/rolling)")
    
    return df

def get_feature_columns(df, exclude_cols=['price', 'log_return', 'vix', 'realized_vol', 'regime']):
    """Get list of feature column names."""
    return [col for col in df.columns if col not in exclude_cols]

def prepare_ml_data(df, feature_cols, target_col='log_return', horizon=5):
    """
    Prepare X (features) and y (multi-step targets) for ML models.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        target_col: Target column name
        horizon: Number of steps ahead to predict
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target matrix (n_samples, horizon)
    """
    X = df[feature_cols].values
    
    # Create multi-step targets
    y = []
    for i in range(len(df)):
        if i + horizon <= len(df):
            y.append(df[target_col].iloc[i:i+horizon].values)
        else:
            y.append(np.full(horizon, np.nan))
    
    y = np.array(y)
    
    # Remove rows where target contains NaN
    valid_mask = ~np.isnan(y).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y

if __name__ == '__main__':
    from data_prep import load_data, clean_data, add_returns
    
    df = load_data()
    df = clean_data(df)
    df = add_returns(df)
    
    df_feat = create_all_features(df)
    print(f"\nFeatures created: {len(df_feat)} rows")
    print(f"Feature columns: {get_feature_columns(df_feat)}")
    
    X, y = prepare_ml_data(df_feat, get_feature_columns(df_feat), horizon=5)
    print(f"\nML data prepared: X shape {X.shape}, y shape {y.shape}")
