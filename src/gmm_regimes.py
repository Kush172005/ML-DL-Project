"""
GMM-based regime detection using causal features.
Provides regime probabilities as additional context for TFT.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def create_gmm_features(residuals, window=24):
    """
    Create causal features for GMM from residuals.
    
    Features:
    - Lagged residual (t-1)
    - Rolling std of residuals (volatility proxy, past-only window)
    - Rolling mean of absolute residuals
    
    Args:
        residuals: Array or Series of residuals from linear model
        window: Window size for rolling statistics
    
    Returns:
        DataFrame with GMM features (NaN rows dropped)
    """
    if isinstance(residuals, np.ndarray):
        residuals = pd.Series(residuals)
    
    df = pd.DataFrame()
    
    # 1. THE PAST ERROR: What was the exact error 1 hour ago? (Shift prevents looking into the future)
    df['resid_lag1'] = residuals.shift(1)
    
    # 2. THE CHAOS METER: How volatile/unstable were the errors over the last 24 hours? 
    df['resid_vol'] = residuals.shift(1).rolling(window).std()
    
    # 3. THE ERROR SIZE: How "big" were the errors on average over the last 24 hours?
    df['resid_abs_mean'] = residuals.abs().shift(1).rolling(window).mean()
    
    # Drop NaN from rolling windows
    df = df.dropna()
    
    return df

class GMMRegimeDetector:
    """
    Gaussian Mixture Model for regime detection.
    Fits on training features only, then predicts regime probabilities.
    """
    
    def __init__(self, n_components=2, random_state=42):
        """
        Args:
            n_components: Number of regimes (2 = high/low volatility)
            random_state: For reproducibility
        """
        self.n_components = n_components
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state,
            max_iter=100
        )
        self.feature_names = None
    
    def fit(self, X_train):
        """
        Fit GMM on training features.
        
        Args:
            X_train: Training feature matrix (n_samples, n_features)
        """
        print(f"Fitting GMM with {self.n_components} components...")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")
        
        self.gmm.fit(X_train)
        
        print(f"  Converged: {self.gmm.converged_}")
        print(f"  BIC: {self.gmm.bic(X_train):.2f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict regime probabilities for each time point.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probabilities (n_samples, n_components)
        """
        if self.gmm is None:
            raise ValueError("GMM must be fitted first")
        
        return self.gmm.predict_proba(X)
    
    def predict(self, X):
        """Predict hard regime labels."""
        return self.gmm.predict(X)

def compute_residuals(y_true, y_fitted):
    """
    Compute residuals from fitted model.
    
    Args:
        y_true: Actual values
        y_fitted: Fitted values from statistical model
    
    Returns:
        Residuals (y_true - y_fitted)
    """
    if isinstance(y_true, pd.Series) and isinstance(y_fitted, pd.Series):
        # Align indices
        common_idx = y_true.index.intersection(y_fitted.index)
        return (y_true.loc[common_idx] - y_fitted.loc[common_idx]).values
    else:
        return np.array(y_true) - np.array(y_fitted)

if __name__ == '__main__':
    from data_prep import load_etth1, clean_data, chronological_split
    from baselines import SARIMAXForecaster
    
    df = load_etth1()
    df = clean_data(df)
    train_idx, val_idx, test_idx = chronological_split(df)
    
    # Fit SARIMAX
    y_train = df.loc[train_idx, 'OT']
    forecaster = SARIMAXForecaster()
    forecaster.fit(y_train)
    
    # Get residuals
    fitted = forecaster.get_fitted_values()
    residuals = compute_residuals(y_train, fitted)
    print(f"\nResiduals: {len(residuals)} points")
    print(f"Residual std: {np.std(residuals):.4f}")
    
    # Create GMM features
    gmm_features = create_gmm_features(residuals, window=24)
    print(f"\nGMM features: {gmm_features.shape}")
    print(gmm_features.head())
    
    # Fit GMM
    gmm = GMMRegimeDetector(n_components=2)
    gmm.fit(gmm_features.values)
    
    # Predict regimes
    probs = gmm.predict_proba(gmm_features.values[:100])
    print(f"\nRegime probabilities (first 5):")
    print(probs[:5])
