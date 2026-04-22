"""
GMM-based regime detection using causal features.
Provides regime probabilities as additional context for TFT.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def create_gmm_features(residuals, window=24):
    """
    Creates 'clues' for the model to understand if the errors are currently 
    stable or chaotic.
    
    We look at:
    1. The last error made (Where did we miss?)
    2. The 'Chaos Meter' (How much did the errors bounce around in the last 24h?)
    3. The 'Error Size' (How big were the misses on average?)
    """
    if isinstance(residuals, np.ndarray):
        residuals = pd.Series(residuals)
    
    df = pd.DataFrame()
    
    # 1. THE PAST ERROR: What was the exact error 1 hour ago? 
    df['resid_lag1'] = residuals.shift(1)
    
    # 2. THE CHAOS METER: How volatile/unstable were the errors over the last 24 hours? 
    df['resid_vol'] = residuals.shift(1).rolling(window).std()
    
    # 3. THE ERROR SIZE: How "big" were the errors on average over the last 24 hours?
    df['resid_abs_mean'] = residuals.abs().shift(1).rolling(window).mean()
    
    # Clean up the data
    df = df.dropna()
    
    return df

class GMMRegimeDetector:
    """
    A 'Situation Detector' that sorts the day's behavior into categories 
    (like 'High Volatility' or 'Low Volatility'). 
    It tells the AI model exactly what kind of situation it's dealing with.
    """
    
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state,
            max_iter=100
        )
        self.feature_names = None
    
    def fit(self, X_train):
        """Trains the detector to recognize different types of days."""
        print(f"Setting up the Situation Detector ({self.n_components} categories)...")
        self.gmm.fit(X_train)
        return self
    
    def predict_proba(self, X):
        """Predicts the probability of being in each situation."""
        if self.gmm is None:
            raise ValueError("Detector must be set up first")
        return self.gmm.predict_proba(X)
    
    def predict(self, X):
        """Gives a definitive label for the current situation."""
        return self.gmm.predict(X)

def compute_residuals(y_true, y_fitted):
    """
    Calculates the 'miss' (error) between the actual temperature and 
    what our base model thought it would be.
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
