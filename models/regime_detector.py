from sklearn.mixture import GaussianMixture
import numpy as np

class MLRegimeDetector:
    """
    Machine Learning Component: Gaussian Mixture Model (GMM).
    Purpose: Categorizes the high-dimensional DL features into interpretable 
    operational regimes (States).
    """
    def __init__(self, n_regimes=3):
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.is_fitted = False

    def fit(self, features):
        """Fit GMM on features extracted by the DL model."""
        self.gmm.fit(features)
        self.is_fitted = True

    def predict_regimes(self, features):
        """Predict regime probabilities for given features."""
        if not self.is_fitted:
            raise ValueError("GMM must be fitted before prediction.")
        return self.gmm.predict_proba(features)

    def get_regime_labels(self, features):
        """Get the most likely regime label."""
        return self.gmm.predict(features)
