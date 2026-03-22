"""
Hybrid forecasting: combining SARIMAX linear base with TFT residual predictions.
Implements the teacher's "Residual Decomposition Architecture".
"""

import numpy as np
import pandas as pd

class ResidualHybridForecaster:
    """
    Combines statistical forecast (SARIMAX) with deep residual forecast (TFT).
    
    Architecture:
        final_forecast = sarimax_forecast + tft_residual_forecast
    
    This is the additive decomposition shown in the teacher's slide.
    """
    
    def __init__(self):
        self.method = 'additive_residual'
    
    def combine(self, sarimax_forecast, tft_residual_forecast):
        """
        Combine SARIMAX and TFT predictions.
        
        Args:
            sarimax_forecast: Predictions from SARIMAX (n_samples, horizon)
            tft_residual_forecast: Predictions from TFT on residuals (n_samples, horizon)
        
        Returns:
            Combined forecast (n_samples, horizon)
        """
        # Simple additive combination
        combined = sarimax_forecast + tft_residual_forecast
        
        return combined
    
    def explain_weights(self):
        """
        For viva: explain that this is NOT weighted averaging.
        It's residual decomposition: TFT learns what SARIMAX missed.
        """
        explanation = """
        Hybrid Method: Residual Decomposition
        
        1. SARIMAX captures linear trend + seasonality
        2. Compute residuals = actual - SARIMAX_fitted
        3. TFT learns to predict these residuals (nonlinear patterns)
        4. Final = SARIMAX_forecast + TFT_residual_forecast
        
        This is different from weighted averaging:
        - Not competing models
        - Complementary: linear base + nonlinear correction
        - Matches teacher's architecture diagram
        """
        return explanation

class SimpleAverageEnsemble:
    """
    Alternative: simple average of SARIMAX and TFT (if both predict OT directly).
    Not used in main pipeline, but kept for comparison.
    """
    
    def combine(self, pred1, pred2):
        """Average two predictions."""
        return (pred1 + pred2) / 2

if __name__ == '__main__':
    # Test hybrid
    np.random.seed(42)
    
    sarimax_pred = np.random.randn(100, 24) + 10
    tft_resid_pred = np.random.randn(100, 24) * 0.5
    
    hybrid = ResidualHybridForecaster()
    combined = hybrid.combine(sarimax_pred, tft_resid_pred)
    
    print(f"SARIMAX shape: {sarimax_pred.shape}")
    print(f"TFT residual shape: {tft_resid_pred.shape}")
    print(f"Combined shape: {combined.shape}")
    print(f"\nHybrid method: {hybrid.method}")
    print(hybrid.explain_weights())
