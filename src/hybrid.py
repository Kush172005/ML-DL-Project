"""
Hybrid forecasting: combining statistical, ML, and DL predictions.
Uses inverse validation MSE weights for uncertainty-aware blending.
"""

import numpy as np
from src.metrics import evaluate_forecast

class HybridForecaster:
    """
    Combines multiple forecasters using weighted averaging.
    Weights are computed based on validation performance (inverse MSE).
    """
    
    def __init__(self):
        self.weights = None
        self.model_names = []
    
    def compute_weights(self, y_val, predictions_dict):
        """
        Compute inverse MSE weights based on validation performance.
        
        Args:
            y_val: Validation targets (n_samples, horizon) or (n_samples,)
            predictions_dict: Dictionary of {model_name: predictions}
        
        Returns:
            Dictionary of normalized weights
        """
        mse_scores = {}
        
        for name, pred in predictions_dict.items():
            mse = np.mean((y_val - pred) ** 2)
            mse_scores[name] = mse
        
        # Inverse MSE weights
        inv_mse = {name: 1.0 / (mse + 1e-10) for name, mse in mse_scores.items()}
        
        # Normalize
        total = sum(inv_mse.values())
        weights = {name: w / total for name, w in inv_mse.items()}
        
        self.weights = weights
        self.model_names = list(weights.keys())
        
        print("\nHybrid Model Weights (based on validation MSE):")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f} (MSE: {mse_scores[name]:.6f})")
        
        return weights
    
    def predict(self, predictions_dict):
        """
        Generate hybrid predictions using computed weights.
        
        Args:
            predictions_dict: Dictionary of {model_name: predictions}
        
        Returns:
            Weighted average predictions
        """
        if self.weights is None:
            raise ValueError("Weights must be computed before prediction")
        
        # Initialize with zeros
        first_pred = list(predictions_dict.values())[0]
        hybrid_pred = np.zeros_like(first_pred)
        
        # Weighted sum
        for name in self.model_names:
            if name in predictions_dict:
                hybrid_pred += self.weights[name] * predictions_dict[name]
        
        return hybrid_pred
    
    def fit_predict(self, y_val, val_predictions, test_predictions):
        """
        Convenience method: compute weights on validation and predict on test.
        
        Args:
            y_val: Validation targets
            val_predictions: Dict of validation predictions
            test_predictions: Dict of test predictions
        
        Returns:
            Test predictions from hybrid model
        """
        self.compute_weights(y_val, val_predictions)
        return self.predict(test_predictions)

class SimpleAverageEnsemble:
    """
    Simple ensemble that averages all model predictions equally.
    Baseline for comparison with weighted hybrid.
    """
    
    def __init__(self):
        pass
    
    def predict(self, predictions_dict):
        """Average all predictions."""
        predictions = list(predictions_dict.values())
        return np.mean(predictions, axis=0)

class StackedEnsemble:
    """
    Stacked ensemble: uses a meta-model to combine base predictions.
    More sophisticated but requires more data.
    """
    
    def __init__(self, meta_model=None):
        from sklearn.linear_model import Ridge
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
    
    def fit(self, y_val, val_predictions_dict):
        """
        Fit meta-model on validation predictions.
        
        Args:
            y_val: Validation targets (n_samples, horizon)
            val_predictions_dict: Dict of {model_name: predictions}
        """
        # Stack predictions as features
        X_meta = np.column_stack([pred.flatten() for pred in val_predictions_dict.values()])
        y_meta = y_val.flatten()
        
        self.meta_model.fit(X_meta, y_meta)
        self.model_names = list(val_predictions_dict.keys())
        
        print(f"\nStacked ensemble trained with {len(self.model_names)} base models")
        
        return self
    
    def predict(self, test_predictions_dict):
        """Generate predictions using meta-model."""
        X_meta = np.column_stack([test_predictions_dict[name].flatten() 
                                 for name in self.model_names])
        
        predictions_flat = self.meta_model.predict(X_meta)
        
        # Reshape back to (n_samples, horizon)
        first_pred = list(test_predictions_dict.values())[0]
        predictions = predictions_flat.reshape(first_pred.shape)
        
        return predictions

def create_hybrid_forecaster(method='inverse_mse'):
    """
    Factory function to create hybrid forecaster.
    
    Args:
        method: 'inverse_mse', 'simple_average', or 'stacked'
    """
    if method == 'inverse_mse':
        return HybridForecaster()
    elif method == 'simple_average':
        return SimpleAverageEnsemble()
    elif method == 'stacked':
        return StackedEnsemble()
    else:
        raise ValueError(f"Unknown method: {method}")

if __name__ == '__main__':
    # Test hybrid forecaster
    np.random.seed(42)
    
    y_val = np.random.randn(100, 5)
    
    # Simulate predictions from different models
    val_preds = {
        'baseline': y_val + np.random.randn(100, 5) * 0.5,
        'ml': y_val + np.random.randn(100, 5) * 0.3,
        'dl': y_val + np.random.randn(100, 5) * 0.4
    }
    
    test_preds = {
        'baseline': np.random.randn(50, 5),
        'ml': np.random.randn(50, 5),
        'dl': np.random.randn(50, 5)
    }
    
    print("Testing Hybrid Forecaster...")
    hybrid = HybridForecaster()
    hybrid.compute_weights(y_val, val_preds)
    hybrid_pred = hybrid.predict(test_preds)
    print(f"Hybrid predictions shape: {hybrid_pred.shape}")
    
    print("\nHybrid forecaster ready!")
