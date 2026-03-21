"""
Machine Learning models for multi-step time series forecasting.
Uses lag and rolling features with gradient boosting or random forest.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

class MLForecaster:
    """
    Multi-step forecaster using scikit-learn models with engineered features.
    """
    
    def __init__(self, model_type='histgb', horizon=5, **model_params):
        """
        Args:
            model_type: 'histgb' or 'rf'
            horizon: Number of steps to forecast
            model_params: Additional parameters for the base model
        """
        self.model_type = model_type
        self.horizon = horizon
        self.model_params = model_params
        self.model = None
        
        # Set default parameters if not provided
        if model_type == 'histgb':
            default_params = {
                'max_iter': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,
                'random_state': 42
            }
            default_params.update(model_params)
            base_model = HistGradientBoostingRegressor(**default_params)
        elif model_type == 'rf':
            default_params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(model_params)
            base_model = RandomForestRegressor(**default_params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Wrap in MultiOutputRegressor for multi-step forecasting
        self.model = MultiOutputRegressor(base_model)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples, horizon)
            X_val: Optional validation features (for monitoring)
            y_val: Optional validation targets
        """
        print(f"Training {self.model_type.upper()} model...")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Horizon: {y_train.shape[1]}")
        
        self.model.fit(X_train, y_train)
        
        # Report training performance
        train_pred = self.model.predict(X_train)
        train_mae = np.mean(np.abs(y_train - train_pred))
        print(f"  Training MAE: {train_mae:.6f}")
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mae = np.mean(np.abs(y_val - val_pred))
            print(f"  Validation MAE: {val_mae:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Generate predictions.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predictions (n_samples, horizon)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def save(self, path):
        """Save model to disk."""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from disk."""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self

class DirectMultiStepForecaster:
    """
    Alternative approach: train separate models for each horizon.
    Can be more flexible but requires more training time.
    """
    
    def __init__(self, model_type='histgb', horizon=5, **model_params):
        self.model_type = model_type
        self.horizon = horizon
        self.model_params = model_params
        self.models = []
        
        # Create one model per horizon
        for h in range(horizon):
            if model_type == 'histgb':
                default_params = {
                    'max_iter': 200,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'n_iter_no_change': 10,
                    'random_state': 42 + h
                }
                default_params.update(model_params)
                model = HistGradientBoostingRegressor(**default_params)
            elif model_type == 'rf':
                default_params = {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'random_state': 42 + h,
                    'n_jobs': -1
                }
                default_params.update(model_params)
                model = RandomForestRegressor(**default_params)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            self.models.append(model)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit separate model for each horizon."""
        print(f"Training {self.horizon} separate {self.model_type.upper()} models...")
        
        for h in range(self.horizon):
            self.models[h].fit(X_train, y_train[:, h])
            
            if (h + 1) % 2 == 0 or h == self.horizon - 1:
                train_pred = self.models[h].predict(X_train)
                train_mae = np.mean(np.abs(y_train[:, h] - train_pred))
                print(f"  Horizon {h+1} - Training MAE: {train_mae:.6f}")
        
        return self
    
    def predict(self, X):
        """Generate predictions from all horizon-specific models."""
        predictions = []
        for h in range(self.horizon):
            pred = self.models[h].predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)

def create_ml_forecaster(model_type='histgb', horizon=5, multi_output=True, **model_params):
    """
    Factory function to create ML forecaster.
    
    Args:
        model_type: 'histgb' or 'rf'
        horizon: Forecast horizon
        multi_output: If True, use MultiOutputRegressor; else use separate models
        model_params: Additional model parameters
    """
    if multi_output:
        return MLForecaster(model_type=model_type, horizon=horizon, **model_params)
    else:
        return DirectMultiStepForecaster(model_type=model_type, horizon=horizon, **model_params)

if __name__ == '__main__':
    # Test ML models
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randn(1000, 5)
    X_test = np.random.randn(100, 20)
    
    print("Testing HistGradientBoosting...")
    model = create_ml_forecaster(model_type='histgb', horizon=5)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Predictions shape: {pred.shape}")
    
    print("\nML models ready!")
