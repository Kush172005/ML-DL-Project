"""
Statistical baseline models for time series forecasting.
Includes seasonal naive and drift methods.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class SeasonalNaive:
    """
    Seasonal Naive forecaster: predicts using the value from the same 
    weekday in the previous week (lag 5 for daily trading data).
    """
    
    def __init__(self, seasonal_period=5):
        self.seasonal_period = seasonal_period
        self.history = None
    
    def fit(self, y):
        """Store training history."""
        self.history = np.array(y)
        return self
    
    def predict(self, horizon=5):
        """
        Predict next horizon steps using seasonal naive approach.
        
        Args:
            horizon: Number of steps to forecast
        
        Returns:
            Array of predictions (horizon,)
        """
        if self.history is None:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        for h in range(horizon):
            # Use value from seasonal_period steps ago
            idx = -(self.seasonal_period - (h % self.seasonal_period))
            if idx == 0:
                idx = -self.seasonal_period
            predictions.append(self.history[idx])
        
        return np.array(predictions)
    
    def forecast_rolling(self, y, horizon=5):
        """
        Generate rolling forecasts for a test set.
        
        Args:
            y: Test data
            horizon: Forecast horizon
        
        Returns:
            predictions: (n_samples, horizon) array
        """
        predictions = []
        
        for i in range(len(y) - horizon + 1):
            # Use all data up to current point
            history = np.concatenate([self.history, y[:i]]) if i > 0 else self.history
            
            # Predict next horizon steps
            pred = []
            for h in range(horizon):
                idx = -(self.seasonal_period - (h % self.seasonal_period))
                if idx == 0:
                    idx = -self.seasonal_period
                pred.append(history[idx])
            
            predictions.append(pred)
        
        return np.array(predictions)

class NaiveDrift:
    """
    Naive drift forecaster: extends the last observed value with 
    the average historical change (drift).
    """
    
    def __init__(self):
        self.last_value = None
        self.drift = None
    
    def fit(self, y):
        """Calculate drift from training data."""
        y = np.array(y)
        self.last_value = y[-1]
        self.drift = (y[-1] - y[0]) / (len(y) - 1)
        return self
    
    def predict(self, horizon=5):
        """Predict with drift."""
        if self.last_value is None:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = [self.last_value + self.drift * (h + 1) for h in range(horizon)]
        return np.array(predictions)
    
    def forecast_rolling(self, y, horizon=5):
        """Generate rolling forecasts."""
        predictions = []
        
        for i in range(len(y) - horizon + 1):
            if i > 0:
                # Update with new data
                history = np.concatenate([np.array([self.last_value]), y[:i]])
                last_val = y[i-1]
                drift = (last_val - history[0]) / len(history)
            else:
                last_val = self.last_value
                drift = self.drift
            
            pred = [last_val + drift * (h + 1) for h in range(horizon)]
            predictions.append(pred)
        
        return np.array(predictions)

class ETSWrapper:
    """
    Wrapper for statsmodels Exponential Smoothing (ETS).
    Simple wrapper for additive trend and seasonal components.
    """
    
    def __init__(self, seasonal_periods=5, trend='add', seasonal='add'):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.model = None
        self.fitted_model = None
    
    def fit(self, y):
        """Fit ETS model."""
        try:
            self.model = ExponentialSmoothing(
                y, 
                seasonal_periods=self.seasonal_periods,
                trend=self.trend,
                seasonal=self.seasonal
            )
            self.fitted_model = self.model.fit()
            return self
        except Exception as e:
            print(f"ETS fitting failed: {e}")
            print("Falling back to simple exponential smoothing")
            self.model = ExponentialSmoothing(y, trend='add', seasonal=None)
            self.fitted_model = self.model.fit()
            return self
    
    def predict(self, horizon=5):
        """Forecast next horizon steps."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.fitted_model.forecast(steps=horizon)
    
    def forecast_rolling(self, y, horizon=5):
        """
        Generate rolling forecasts.
        Note: This refits the model at each step, which is computationally expensive.
        """
        predictions = []
        
        # Get initial training data from the fitted model
        train_data = self.fitted_model.model.endog
        
        for i in range(len(y) - horizon + 1):
            # Combine training data with test data seen so far
            if i > 0:
                current_data = np.concatenate([train_data, y[:i]])
            else:
                current_data = train_data
            
            # Refit and predict
            try:
                model = ExponentialSmoothing(
                    current_data,
                    seasonal_periods=self.seasonal_periods,
                    trend=self.trend,
                    seasonal=self.seasonal
                )
                fitted = model.fit()
                pred = fitted.forecast(steps=horizon)
            except:
                # Fallback to simpler model
                model = ExponentialSmoothing(current_data, trend='add', seasonal=None)
                fitted = model.fit()
                pred = fitted.forecast(steps=horizon)
            
            predictions.append(pred)
        
        return np.array(predictions)

if __name__ == '__main__':
    # Test baselines
    np.random.seed(42)
    y_train = np.cumsum(np.random.randn(100)) * 0.1
    y_test = np.cumsum(np.random.randn(20)) * 0.1
    
    print("Testing Seasonal Naive...")
    sn = SeasonalNaive(seasonal_period=5)
    sn.fit(y_train)
    pred = sn.predict(horizon=5)
    print(f"Single prediction: {pred}")
    
    rolling_pred = sn.forecast_rolling(y_test, horizon=5)
    print(f"Rolling predictions shape: {rolling_pred.shape}")
    
    print("\nTesting Naive Drift...")
    nd = NaiveDrift()
    nd.fit(y_train)
    pred = nd.predict(horizon=5)
    print(f"Single prediction: {pred}")
    
    print("\nBaseline models ready!")
