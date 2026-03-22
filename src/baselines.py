"""
Statistical baseline: SARIMAX for trend and seasonality modeling.
This forms the "linear base" in the teacher's architecture.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

class SARIMAXForecaster:
    """
    SARIMAX wrapper for hourly data with daily seasonality.
    Captures linear trend and seasonal patterns.
    """
    
    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24)):
        """
        Args:
            order: (p, d, q) for ARIMA
            seasonal_order: (P, D, Q, s) for seasonal component
                Default s=24 for daily seasonality in hourly data
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.train_data = None
    
    def fit(self, y_train):
        """
        Fit SARIMAX on training data.
        
        Args:
            y_train: Training time series (1D array or Series)
        """
        print(f"Fitting SARIMAX{self.order} with seasonal{self.seasonal_order}...")
        
        self.train_data = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
        
        try:
            self.model = SARIMAX(
                self.train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = self.model.fit(disp=False, maxiter=200)
            print(f"  AIC: {self.fitted_model.aic:.2f}")
            print(f"  Converged: {self.fitted_model.mle_retvals['converged']}")
            
        except Exception as e:
            print(f"SARIMAX fitting failed: {e}")
            print("Falling back to simpler ARIMA(1,0,1)")
            self.model = SARIMAX(
                self.train_data,
                order=(1, 0, 1),
                seasonal_order=(0, 0, 0, 0)
            )
            self.fitted_model = self.model.fit(disp=False, maxiter=100)
        
        return self
    
    def get_fitted_values(self):
        """Get in-sample fitted values (for computing residuals)."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.fittedvalues
    
    def forecast(self, steps=24):
        """
        Forecast next 'steps' periods.
        
        Args:
            steps: Forecast horizon
        
        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.forecast(steps=steps)
    
    def forecast_rolling(self, y_test, horizon=24, refit_every=None):
        """
        Generate rolling forecasts on test set.
        
        Args:
            y_test: Test data
            horizon: Forecast horizon
            refit_every: If None, use recursive forecasting without refitting.
                        If int, refit every N steps (slower but more adaptive)
        
        Returns:
            predictions: (n_forecasts, horizon) array
        """
        predictions = []
        
        if refit_every is None:
            # Simple approach: extend with observed test data, forecast recursively
            for i in range(len(y_test) - horizon + 1):
                if i == 0:
                    # First forecast from trained model
                    pred = self.forecast(steps=horizon)
                else:
                    # Append observed data and reforecast
                    extended_data = pd.concat([
                        self.train_data,
                        pd.Series(y_test[:i], index=range(len(self.train_data), len(self.train_data) + i))
                    ])
                    
                    # Quick forecast without full refit
                    try:
                        model_temp = SARIMAX(
                            extended_data,
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        fitted_temp = model_temp.fit(disp=False, maxiter=50, method='bfgs')
                        pred = fitted_temp.forecast(steps=horizon)
                    except:
                        # Fallback: use last known pattern
                        pred = self.forecast(steps=horizon)
                
                predictions.append(pred)
        
        return np.array(predictions)

class SimpleSeasonalBaseline:
    """
    Fallback: Seasonal naive for hourly data (lag-24).
    Used if SARIMAX is too slow or unstable.
    """
    
    def __init__(self, seasonal_period=24):
        self.seasonal_period = seasonal_period
        self.history = None
    
    def fit(self, y_train):
        self.history = np.array(y_train)
        return self
    
    def forecast(self, steps=24):
        """Repeat last seasonal cycle."""
        predictions = []
        for h in range(steps):
            idx = -(self.seasonal_period - (h % self.seasonal_period))
            if idx == 0:
                idx = -self.seasonal_period
            predictions.append(self.history[idx])
        return np.array(predictions)

if __name__ == '__main__':
    from data_prep import load_etth1, clean_data, chronological_split
    
    df = load_etth1()
    df = clean_data(df)
    train_idx, val_idx, test_idx = chronological_split(df)
    
    # Test SARIMAX
    y_train = df.loc[train_idx, 'OT']
    forecaster = SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    forecaster.fit(y_train)
    
    pred = forecaster.forecast(steps=24)
    print(f"\n24-step forecast: {pred[:5]}...")
    
    fitted = forecaster.get_fitted_values()
    print(f"Fitted values: {len(fitted)} points")
