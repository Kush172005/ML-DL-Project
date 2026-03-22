"""
Evaluation metrics for multi-step forecasting.
Includes per-horizon metrics and regime-specific analysis.
"""

import numpy as np
import pandas as pd

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, epsilon=1e-10):
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).
    Primary metric per teacher's specification.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-10)) * 100

def evaluate_forecast(y_true, y_pred, metric_names=['sMAPE', 'MAE', 'RMSE']):
    """
    Evaluate forecast with multiple metrics.
    
    Args:
        y_true: True values (n_samples, horizon) or (n_samples,)
        y_pred: Predicted values (n_samples, horizon) or (n_samples,)
        metric_names: List of metrics to compute
    
    Returns:
        Dictionary of metric values
    """
    results = {}
    
    if 'MAE' in metric_names:
        results['MAE'] = mae(y_true, y_pred)
    if 'RMSE' in metric_names:
        results['RMSE'] = rmse(y_true, y_pred)
    if 'MAPE' in metric_names:
        results['MAPE'] = mape(y_true, y_pred)
    if 'sMAPE' in metric_names:
        results['sMAPE'] = smape(y_true, y_pred)
    
    return results

def evaluate_per_horizon(y_true, y_pred, horizon_names=None):
    """
    Evaluate metrics for each forecast horizon separately.
    
    Args:
        y_true: True values (n_samples, horizon)
        y_pred: Predicted values (n_samples, horizon)
        horizon_names: Optional list of horizon labels
    
    Returns:
        DataFrame with metrics per horizon
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    horizon = y_true.shape[1]
    
    if horizon_names is None:
        horizon_names = [f'H+{i+1}' for i in range(horizon)]
    
    results = []
    for h in range(horizon):
        metrics = evaluate_forecast(y_true[:, h], y_pred[:, h])
        metrics['Horizon'] = horizon_names[h]
        results.append(metrics)
    
    return pd.DataFrame(results)

def evaluate_by_regime(y_true, y_pred, regime_labels, regime_col='regime'):
    """
    Evaluate metrics separately for different regimes (e.g., high/low volatility).
    
    Args:
        y_true: True values (n_samples, horizon) or (n_samples,)
        y_pred: Predicted values (n_samples, horizon) or (n_samples,)
        regime_labels: Array or Series of regime labels aligned with y_true
        regime_col: Name for regime column in output
    
    Returns:
        DataFrame with metrics per regime
    """
    if isinstance(regime_labels, pd.Series):
        regime_labels = regime_labels.values
    
    unique_regimes = np.unique(regime_labels)
    results = []
    
    for regime in unique_regimes:
        mask = regime_labels == regime
        if mask.sum() > 0:
            metrics = evaluate_forecast(y_true[mask], y_pred[mask])
            metrics[regime_col] = regime
            metrics['n_samples'] = mask.sum()
            results.append(metrics)
    
    return pd.DataFrame(results)

def print_evaluation_summary(y_true, y_pred, regime_labels=None, model_name='Model'):
    """Print comprehensive evaluation summary."""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Summary")
    print(f"{'='*60}")
    
    # Overall metrics
    overall = evaluate_forecast(y_true, y_pred)
    print(f"\nOverall Performance:")
    for metric, value in overall.items():
        print(f"  {metric}: {value:.6f}")
    
    # Per-horizon metrics (if multi-step)
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        print(f"\nPer-Horizon Performance:")
        per_horizon = evaluate_per_horizon(y_true, y_pred)
        print(per_horizon.to_string(index=False))
    
    # Regime-specific metrics
    if regime_labels is not None:
        print(f"\nRegime-Specific Performance:")
        by_regime = evaluate_by_regime(y_true, y_pred, regime_labels)
        print(by_regime.to_string(index=False))
    
    print(f"{'='*60}\n")

if __name__ == '__main__':
    # Test metrics
    np.random.seed(42)
    y_true = np.random.randn(100, 5)
    y_pred = y_true + np.random.randn(100, 5) * 0.1
    
    print("Testing metrics module...")
    results = evaluate_forecast(y_true, y_pred)
    print(f"Overall: {results}")
    
    per_horizon = evaluate_per_horizon(y_true, y_pred)
    print(f"\nPer horizon:\n{per_horizon}")
    
    regime_labels = np.random.choice(['high_vol', 'low_vol'], size=100)
    by_regime = evaluate_by_regime(y_true, y_pred, regime_labels)
    print(f"\nBy regime:\n{by_regime}")
