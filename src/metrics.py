"""
Evaluation metrics for multi-step forecasting.
Phase 2 adds MASE (scaled against seasonal naive), training-curve plots,
and a per-horizon bar chart.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mae(y_true, y_pred):
    """How much the model was off by, on average (in original units)."""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Similar to MAE, but penalizes large mistakes more heavily."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, epsilon=1e-10):
    """The average percentage the model was off by."""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

def smape(y_true, y_pred):
    """
    A fair version of the percentage error that treats over-predictions 
    and under-predictions equally. This is our main 'grade' metric.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-10)) * 100

def mase(y_true, y_pred, y_train, seasonal_period=24):
    """
    This shows if our model is smarter than a 'lazy' guess (guessing 
    tomorrow will be like yesterday). If this is less than 1, we win.
    """
    y_train = np.array(y_train).ravel()
    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = np.mean(naive_errors) + 1e-10
    return mae(y_true, y_pred) / scale


def seasonal_naive_forecast_windows(y_series, test_start_pos, n_windows, horizon, seasonal_period=24):
    """
    The 'lazy' prediction: just guess that the temperature today will be the 
    same as it was at the same time yesterday.
    """
    y = np.asarray(y_series).ravel()
    out = np.empty((n_windows, horizon))
    for i in range(n_windows):
        for h in range(horizon):
            idx = test_start_pos + i + h - seasonal_period
            out[i, h] = y[idx] if idx >= 0 else y[0]
    return out


def print_training_interpretation(train_losses, val_losses):
    """Translates the training math into a simple explanation of the model's progress."""
    if not train_losses or not val_losses:
        return
    last_t, last_v = train_losses[-1], val_losses[-1]
    ratio = last_v / (last_t + 1e-12)
    print("\n--- How the model learned ---")
    print(f"  Final scores: Training error = {last_t:.4f}, Validation error = {last_v:.4f}.")
    print("  Note: If validation error is much higher than training, it means")
    print("  the test year was harder to predict than the training years.")


def print_per_horizon_diagnosis(per_horizon_df, metric='sMAPE'):
    """Tells us which future hour was the easiest or hardest to predict."""
    if per_horizon_df is None or per_horizon_df.empty or metric not in per_horizon_df.columns:
        return
    best = per_horizon_df.loc[per_horizon_df[metric].idxmin()]
    worst = per_horizon_df.loc[per_horizon_df[metric].idxmax()]
    print("\n--- Which hour to focus on ---")
    print(f"  Easiest hour: {best['Horizon']} (error only {best[metric]:.4f})")
    print(f"  Hardest hour: {worst['Horizon']} (error reached {worst[metric]:.4f})")


def evaluate_forecast(y_true, y_pred, metric_names=['sMAPE', 'MAE', 'RMSE']):
    """Calculates all the different ways we measure the model's accuracy."""
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
    """Breaks down the accuracy for each hour into the future (Hour +1, +2, etc.)."""
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
    """Checks if the model works better in some situations (like stable days) over others."""
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
    """Prints out a clear summary of how the model did overall."""
    print(f"\n{'='*60}")
    print(f"{model_name} Report Card")
    print(f"{'='*60}")
    
    overall = evaluate_forecast(y_true, y_pred)
    print(f"\nOverall Performance:")
    for metric, value in overall.items():
        print(f"  {metric}: {value:.6f}")
    
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        print(f"\nPerformance for each future hour:")
        per_horizon = evaluate_per_horizon(y_true, y_pred)
        print(per_horizon.to_string(index=False))
    
    if regime_labels is not None:
        print(f"\nPerformance in different situations:")
        by_regime = evaluate_by_regime(y_true, y_pred, regime_labels)
        print(by_regime.to_string(index=False))
    
    print(f"{'='*60}\n")

def plot_training_curves(train_losses, val_losses, save_path=None):
    """Creates a chart showing how the model's errors went down as it practiced."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Practice Error', linewidth=1.5)
    ax.plot(epochs, val_losses, label='Test Error', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Study Rounds')
    ax.set_ylabel('Error Size')
    ax.set_title('Learning Progress')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved learning chart to {save_path}")
    return fig


def plot_per_horizon(per_horizon_df, metric='sMAPE', save_path=None):
    """Creates a bar chart showing which future hour was hardest to guess."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(per_horizon_df['Horizon'], per_horizon_df[metric], alpha=0.8)
    ax.set_xlabel('Future Hour')
    ax.set_ylabel('Error Score')
    ax.set_title('Errors per Forecast Hour')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved bar chart to {save_path}")
    return fig


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
