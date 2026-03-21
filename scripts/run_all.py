"""
End-to-end training and evaluation script.
Trains all models (baseline, ML, DL, hybrid) and generates results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_prep import load_data, clean_data, add_returns, chronological_split, compute_regime_labels
from src.features import create_all_features, get_feature_columns, prepare_ml_data
from src.baselines import SeasonalNaive, NaiveDrift
from src.models_ml import create_ml_forecaster
from src.models_dl import TimeSeriesDataset, create_dl_forecaster, DLForecasterTrainer
from src.hybrid import HybridForecaster
from src.metrics import evaluate_forecast, evaluate_per_horizon, evaluate_by_regime, print_evaluation_summary
from src.failure_analysis import summarize_failure_buckets, plot_failure_diagnostics, export_worst_windows

import torch
from torch.utils.data import DataLoader

# Configuration
HORIZON = 5
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def main():
    print("="*80)
    print("HYBRID TEMPORAL FORECASTER - PHASE 1")
    print("="*80)
    
    # Create output directories
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # =========================================================================
    # 1. DATA PREPARATION
    # =========================================================================
    print("\n[1/6] Loading and preparing data...")
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    df = load_data(data_dir)
    df = clean_data(df)
    df = add_returns(df)
    df = compute_regime_labels(df)
    
    # Chronological split
    train_idx, val_idx, test_idx = chronological_split(df, train_frac=0.7, val_frac=0.15)
    
    # =========================================================================
    # 2. BASELINE MODELS
    # =========================================================================
    print("\n[2/6] Training baseline models...")
    
    # Seasonal Naive
    sn_model = SeasonalNaive(seasonal_period=5)
    sn_model.fit(df.loc[train_idx, 'log_return'].values)
    
    # Generate baseline predictions on test set
    test_returns = df.loc[test_idx, 'log_return'].values
    baseline_pred = sn_model.forecast_rolling(test_returns, horizon=HORIZON)
    
    # Align test targets
    y_test_baseline = []
    for i in range(len(baseline_pred)):
        y_test_baseline.append(test_returns[i:i+HORIZON])
    y_test_baseline = np.array(y_test_baseline)
    
    print(f"Baseline predictions shape: {baseline_pred.shape}")
    
    # =========================================================================
    # 3. ML MODEL
    # =========================================================================
    print("\n[3/6] Training ML model...")
    
    # Create features
    df_feat = create_all_features(df, include_calendar=True, include_vix=True)
    feature_cols = get_feature_columns(df_feat)
    
    # Prepare ML data
    X_full, y_full = prepare_ml_data(df_feat, feature_cols, target_col='log_return', horizon=HORIZON)
    
    # Align indices after feature engineering
    valid_idx = df_feat.index[:len(X_full)]
    
    # Split based on original splits
    train_mask = valid_idx.isin(train_idx)
    val_mask = valid_idx.isin(val_idx)
    test_mask = valid_idx.isin(test_idx)
    
    X_train = X_full[train_mask]
    y_train = y_full[train_mask]
    X_val = X_full[val_mask]
    y_val = y_full[val_mask]
    X_test = X_full[test_mask]
    y_test = y_full[test_mask]
    
    print(f"ML data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train HistGradientBoosting
    ml_model = create_ml_forecaster(model_type='histgb', horizon=HORIZON)
    ml_model.fit(X_train, y_train, X_val, y_val)
    
    ml_pred_val = ml_model.predict(X_val)
    ml_pred_test = ml_model.predict(X_test)
    
    # =========================================================================
    # 4. DEEP LEARNING MODEL
    # =========================================================================
    print("\n[4/6] Training deep learning model...")
    
    # Prepare sequence data (use price + returns + vix)
    sequence_data = df[['log_return', 'vix']].values
    
    # Normalize for DL
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sequence_data_scaled = scaler.fit_transform(sequence_data)
    
    # Create datasets
    window_size = 20
    
    train_data = sequence_data_scaled[:len(train_idx)]
    val_data = sequence_data_scaled[len(train_idx):len(train_idx)+len(val_idx)]
    test_data = sequence_data_scaled[len(train_idx)+len(val_idx):]
    
    train_dataset = TimeSeriesDataset(train_data, window_size=window_size, horizon=HORIZON, target_col_idx=0)
    val_dataset = TimeSeriesDataset(
        np.vstack([train_data, val_data]), 
        window_size=window_size, 
        horizon=HORIZON, 
        target_col_idx=0
    )
    # Only use validation portion
    val_dataset.indices = [i for i in val_dataset.indices if i >= len(train_data) - window_size]
    
    test_dataset = TimeSeriesDataset(
        sequence_data_scaled,
        window_size=window_size,
        horizon=HORIZON,
        target_col_idx=0
    )
    # Only use test portion
    test_dataset.indices = [i for i in test_dataset.indices if i >= len(train_idx) + len(val_idx) - window_size]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"DL datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create and train LSTM
    dl_model = create_dl_forecaster(
        model_type='lstm',
        input_size=sequence_data_scaled.shape[1],
        hidden_size=64,
        num_layers=2,
        horizon=HORIZON,
        dropout=0.2
    )
    
    trainer = DLForecasterTrainer(dl_model, device='cpu', learning_rate=0.001)
    trainer.fit(train_loader, val_loader, epochs=100, patience=15, verbose=True)
    
    # Generate predictions
    dl_pred_val = trainer.predict(val_loader)
    dl_pred_test = trainer.predict(test_loader)
    
    # Get corresponding targets
    y_val_dl = []
    for _, y in val_loader:
        y_val_dl.append(y.numpy())
    y_val_dl = np.vstack(y_val_dl)
    
    y_test_dl = []
    for _, y in test_loader:
        y_test_dl.append(y.numpy())
    y_test_dl = np.vstack(y_test_dl)
    
    # =========================================================================
    # 5. HYBRID MODEL
    # =========================================================================
    print("\n[5/6] Creating hybrid model...")
    
    # Align predictions - ML and DL may have different sample counts
    # Use the smaller set (DL) as reference
    min_val_samples = min(len(ml_pred_val), len(dl_pred_val))
    min_test_samples = min(len(ml_pred_test), len(dl_pred_test))
    
    # Trim to match
    ml_pred_val_aligned = ml_pred_val[:min_val_samples]
    dl_pred_val_aligned = dl_pred_val[:min_val_samples]
    y_val_aligned = y_val[:min_val_samples]
    
    ml_pred_test_aligned = ml_pred_test[:min_test_samples]
    dl_pred_test_aligned = dl_pred_test[:min_test_samples]
    y_test_aligned = y_test[:min_test_samples]
    X_test_aligned = X_test[:min_test_samples]
    
    hybrid = HybridForecaster()
    
    # Validation predictions (for weight computation)
    val_predictions = {
        'ML': ml_pred_val_aligned,
        'DL': dl_pred_val_aligned
    }
    
    # Test predictions
    test_predictions = {
        'ML': ml_pred_test_aligned,
        'DL': dl_pred_test_aligned
    }
    
    # Compute weights and predict
    hybrid.compute_weights(y_val_aligned, val_predictions)
    hybrid_pred = hybrid.predict(test_predictions)
    
    # =========================================================================
    # 6. EVALUATION
    # =========================================================================
    print("\n[6/6] Evaluating all models...")
    
    # Evaluate each model
    models_results = {}
    
    # Baseline (on its own test set)
    print_evaluation_summary(y_test_baseline, baseline_pred, model_name='Seasonal Naive Baseline')
    models_results['Baseline'] = evaluate_forecast(y_test_baseline, baseline_pred)
    
    # ML
    print_evaluation_summary(y_test_aligned, ml_pred_test_aligned, model_name='ML (HistGradientBoosting)')
    models_results['ML'] = evaluate_forecast(y_test_aligned, ml_pred_test_aligned)
    
    # DL
    print_evaluation_summary(y_test_aligned, dl_pred_test_aligned, model_name='DL (LSTM)')
    models_results['DL'] = evaluate_forecast(y_test_aligned, dl_pred_test_aligned)
    
    # Hybrid
    print_evaluation_summary(y_test_aligned, hybrid_pred, model_name='Hybrid (Weighted Ensemble)')
    models_results['Hybrid'] = evaluate_forecast(y_test_aligned, hybrid_pred)

    # =========================================================================
    # 6b. REGIME SLICE (uses only past-based rolling vol feature — no extra labels)
    # =========================================================================
    if 'rolling_std_20' in feature_cols:
        j = feature_cols.index('rolling_std_20')
        rs_test = X_test_aligned[:, j]
        med = np.median(rs_test)
        reg_labels = np.where(rs_test > med, 'high_past_vol', 'low_past_vol')
        print("\n--- ML performance by past-volatility bucket (rolling_std_20 median split) ---")
        reg_df = evaluate_by_regime(
            y_test_aligned, ml_pred_test_aligned, reg_labels, regime_col='bucket'
        )
        print(reg_df.to_string(index=False))
        reg_df.to_csv(figures_dir / 'ml_regime_slice.csv', index=False)

    # =========================================================================
    # 6c. FAILURE ANALYSIS (ML, H+1): worst errors vs |actual return|
    # =========================================================================
    print("\n--- Failure analysis (ML, horizon H+1) ---")
    summ, worst_i, _ = summarize_failure_buckets(
        y_test_aligned, ml_pred_test_aligned, horizon_idx=0, n_worst=15
    )
    for k, v in summ.items():
        print(f"  {k}: {v}")
    fail_plot = plot_failure_diagnostics(
        y_test_aligned, ml_pred_test_aligned, figures_dir, prefix="ml", horizon_idx=0
    )
    fail_csv = export_worst_windows(
        worst_i, y_test_aligned, ml_pred_test_aligned,
        figures_dir / "ml_worst_h1_errors.csv", horizon_idx=0
    )
    print(f"  Saved: {fail_plot}")
    print(f"  Saved: {fail_csv}")
    
    # =========================================================================
    # 7. RESULTS SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(models_results).T
    print(results_df.to_string())
    
    # Save results
    results_df.to_csv(figures_dir / 'results_summary.csv')
    print(f"\nResults saved to {figures_dir / 'results_summary.csv'}")
    
    # =========================================================================
    # 8. VISUALIZATION
    # =========================================================================
    print("\nGenerating visualizations...")
    
    # Plot predictions vs actual for a sample window
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sample_idx = 50
    sample_end = min(sample_idx + 50, len(y_test_aligned))
    sample_range = range(sample_idx, sample_end)
    
    # Flatten for plotting (show first horizon only)
    axes[0, 0].plot(sample_range, y_test_aligned[sample_idx:sample_end, 0], 'k-', label='Actual', linewidth=2)
    axes[0, 0].plot(sample_range, ml_pred_test_aligned[sample_idx:sample_end, 0], '--', label='ML', alpha=0.7)
    axes[0, 0].set_title('ML Model (H+1 predictions)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(sample_range, y_test_aligned[sample_idx:sample_end, 0], 'k-', label='Actual', linewidth=2)
    axes[0, 1].plot(sample_range, dl_pred_test_aligned[sample_idx:sample_end, 0], '--', label='DL', alpha=0.7)
    axes[0, 1].set_title('DL Model (H+1 predictions)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(sample_range, y_test_aligned[sample_idx:sample_end, 0], 'k-', label='Actual', linewidth=2)
    axes[1, 0].plot(sample_range, hybrid_pred[sample_idx:sample_end, 0], '--', label='Hybrid', alpha=0.7)
    axes[1, 0].set_title('Hybrid Model (H+1 predictions)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Bar chart of overall performance
    axes[1, 1].bar(results_df.index, results_df['MAE'])
    axes[1, 1].set_title('Model Comparison (MAE)')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {figures_dir / 'model_comparison.png'}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
