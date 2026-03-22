"""
End-to-end training pipeline for Hybrid Temporal Forecaster (ETTh1).
Architecture: SARIMAX (linear base) + GMM (regimes) + TFT (residuals) → Combined forecast
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data_prep import load_etth1, clean_data, chronological_split, plot_target_and_splits
from src.baselines import SARIMAXForecaster
from src.gmm_regimes import compute_residuals
from src.gmm_regimes import create_gmm_features, GMMRegimeDetector
from src.models_tft import TFTDataset, train_tft_simple, predict_tft_simple
from src.hybrid import ResidualHybridForecaster
from src.metrics import evaluate_forecast, evaluate_per_horizon, print_evaluation_summary

# Configuration
HORIZON = 24  # 24 hours ahead
ENCODER_LENGTH = 48  # 48 hours lookback
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def main():
    print("="*80)
    print("HYBRID TEMPORAL FORECASTER - ETTh1 (Teacher Architecture)")
    print("="*80)
    print("Architecture: SARIMAX → Residuals → GMM Regimes → TFT → Combined Forecast")
    print("="*80)
    
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # =========================================================================
    # 1. DATA PREPARATION
    # =========================================================================
    print("\n[1/6] Loading ETTh1 data...")
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    df = load_etth1(data_dir)
    df = clean_data(df)
    
    train_idx, val_idx, test_idx = chronological_split(df, train_frac=0.6, val_frac=0.2)
    
    # Plot splits
    plot_target_and_splits(df, train_idx, val_idx, test_idx, 
                          save_path=figures_dir / 'data_splits.png')
    
    # =========================================================================
    # 2. SARIMAX LINEAR BASE
    # =========================================================================
    print("\n[2/6] Training SARIMAX linear base...")
    
    y_train = df.loc[train_idx, 'OT']
    y_val = df.loc[val_idx, 'OT']
    y_test = df.loc[test_idx, 'OT']
    
    # Fit SARIMAX
    sarimax = SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    sarimax.fit(y_train)
    
    # Get in-sample fitted values and compute residuals
    fitted_train = sarimax.get_fitted_values()
    residuals_train = compute_residuals(y_train, fitted_train)
    
    print(f"  Train residuals: mean={np.mean(residuals_train):.4f}, std={np.std(residuals_train):.4f}")
    
    # Generate SARIMAX forecasts for validation and test
    # For simplicity: forecast from end of train for each window
    # (In production, would refit periodically)
    
    sarimax_val_forecasts = []
    sarimax_test_forecasts = []
    
    # Validation forecasts (simplified: one long forecast)
    val_forecast_full = sarimax.forecast(steps=len(val_idx))
    for i in range(len(val_idx) - HORIZON + 1):
        sarimax_val_forecasts.append(val_forecast_full[i:i+HORIZON].values)
    sarimax_val_forecasts = np.array(sarimax_val_forecasts)
    
    # Test forecasts (extend with validation data)
    extended_train = pd.concat([y_train, y_val])
    sarimax_test = SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    sarimax_test.fit(extended_train)
    
    test_forecast_full = sarimax_test.forecast(steps=len(test_idx))
    for i in range(len(test_idx) - HORIZON + 1):
        sarimax_test_forecasts.append(test_forecast_full[i:i+HORIZON].values)
    sarimax_test_forecasts = np.array(sarimax_test_forecasts)
    
    print(f"  SARIMAX val forecasts: {sarimax_val_forecasts.shape}")
    print(f"  SARIMAX test forecasts: {sarimax_test_forecasts.shape}")
    
    # =========================================================================
    # 3. GMM REGIME DETECTION
    # =========================================================================
    print("\n[3/6] Detecting regimes with GMM...")
    
    # Create GMM features from training residuals
    gmm_features_train = create_gmm_features(residuals_train, window=24)
    
    # Fit GMM on training features only
    gmm = GMMRegimeDetector(n_components=2, random_state=RANDOM_SEED)
    gmm.fit(gmm_features_train.values)
    
    # For val/test, we need to compute residuals in a causal way
    # Use SARIMAX fitted/forecast values
    residuals_val = y_val.values - val_forecast_full[:len(y_val)].values
    residuals_test = y_test.values - test_forecast_full[:len(y_test)].values
    
    gmm_features_val = create_gmm_features(residuals_val, window=24)
    gmm_features_test = create_gmm_features(residuals_test, window=24)
    
    # Get regime probabilities
    regime_probs_train = gmm.predict_proba(gmm_features_train.values)
    regime_probs_val = gmm.predict_proba(gmm_features_val.values)
    regime_probs_test = gmm.predict_proba(gmm_features_test.values)
    
    print(f"  Regime probs - Train: {regime_probs_train.shape}, Val: {regime_probs_val.shape}, Test: {regime_probs_test.shape}")
    
    # =========================================================================
    # 4. PREPARE DATA FOR TFT
    # =========================================================================
    print("\n[4/6] Preparing data for TFT...")
    
    # Align residuals and regime probs (GMM features drop some rows due to rolling)
    # Use the indices after GMM feature creation
    train_start_idx = len(residuals_train) - len(regime_probs_train)
    val_start_idx = len(residuals_val) - len(regime_probs_val)
    test_start_idx = len(residuals_test) - len(regime_probs_test)
    
    residuals_train_aligned = residuals_train[train_start_idx:]
    residuals_val_aligned = residuals_val[val_start_idx:]
    residuals_test_aligned = residuals_test[test_start_idx:]
    
    dates_train_aligned = train_idx[train_start_idx:]
    dates_val_aligned = val_idx[val_start_idx:]
    dates_test_aligned = test_idx[test_start_idx:]
    
    # Get covariates (HUFL, HULL, MUFL, MULL)
    covariates_cols = ['HUFL', 'HULL', 'MUFL', 'MULL']
    covariates_train = df.loc[dates_train_aligned, covariates_cols].values
    covariates_val = df.loc[dates_val_aligned, covariates_cols].values
    covariates_test = df.loc[dates_test_aligned, covariates_cols].values
    
    # Create TFT datasets
    train_dataset = TFTDataset(
        residuals_train_aligned,
        regime_probs_train,
        covariates_train,
        encoder_length=ENCODER_LENGTH,
        horizon=HORIZON
    )
    
    val_dataset = TFTDataset(
        residuals_val_aligned,
        regime_probs_val,
        covariates_val,
        encoder_length=ENCODER_LENGTH,
        horizon=HORIZON
    )
    
    test_dataset = TFTDataset(
        residuals_test_aligned,
        regime_probs_test,
        covariates_test,
        encoder_length=ENCODER_LENGTH,
        horizon=HORIZON
    )
    
    print(f"  TFT datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # =========================================================================
    # 5. TRAIN TFT
    # =========================================================================
    print("\n[5/6] Training Temporal Fusion Transformer...")
    
    # Calculate input sizes
    encoder_input_size = 1 + regime_probs_train.shape[1] + covariates_train.shape[1]  # residual + regimes + covariates
    decoder_input_size = regime_probs_train.shape[1] + covariates_train.shape[1]  # regimes + covariates
    
    tft_model = train_tft_simple(
        train_dataset,
        val_dataset,
        encoder_input_size=encoder_input_size,
        decoder_input_size=decoder_input_size,
        hidden_size=64,
        num_layers=2,
        horizon=HORIZON,
        dropout=0.2,
        learning_rate=0.001,
        epochs=50,
        patience=10,
        batch_size=64
    )
    
    # Generate TFT predictions (on residuals)
    print("\n  Generating TFT predictions...")
    tft_val_pred = predict_tft_simple(tft_model, val_dataset, batch_size=128)
    tft_test_pred = predict_tft_simple(tft_model, test_dataset, batch_size=128)
    
    print(f"  TFT val predictions: {tft_val_pred.shape}")
    print(f"  TFT test predictions: {tft_test_pred.shape}")
    
    # =========================================================================
    # 6. COMBINE AND EVALUATE
    # =========================================================================
    print("\n[6/6] Combining forecasts and evaluating...")
    
    hybrid = ResidualHybridForecaster()
    
    # Align SARIMAX and TFT predictions
    # TFT produces fewer samples due to encoder length requirement
    n_val_tft = len(tft_val_pred)
    n_test_tft = len(tft_test_pred)
    
    sarimax_val_aligned = sarimax_val_forecasts[:n_val_tft]
    sarimax_test_aligned = sarimax_test_forecasts[:n_test_tft]
    
    # Combined forecasts
    combined_val = hybrid.combine(sarimax_val_aligned, tft_val_pred)
    combined_test = hybrid.combine(sarimax_test_aligned, tft_test_pred)
    
    # Prepare ground truth
    y_val_windows = []
    for i in range(n_val_tft):
        y_val_windows.append(y_val.values[i:i+HORIZON])
    y_val_windows = np.array(y_val_windows)
    
    y_test_windows = []
    for i in range(n_test_tft):
        y_test_windows.append(y_test.values[i:i+HORIZON])
    y_test_windows = np.array(y_test_windows)
    
    # Evaluate all models
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    models_results = {}
    
    # SARIMAX only
    print_evaluation_summary(y_test_windows, sarimax_test_aligned, model_name='SARIMAX (Linear Base)')
    models_results['SARIMAX'] = evaluate_forecast(y_test_windows, sarimax_test_aligned)
    
    # TFT only (on residuals, so add back SARIMAX for fair comparison)
    tft_only_test = sarimax_test_aligned + tft_test_pred
    print_evaluation_summary(y_test_windows, tft_only_test, model_name='TFT (with SARIMAX base)')
    models_results['TFT'] = evaluate_forecast(y_test_windows, tft_only_test)
    
    # Combined (same as TFT in this architecture)
    print_evaluation_summary(y_test_windows, combined_test, model_name='Hybrid (SARIMAX + TFT)')
    models_results['Hybrid'] = evaluate_forecast(y_test_windows, combined_test)
    
    # =========================================================================
    # 7. RESULTS SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(models_results).T
    print(results_df.to_string())
    
    results_df.to_csv(figures_dir / 'results_summary.csv')
    print(f"\nResults saved to {figures_dir / 'results_summary.csv'}")
    
    # =========================================================================
    # 8. VISUALIZATION
    # =========================================================================
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sample window for plotting
    sample_idx = 100
    sample_end = min(sample_idx + 100, len(y_test_windows))
    sample_range = range(sample_idx, sample_end)
    
    # SARIMAX
    axes[0, 0].plot(sample_range, y_test_windows[sample_idx:sample_end, 0], 'k-', 
                   label='Actual', linewidth=2, alpha=0.8)
    axes[0, 0].plot(sample_range, sarimax_test_aligned[sample_idx:sample_end, 0], '--', 
                   label='SARIMAX', alpha=0.7, linewidth=1.5)
    axes[0, 0].set_title('SARIMAX Linear Base (H+1)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylabel('OT (Oil Temperature)')
    
    # TFT residuals
    axes[0, 1].plot(sample_range, tft_test_pred[sample_idx:sample_end, 0], 
                   label='TFT Residual Pred', alpha=0.7, linewidth=1.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('TFT Residual Predictions (H+1)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylabel('Residual')
    
    # Combined
    axes[1, 0].plot(sample_range, y_test_windows[sample_idx:sample_end, 0], 'k-', 
                   label='Actual', linewidth=2, alpha=0.8)
    axes[1, 0].plot(sample_range, combined_test[sample_idx:sample_end, 0], '--', 
                   label='Hybrid', alpha=0.7, linewidth=1.5, color='green')
    axes[1, 0].set_title('Hybrid Forecast (H+1)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('OT (Oil Temperature)')
    
    # Model comparison bar chart
    axes[1, 1].bar(results_df.index, results_df['sMAPE'])
    axes[1, 1].set_title('Model Comparison (sMAPE)')
    axes[1, 1].set_ylabel('sMAPE (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {figures_dir / 'model_comparison.png'}")
    
    # =========================================================================
    # 9. PER-HORIZON ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("PER-HORIZON BREAKDOWN (Hybrid)")
    print("="*80)
    
    per_horizon = evaluate_per_horizon(y_test_windows, combined_test)
    print(per_horizon.to_string(index=False))
    per_horizon.to_csv(figures_dir / 'per_horizon_results.csv', index=False)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - {figures_dir / 'results_summary.csv'}")
    print(f"  - {figures_dir / 'per_horizon_results.csv'}")
    print(f"  - {figures_dir / 'model_comparison.png'}")
    print(f"  - {figures_dir / 'data_splits.png'}")

if __name__ == '__main__':
    main()
