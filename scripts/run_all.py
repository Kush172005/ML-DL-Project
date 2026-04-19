"""
Phase 2 end-to-end pipeline for the Hybrid Temporal Forecaster.

Architecture:
  SARIMAX (linear seasonality)
  → residuals
  → GMM regime probabilities
  → TemporalHybridNet (BiLSTM + Multi-Head Attention + gated residual)
  → additive hybrid forecast

New in Phase 2 vs Phase 1:
  - Hour/day-of-week cyclic time features (sin/cos) added as covariates
  - TemporalHybridNet replaces SimplifiedTFT
  - Training loss curves saved to figures/
  - MASE added to evaluation table
  - Per-horizon bar chart
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data_prep import load_etth1, clean_data, chronological_split, plot_target_and_splits, add_time_features
from src.baselines import SARIMAXForecaster
from src.gmm_regimes import compute_residuals, create_gmm_features, GMMRegimeDetector
from src.models_tft import TFTDataset, train_tft_simple, predict_tft_simple
from src.hybrid import ResidualHybridForecaster
from src.metrics import (
    evaluate_forecast, evaluate_per_horizon, print_evaluation_summary,
    mase, plot_training_curves, plot_per_horizon,
    seasonal_naive_forecast_windows, print_training_interpretation,
    print_per_horizon_diagnosis,
)

HORIZON = 24
ENCODER_LENGTH = 48
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

COVARIATE_COLS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']


def main():
    print("=" * 80)
    print("HYBRID TEMPORAL FORECASTER — Phase 2 (ETTh1)")
    print("=" * 80)
    print("SARIMAX → Residuals → GMM → TemporalHybridNet (BiLSTM + Attention) → Combined")
    print("=" * 80)

    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True, parents=True)

    # -------------------------------------------------------------------------
    # 1. DATA PREPARATION
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading and preparing ETTh1 data...")
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    df = load_etth1(data_dir)
    df = clean_data(df)
    df = add_time_features(df)   # Phase 2: cyclical hour + day-of-week features

    train_idx, val_idx, test_idx = chronological_split(df, train_frac=0.6, val_frac=0.2)
    plot_target_and_splits(df, train_idx, val_idx, test_idx,
                           save_path=figures_dir / 'data_splits.png')

    # -------------------------------------------------------------------------
    # 2. SARIMAX LINEAR BASE
    # -------------------------------------------------------------------------
    print("\n[2/6] Training SARIMAX linear base...")
    y_train = df.loc[train_idx, 'OT']
    y_val   = df.loc[val_idx,   'OT']
    y_test  = df.loc[test_idx,  'OT']

    sarimax = SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    sarimax.fit(y_train)

    fitted_train    = sarimax.get_fitted_values()
    residuals_train = compute_residuals(y_train, fitted_train)
    print(f"  Train residuals: mean={np.mean(residuals_train):.4f}, std={np.std(residuals_train):.4f}")

    val_forecast_full = sarimax.forecast(steps=len(val_idx))
    sarimax_val_forecasts = [
        val_forecast_full[i:i + HORIZON].values
        for i in range(len(val_idx) - HORIZON + 1)
    ]
    sarimax_val_forecasts = np.array(sarimax_val_forecasts)

    sarimax_test = SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    sarimax_test.fit(pd.concat([y_train, y_val]))
    test_forecast_full = sarimax_test.forecast(steps=len(test_idx))
    sarimax_test_forecasts = [
        test_forecast_full[i:i + HORIZON].values
        for i in range(len(test_idx) - HORIZON + 1)
    ]
    sarimax_test_forecasts = np.array(sarimax_test_forecasts)

    print(f"  SARIMAX val windows: {sarimax_val_forecasts.shape}")
    print(f"  SARIMAX test windows: {sarimax_test_forecasts.shape}")

    # -------------------------------------------------------------------------
    # 3. GMM REGIME DETECTION
    # -------------------------------------------------------------------------
    print("\n[3/6] Detecting volatility regimes with GMM...")
    gmm_features_train = create_gmm_features(residuals_train, window=24)

    gmm = GMMRegimeDetector(n_components=2, random_state=RANDOM_SEED)
    gmm.fit(gmm_features_train.values)

    residuals_val  = y_val.values  - val_forecast_full[:len(y_val)].values
    residuals_test = y_test.values - test_forecast_full[:len(y_test)].values

    gmm_features_val  = create_gmm_features(residuals_val,  window=24)
    gmm_features_test = create_gmm_features(residuals_test, window=24)

    regime_probs_train = gmm.predict_proba(gmm_features_train.values)
    regime_probs_val   = gmm.predict_proba(gmm_features_val.values)
    regime_probs_test  = gmm.predict_proba(gmm_features_test.values)

    print(f"  Regime shapes — train: {regime_probs_train.shape}, val: {regime_probs_val.shape}, test: {regime_probs_test.shape}")

    # -------------------------------------------------------------------------
    # 4. PREPARE DATASETS FOR TemporalHybridNet
    # -------------------------------------------------------------------------
    print("\n[4/6] Building TFT datasets (with time features)...")

    train_start = len(residuals_train) - len(regime_probs_train)
    val_start   = len(residuals_val)   - len(regime_probs_val)
    test_start  = len(residuals_test)  - len(regime_probs_test)

    residuals_train_a = residuals_train[train_start:]
    residuals_val_a   = residuals_val[val_start:]
    residuals_test_a  = residuals_test[test_start:]

    dates_train_a = train_idx[train_start:]
    dates_val_a   = val_idx[val_start:]
    dates_test_a  = test_idx[test_start:]

    covariates_train = df.loc[dates_train_a, COVARIATE_COLS].values
    covariates_val   = df.loc[dates_val_a,   COVARIATE_COLS].values
    covariates_test  = df.loc[dates_test_a,  COVARIATE_COLS].values

    train_dataset = TFTDataset(residuals_train_a, regime_probs_train, covariates_train,
                               encoder_length=ENCODER_LENGTH, horizon=HORIZON)
    val_dataset   = TFTDataset(residuals_val_a,   regime_probs_val,   covariates_val,
                               encoder_length=ENCODER_LENGTH, horizon=HORIZON)
    test_dataset  = TFTDataset(residuals_test_a,  regime_probs_test,  covariates_test,
                               encoder_length=ENCODER_LENGTH, horizon=HORIZON)

    print(f"  Dataset sizes — train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    # -------------------------------------------------------------------------
    # 5. TRAIN TemporalHybridNet
    # -------------------------------------------------------------------------
    print("\n[5/6] Training TemporalHybridNet...")
    encoder_input_size = 1 + regime_probs_train.shape[1] + covariates_train.shape[1]
    decoder_input_size =     regime_probs_train.shape[1] + covariates_train.shape[1]

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
        batch_size=64,
        weight_decay=1e-4,
        n_heads=2,
        bidirectional=True,
    )

    # Save training curves
    plot_training_curves(
        tft_model.train_losses,
        tft_model.val_losses,
        save_path=figures_dir / 'training_curves.png',
    )
    print_training_interpretation(tft_model.train_losses, tft_model.val_losses)

    print("\n  Generating predictions...")
    tft_val_pred  = predict_tft_simple(tft_model, val_dataset,  batch_size=128)
    tft_test_pred = predict_tft_simple(tft_model, test_dataset, batch_size=128)
    print(f"  TFT val: {tft_val_pred.shape}, test: {tft_test_pred.shape}")

    # -------------------------------------------------------------------------
    # 6. COMBINE AND EVALUATE
    # -------------------------------------------------------------------------
    print("\n[6/6] Combining forecasts and evaluating...")
    hybrid = ResidualHybridForecaster()

    n_val_tft  = len(tft_val_pred)
    n_test_tft = len(tft_test_pred)

    sarimax_val_aligned  = sarimax_val_forecasts[:n_val_tft]
    sarimax_test_aligned = sarimax_test_forecasts[:n_test_tft]

    combined_val  = hybrid.combine(sarimax_val_aligned,  tft_val_pred)
    combined_test = hybrid.combine(sarimax_test_aligned, tft_test_pred)

    y_val_windows  = np.array([y_val.values[i:i + HORIZON]  for i in range(n_val_tft)])
    y_test_windows = np.array([y_test.values[i:i + HORIZON] for i in range(n_test_tft)])

    # Trivial baseline (lag-24 seasonal naive) — same windows as hybrid; rubric: compare to naive
    test_start_pos = len(train_idx) + len(val_idx)
    naive_test_pred = seasonal_naive_forecast_windows(
        df['OT'].values, test_start_pos, n_test_tft, HORIZON, seasonal_period=24
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    models_results = {}

    print_evaluation_summary(y_test_windows, naive_test_pred, model_name='Seasonal naive (lag-24 baseline)')
    models_results['SeasonalNaive'] = evaluate_forecast(y_test_windows, naive_test_pred)

    print_evaluation_summary(y_test_windows, sarimax_test_aligned, model_name='SARIMAX (Linear Base)')
    models_results['SARIMAX'] = evaluate_forecast(y_test_windows, sarimax_test_aligned)

    tft_only_test = sarimax_test_aligned + tft_test_pred
    print_evaluation_summary(y_test_windows, tft_only_test, model_name='TemporalHybridNet (SARIMAX + residuals)')
    models_results['HybridNet'] = evaluate_forecast(y_test_windows, tft_only_test)

    print_evaluation_summary(y_test_windows, combined_test, model_name='Hybrid (SARIMAX + TemporalHybridNet)')
    models_results['Hybrid'] = evaluate_forecast(y_test_windows, combined_test)

    # MASE — scales MAE against seasonal naive baseline
    mase_naive   = mase(y_test_windows, naive_test_pred,      y_train.values)
    mase_sarimax = mase(y_test_windows, sarimax_test_aligned, y_train.values)
    mase_hybrid  = mase(y_test_windows, combined_test,        y_train.values)
    print(f"\nMASE (lower is better; <1 beats in-sample seasonal-naive scale):")
    print(f"  Seasonal naive:  {mase_naive:.4f}")
    print(f"  SARIMAX:         {mase_sarimax:.4f}")
    print(f"  Hybrid (Phase 2): {mase_hybrid:.4f}")

    # -------------------------------------------------------------------------
    # 7. SAVE RESULTS
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(models_results).T
    results_df.to_csv(figures_dir / 'results_summary.csv')
    print(f"\nResults saved to {figures_dir / 'results_summary.csv'}")

    mase_df = pd.DataFrame({
        'Model': ['SeasonalNaive', 'SARIMAX', 'Hybrid'],
        'MASE': [mase_naive, mase_sarimax, mase_hybrid],
    })
    mase_df.to_csv(figures_dir / 'mase_results.csv', index=False)

    # -------------------------------------------------------------------------
    # 8. VISUALISATIONS
    # -------------------------------------------------------------------------
    print("\nGenerating visualisations...")

    sample_idx  = 100
    sample_end  = min(sample_idx + 100, len(y_test_windows))
    sample_range = range(sample_idx, sample_end)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(sample_range, y_test_windows[sample_idx:sample_end, 0],
                    'k-', label='Actual', linewidth=2, alpha=0.8)
    axes[0, 0].plot(sample_range, sarimax_test_aligned[sample_idx:sample_end, 0],
                    '--', label='SARIMAX', alpha=0.7, linewidth=1.5)
    axes[0, 0].set_title('SARIMAX Linear Base (H+1)')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylabel('OT (Oil Temperature)')

    axes[0, 1].plot(sample_range, tft_test_pred[sample_idx:sample_end, 0],
                    label='Residual Pred', alpha=0.7, linewidth=1.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('TemporalHybridNet Residual Predictions (H+1)')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylabel('Residual')

    axes[1, 0].plot(sample_range, y_test_windows[sample_idx:sample_end, 0],
                    'k-', label='Actual', linewidth=2, alpha=0.8)
    axes[1, 0].plot(sample_range, combined_test[sample_idx:sample_end, 0],
                    '--', label='Hybrid', alpha=0.7, linewidth=1.5, color='green')
    axes[1, 0].set_title('Hybrid Forecast (H+1)')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('OT (Oil Temperature)')

    axes[1, 1].bar(results_df.index, results_df['sMAPE'], alpha=0.8)
    axes[1, 1].set_title('Model Comparison (sMAPE)')
    axes[1, 1].set_ylabel('sMAPE (%)')
    axes[1, 1].tick_params(axis='x', rotation=30)
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(figures_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {figures_dir / 'model_comparison.png'}")

    # -------------------------------------------------------------------------
    # 9. PER-HORIZON ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PER-HORIZON BREAKDOWN (Hybrid)")
    print("=" * 80)
    per_horizon = evaluate_per_horizon(y_test_windows, combined_test)
    print(per_horizon.to_string(index=False))
    per_horizon.to_csv(figures_dir / 'per_horizon_results.csv', index=False)

    plot_per_horizon(per_horizon, metric='sMAPE',
                     save_path=figures_dir / 'per_horizon_smape.png')
    print_per_horizon_diagnosis(per_horizon, metric='sMAPE')

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE — Phase 2")
    print("=" * 80)
    print(f"\nOutputs in {figures_dir}:")
    print("  results_summary.csv, mase_results.csv, per_horizon_results.csv")
    print("  training_curves.png, model_comparison.png, per_horizon_smape.png, data_splits.png")


if __name__ == '__main__':
    main()
