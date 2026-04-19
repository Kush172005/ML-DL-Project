"""
Ablation study for the Phase 2 Hybrid Temporal Forecaster.

Systematically removes one component at a time to measure each part's
contribution to test-set performance. Results go to figures/ablation_results.csv
and figures/ablation_comparison.png.

Variants:
  Full        — complete Phase 2 model (BiLSTM + Attention + GMM + covariates + time features)
  No-Attention — identity pass-through instead of multi-head attention
  No-GMM       — regime probability inputs zeroed out
  No-Covariates — HUFL/HULL/MUFL/MULL + time features zeroed out
  Unidirectional — standard LSTM encoder instead of BiLSTM
  SARIMAX-only — no neural correction; establishes the linear floor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_prep import load_etth1, clean_data, chronological_split, add_time_features
from src.baselines import SARIMAXForecaster
from src.gmm_regimes import compute_residuals, create_gmm_features, GMMRegimeDetector
from src.models_tft import TFTDataset, TemporalHybridNet, predict_tft_simple, _warmup_cosine_schedule
from src.hybrid import ResidualHybridForecaster
from src.metrics import evaluate_forecast, mase
from torch.optim.lr_scheduler import LambdaLR

HORIZON = 24
ENCODER_LENGTH = 48
RANDOM_SEED = 42
COVARIATE_COLS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']


def train_variant(train_ds, val_ds, enc_size, dec_size, bidirectional=True, use_attention=True,
                  epochs=50, patience=10, batch_size=64):
    """
    Train a TemporalHybridNet variant. If use_attention=False, the attention
    block is bypassed via a monkey-patched forward method so the rest of the
    architecture (LSTM, fusion, weights) stays identical.
    """
    model = TemporalHybridNet(
        encoder_input_size=enc_size,
        decoder_input_size=dec_size,
        hidden_size=64,
        num_layers=2,
        horizon=HORIZON,
        dropout=0.2,
        n_heads=2,
        bidirectional=bidirectional,
    )

    if not use_attention:
        # Replace attention + gated residual with identity so LSTM output flows
        # directly to fusion — isolates the attention contribution
        def forward_no_attn(self, encoder_input, decoder_input):
            lstm_out, _ = self.encoder(encoder_input)
            enc_ctx = torch.relu(self.encoder_proj(lstm_out))[:, -1, :]
            dec_proj = torch.relu(self.decoder_proj(decoder_input))
            dec_ctx = dec_proj.mean(dim=1)
            fused = torch.cat([enc_ctx, dec_ctx], dim=1)
            return self.fusion(fused)
        import types
        model.forward = types.MethodType(forward_no_attn, model)

    device = torch.device('cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    total_steps  = epochs * len(train_loader)
    warmup_steps = min(10 * len(train_loader), total_steps // 10)
    scheduler = LambdaLR(optimizer, _warmup_cosine_schedule(warmup_steps, total_steps))

    best_val = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for enc, dec, tgt in train_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss = criterion(model(enc, dec), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for enc, dec, tgt in val_loader:
                enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
                val_loss += criterion(model(enc, dec), tgt).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model


def run_ablation():
    print("=" * 70)
    print("ABLATION STUDY — Hybrid Temporal Forecaster Phase 2")
    print("=" * 70)

    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True, parents=True)

    # --- Data ---
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    df = load_etth1(data_dir)
    df = clean_data(df)
    df = add_time_features(df)
    train_idx, val_idx, test_idx = chronological_split(df, train_frac=0.6, val_frac=0.2)

    y_train = df.loc[train_idx, 'OT']
    y_val   = df.loc[val_idx,   'OT']
    y_test  = df.loc[test_idx,  'OT']

    # --- SARIMAX base (shared across all neural variants) ---
    sarimax = SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    sarimax.fit(y_train)
    fitted_train    = sarimax.get_fitted_values()
    residuals_train = compute_residuals(y_train, fitted_train)

    val_fc_full  = sarimax.forecast(steps=len(val_idx))
    sarimax_test_model = SARIMAXForecaster(order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    sarimax_test_model.fit(pd.concat([y_train, y_val]))
    test_fc_full = sarimax_test_model.forecast(steps=len(test_idx))

    sarimax_val_wins  = np.array([val_fc_full[i:i+HORIZON].values  for i in range(len(val_idx)  - HORIZON + 1)])
    sarimax_test_wins = np.array([test_fc_full[i:i+HORIZON].values for i in range(len(test_idx) - HORIZON + 1)])

    # --- GMM (shared) ---
    gmm_features_train = create_gmm_features(residuals_train, window=24)
    gmm = GMMRegimeDetector(n_components=2, random_state=RANDOM_SEED)
    gmm.fit(gmm_features_train.values)

    residuals_val  = y_val.values  - val_fc_full[:len(y_val)].values
    residuals_test = y_test.values - test_fc_full[:len(y_test)].values

    gmm_feat_val  = create_gmm_features(residuals_val,  window=24)
    gmm_feat_test = create_gmm_features(residuals_test, window=24)

    rp_train = gmm.predict_proba(gmm_features_train.values)
    rp_val   = gmm.predict_proba(gmm_feat_val.values)
    rp_test  = gmm.predict_proba(gmm_feat_test.values)

    # Alignment
    train_start = len(residuals_train) - len(rp_train)
    val_start   = len(residuals_val)   - len(rp_val)
    test_start  = len(residuals_test)  - len(rp_test)

    resid_tr = residuals_train[train_start:]
    resid_vl = residuals_val[val_start:]
    resid_te = residuals_test[test_start:]

    dates_tr = train_idx[train_start:]
    dates_vl = val_idx[val_start:]
    dates_te = test_idx[test_start:]

    cov_train_full = df.loc[dates_tr, COVARIATE_COLS].values
    cov_val_full   = df.loc[dates_vl, COVARIATE_COLS].values
    cov_test_full  = df.loc[dates_te, COVARIATE_COLS].values

    hybrid_combiner = ResidualHybridForecaster()

    def evaluate_variant(name, rp_tr, rp_vl, rp_te, cov_tr, cov_vl, cov_te,
                         bidirectional=True, use_attention=True):
        print(f"\n--- Variant: {name} ---")
        np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

        train_ds = TFTDataset(resid_tr, rp_tr, cov_tr, ENCODER_LENGTH, HORIZON)
        val_ds   = TFTDataset(resid_vl, rp_vl, cov_vl, ENCODER_LENGTH, HORIZON)
        test_ds  = TFTDataset(resid_te, rp_te, cov_te, ENCODER_LENGTH, HORIZON)

        enc_size = 1 + rp_tr.shape[1] + cov_tr.shape[1]
        dec_size =     rp_tr.shape[1] + cov_tr.shape[1]

        model = train_variant(train_ds, val_ds, enc_size, dec_size,
                              bidirectional=bidirectional, use_attention=use_attention)
        preds = predict_tft_simple(model, test_ds, batch_size=128)

        n = len(preds)
        sarimax_aligned = sarimax_test_wins[:n]
        combined = hybrid_combiner.combine(sarimax_aligned, preds)

        y_windows = np.array([y_test.values[i:i+HORIZON] for i in range(n)])
        metrics = evaluate_forecast(y_windows, combined)
        metrics['MASE'] = mase(y_windows, combined, y_train.values)
        print(f"  sMAPE={metrics['sMAPE']:.4f}  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  MASE={metrics['MASE']:.4f}")
        return metrics

    results = {}

    # SARIMAX-only baseline (no neural component)
    print("\n--- Variant: SARIMAX-only ---")
    n_sarimax = sarimax_test_wins.shape[0]
    y_sarimax_windows = np.array([y_test.values[i:i+HORIZON] for i in range(n_sarimax)])
    sarimax_metrics = evaluate_forecast(y_sarimax_windows, sarimax_test_wins)
    sarimax_metrics['MASE'] = mase(y_sarimax_windows, sarimax_test_wins, y_train.values)
    results['SARIMAX-only'] = sarimax_metrics
    print(f"  sMAPE={sarimax_metrics['sMAPE']:.4f}  MAE={sarimax_metrics['MAE']:.4f}  RMSE={sarimax_metrics['RMSE']:.4f}  MASE={sarimax_metrics['MASE']:.4f}")

    # Unidirectional LSTM (no BiLSTM)
    results['Unidirectional'] = evaluate_variant(
        'Unidirectional (no BiLSTM)',
        rp_train, rp_val, rp_test,
        cov_train_full, cov_val_full, cov_test_full,
        bidirectional=False, use_attention=True,
    )

    # No attention (BiLSTM only, attention bypassed)
    results['No-Attention'] = evaluate_variant(
        'No-Attention',
        rp_train, rp_val, rp_test,
        cov_train_full, cov_val_full, cov_test_full,
        bidirectional=True, use_attention=False,
    )

    # No GMM — zero regime probabilities
    zero_rp_tr = np.zeros_like(rp_train)
    zero_rp_vl = np.zeros_like(rp_val)
    zero_rp_te = np.zeros_like(rp_test)
    results['No-GMM'] = evaluate_variant(
        'No-GMM (regime probs zeroed)',
        zero_rp_tr, zero_rp_vl, zero_rp_te,
        cov_train_full, cov_val_full, cov_test_full,
    )

    # No covariates — zero HUFL/HULL/MUFL/MULL + time features
    zero_cov_tr = np.zeros_like(cov_train_full)
    zero_cov_vl = np.zeros_like(cov_val_full)
    zero_cov_te = np.zeros_like(cov_test_full)
    results['No-Covariates'] = evaluate_variant(
        'No-Covariates (covariates zeroed)',
        rp_train, rp_val, rp_test,
        zero_cov_tr, zero_cov_vl, zero_cov_te,
    )

    # Full Phase 2 model
    results['Full-Phase2'] = evaluate_variant(
        'Full Phase 2 (BiLSTM + Attention + GMM + Covariates)',
        rp_train, rp_val, rp_test,
        cov_train_full, cov_val_full, cov_test_full,
        bidirectional=True, use_attention=True,
    )

    # --- Save results ---
    ablation_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Variant'})
    ablation_df.to_csv(figures_dir / 'ablation_results.csv', index=False)
    print(f"\nAblation results saved to {figures_dir / 'ablation_results.csv'}")
    print(ablation_df.to_string(index=False))

    # --- Bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ['sMAPE', 'MAE', 'MASE']):
        bars = ax.bar(ablation_df['Variant'], ablation_df[metric], alpha=0.8)
        ax.set_title(f'Ablation: {metric}')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=40)
        ax.grid(alpha=0.3, axis='y')
        # Highlight the full model
        for i, (bar, variant) in enumerate(zip(bars, ablation_df['Variant'])):
            if variant == 'Full-Phase2':
                bar.set_edgecolor('darkblue')
                bar.set_linewidth(2)

    plt.suptitle('Ablation Study — Hybrid Temporal Forecaster (Phase 2)', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {figures_dir / 'ablation_comparison.png'}")

    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    run_ablation()
