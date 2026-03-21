# Quick Start Guide - Phase 1 Evaluation

This guide helps you reproduce the complete Hybrid Temporal Forecaster project in under 15 minutes.

## Prerequisites

- Python 3.9+ (tested on Python 3.14)
- 2GB free disk space
- Internet connection (for data download)

## Step 1: Setup Environment (2 minutes)

```bash
cd /Users/kush/Downloads/AML_DL_Project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Download Data (1 minute)

```bash
python scripts/download_data.py
```

**Output**: SPY and VIX CSV files in `data/raw/` (~3772 rows each)

## Step 3: Exploratory Data Analysis (Optional, 5 minutes)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

**Key Outputs**:
- Distribution plots showing fat tails
- Rolling volatility revealing regime shifts
- ACF/PACF plots justifying ML/DL
- Chronological split visualization

## Step 4: Train All Models (7 minutes)

```bash
python scripts/run_all.py
```

**Progress**:
1. [1/6] Data preparation and splits
2. [2/6] Baseline (Seasonal Naive)
3. [3/6] ML (HistGradientBoosting) - ~2 min
4. [4/6] DL (LSTM) - ~5 min
5. [5/6] Hybrid ensemble weighting
6. [6/6] Evaluation and visualization

**Outputs**:
- `figures/results_summary.csv` - Performance metrics
- `figures/model_comparison.png` - Visual comparison
- Console: Detailed per-horizon and regime-specific metrics

## Step 5: Review Results

### Expected Performance

| Model | MAE | RMSE |
|-------|-----|------|
| Baseline | 0.00952 | 0.01304 |
| **ML** | **0.00681** | **0.00926** |
| DL | 0.04977 | 0.13500 |
| Hybrid | 0.00682 | 0.00927 |

**Key Insight**: ML model achieves 28% error reduction vs baseline. Hybrid correctly weights ML at 99.8%.

### Generated Files

```
figures/
├── model_comparison.png     # Side-by-side predictions
└── results_summary.csv      # Numeric results

data/raw/
├── SPY_daily.csv           # Historical price data
└── VIX_daily.csv           # Volatility index
```

## Step 6: Understand the Implementation

### Core Modules

1. **`src/data_prep.py`**: Chronological splits, leakage prevention
2. **`src/features.py`**: Causal lag/rolling features (16 total)
3. **`src/baselines.py`**: Seasonal naive forecaster
4. **`src/models_ml.py`**: HistGradientBoosting multi-step
5. **`src/models_dl.py`**: LSTM encoder-decoder
6. **`src/hybrid.py`**: Inverse-MSE weighted ensemble
7. **`src/metrics.py`**: Per-horizon and regime evaluation

### Documentation

- **`README.md`**: Full project overview, theory, results
- **`PHASE1_EXPLAINER.md`**: Rubric-aligned detailed explanation
- **`reports/main.tex`**: LaTeX report template

## Troubleshooting

### Issue: "Module not found"
```bash
# Ensure you're in the project root and venv is activated
cd /Users/kush/Downloads/AML_DL_Project
source venv/bin/activate
```

### Issue: "Data download fails"
```bash
# Check internet connection, then retry
python scripts/download_data.py
```

### Issue: "LSTM training slow"
- Expected on CPU (~5 min)
- For faster training, reduce epochs in `scripts/run_all.py`:
  ```python
  trainer.fit(train_loader, val_loader, epochs=50, ...)  # was 100
  ```

## Presentation Demo Script

For 10-minute jury presentation:

```bash
# 1. Show project structure (30 sec)
tree -L 2 -I 'venv|__pycache__'

# 2. Show data (30 sec)
head -5 data/raw/SPY_daily.csv

# 3. Run training (or show pre-recorded) (5 min)
python scripts/run_all.py

# 4. Show results (1 min)
cat figures/results_summary.csv
open figures/model_comparison.png

# 5. Explain hybrid weights (1 min)
# "ML: 99.8%, DL: 0.2% - system learned ML is superior"

# 6. Show code quality (1 min)
# Open src/features.py and explain causal rolling windows

# 7. Q&A (2 min)
# Refer to PHASE1_EXPLAINER.md section 9 for anticipated questions
```

## Viva Voce Preparation

**Top 5 Questions**:

1. **"Why did DL underperform?"**
   - Limited data (~2600 samples) vs model capacity (~50k params)
   - Daily returns are noisy; DL needs more data
   - Hybrid automatically downweighted it

2. **"How do you prevent leakage?"**
   - Chronological splits (no shuffling)
   - Causal features: `.shift(1).rolling(window)`
   - See `src/features.py` line 23

3. **"Why inverse-MSE weights?"**
   - Theoretical foundation: Bates & Granger (1969)
   - Automatically adapts to model strengths
   - No manual tuning required

4. **"What if markets change?"**
   - Regime-specific evaluation tests this
   - Future: online learning, explicit regime detection
   - See PHASE1_EXPLAINER.md section 10

5. **"Can this be used for trading?"**
   - **No, educational only**
   - Real trading needs: transaction costs, slippage, risk management
   - See README.md "Ethical Considerations"

## Phase 2 Extensions (Preview)

1. **Uncertainty quantification**: Conformal prediction intervals
2. **Attention mechanisms**: Transformer-based models
3. **Online learning**: Incremental updates
4. **Multi-asset**: Portfolio-level forecasting
5. **Regime detection**: Explicit HMM or change-point detection

## Contact & Support

For implementation questions:
- Review `PHASE1_EXPLAINER.md` (comprehensive Q&A)
- Check code docstrings in `src/` modules
- Examine `notebooks/01_eda.ipynb` for data insights

---

**Estimated Total Time**: 15 minutes (2 setup + 1 data + 7 training + 5 review)

**Ready for Phase 1 Evaluation**: ✅
