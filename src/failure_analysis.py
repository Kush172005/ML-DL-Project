"""
Lightweight failure analysis for multi-step forecasts.
Identifies where errors are largest and relates them to simple signal properties.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def horizon_absolute_errors(y_true, y_pred, horizon_idx=0):
    """Per-sample absolute error at a single forecast step."""
    return np.abs(y_true[:, horizon_idx] - y_pred[:, horizon_idx])


def summarize_failure_buckets(y_true, y_pred, horizon_idx=0, n_worst=15):
    """
    Compare properties of the worst vs best prediction days (by H+1 error).

    Large errors on days with large |return| are expected: targets are heavy-tailed
    and close to zero-mean, so signal-to-noise is low.
    """
    e = horizon_absolute_errors(y_true, y_pred, horizon_idx)
    y1 = y_true[:, horizon_idx]
    n = len(e)
    worst_idx = np.argsort(e)[-n_worst:]
    best_idx = np.argsort(e)[:n_worst]

    summary = {
        "n_samples": n,
        "mean_ae": float(np.mean(e)),
        "median_ae": float(np.median(e)),
        "worst_mean_abs_target": float(np.mean(np.abs(y1[worst_idx]))),
        "best_mean_abs_target": float(np.mean(np.abs(y1[best_idx]))),
        "worst_mean_error": float(np.mean(e[worst_idx])),
        "best_mean_error": float(np.mean(e[best_idx])),
    }
    return summary, worst_idx, best_idx


def plot_failure_diagnostics(y_true, y_pred, figures_dir, prefix="ml", horizon_idx=0):
    """Scatter |target| vs absolute error; histogram of errors."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    e = horizon_absolute_errors(y_true, y_pred, horizon_idx)
    y1 = y_true[:, horizon_idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(np.abs(y1), e, alpha=0.35, s=12)
    axes[0].set_xlabel("|Actual return| at H+1")
    axes[0].set_ylabel("Absolute error (ML)")
    axes[0].set_title("Larger moves → often harder to predict")
    axes[0].grid(alpha=0.3)

    axes[1].hist(e, bins=40, edgecolor="black", alpha=0.75)
    axes[1].set_xlabel("Absolute error (H+1)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error distribution (mostly small, fat tail)")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = figures_dir / f"{prefix}_failure_diagnostics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def export_worst_windows(worst_idx, y_true, y_pred, path, horizon_idx=0, max_rows=25):
    """Save a small CSV of the worst H+1 errors for the report/viva."""
    path = Path(path)
    e = horizon_absolute_errors(y_true, y_pred, horizon_idx)
    rows = []
    for i in worst_idx[::-1]:
        rows.append(
            {
                "row_index": int(i),
                "abs_error_h1": float(e[i]),
                "actual_h1": float(y_true[i, horizon_idx]),
                "pred_h1": float(y_pred[i, horizon_idx]),
            }
        )
    pd.DataFrame(rows[:max_rows]).to_csv(path, index=False)
    return path
