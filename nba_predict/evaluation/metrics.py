"""Evaluation metrics for all model types."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray | None = None) -> dict[str, float]:
    """Compute classification metrics for game winner prediction."""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "n_samples": len(y_true),
    }

    if y_prob is not None:
        results["auc_roc"] = roc_auc_score(y_true, y_prob)
        results["log_loss"] = log_loss(y_true, y_prob)

    return results


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for point spread / player performance."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "n_samples": len(y_true),
    }


def feature_importance(model, feature_names: list[str], top_n: int = 20) -> pd.DataFrame:
    """Extract top feature importances from a trained XGBoost model."""
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                        n_bins: int = 10) -> dict[str, float]:
    """Compute probability calibration metrics.

    Returns:
        brier_score: Mean squared error of probabilities (lower = better).
        ece: Expected Calibration Error — weighted avg of |accuracy - confidence|
             across bins. Measures how well probabilities match reality.
    """
    brier = brier_score_loss(y_true, y_prob)

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    ece /= len(y_true)

    return {"brier_score": brier, "ece": ece}


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.4f}")
        else:
            print(f"  {key:20s}: {val}")
