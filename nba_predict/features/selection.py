"""Feature selection utilities for reducing noise and multicollinearity.

Two complementary strategies:
  1. Importance threshold — drop features below a minimum XGBoost importance.
  2. Correlation pruning — among highly correlated pairs, keep the one
     with higher importance.
"""

import numpy as np
import pandas as pd


def importance_threshold_selection(
    model, feature_cols: list[str], threshold: float = 0.005
) -> list[str]:
    """Keep only features with importance >= threshold.

    Args:
        model: Trained XGBoost model with feature_importances_ attribute.
        feature_cols: Original feature column names (same order as training).
        threshold: Minimum importance to retain a feature.

    Returns:
        Filtered list of feature names.
    """
    importances = model.feature_importances_
    mask = importances >= threshold
    selected = [col for col, keep in zip(feature_cols, mask) if keep]
    dropped = len(feature_cols) - len(selected)
    if dropped > 0:
        print(f"  Feature selection: kept {len(selected)}/{len(feature_cols)} "
              f"(dropped {dropped} below {threshold} importance)")
    return selected


def correlation_pruning(
    df: pd.DataFrame, feature_cols: list[str],
    model=None, threshold: float = 0.95
) -> list[str]:
    """Remove one of each highly correlated feature pair.

    When a model is provided, the feature with lower importance is dropped.
    Otherwise, the second feature in the pair (by column order) is dropped.

    Args:
        df: DataFrame containing the feature columns.
        feature_cols: Feature column names to consider.
        model: Optional trained model for importance-based tie-breaking.
        threshold: Correlation coefficient above which to prune.

    Returns:
        Pruned list of feature names.
    """
    corr = df[feature_cols].corr().abs()

    # Build importance lookup if model provided
    if model is not None:
        imp = dict(zip(feature_cols, model.feature_importances_))
    else:
        imp = {col: i for i, col in enumerate(reversed(feature_cols))}

    to_drop = set()
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if corr.iloc[i, j] > threshold:
                col_i, col_j = feature_cols[i], feature_cols[j]
                if col_i in to_drop or col_j in to_drop:
                    continue
                # Drop the less important one
                if imp.get(col_i, 0) >= imp.get(col_j, 0):
                    to_drop.add(col_j)
                else:
                    to_drop.add(col_i)

    selected = [c for c in feature_cols if c not in to_drop]
    if to_drop:
        print(f"  Correlation pruning: dropped {len(to_drop)} features "
              f"(>{threshold} corr), kept {len(selected)}/{len(feature_cols)}")
    return selected
