"""Naive baseline models for comparison.

A prediction model is only valuable if it beats simple heuristics.
These baselines establish the floor our models must exceed.
"""

import numpy as np
import pandas as pd

from nba_predict.evaluation.metrics import classification_metrics, regression_metrics


def always_home_baseline(y_true: np.ndarray) -> dict[str, float]:
    """Baseline: always predict the home team wins.
    Historical NBA home win rate is ~59%."""
    y_pred = np.ones_like(y_true)
    y_prob = np.full_like(y_true, 0.59, dtype=float)
    metrics = classification_metrics(y_true, y_pred, y_prob)
    metrics["name"] = "Always Home"
    return metrics


def better_record_baseline(df: pd.DataFrame) -> dict[str, float]:
    """Baseline: predict the team with the better win% this season."""
    y_true = df["home_win"].values
    home_better = (df["home_win_pct_season"].fillna(0.5) >
                   df["away_win_pct_season"].fillna(0.5)).astype(int)
    metrics = classification_metrics(y_true, home_better.values)
    metrics["name"] = "Better Record"
    return metrics


def srs_baseline(df: pd.DataFrame) -> dict[str, float]:
    """Baseline: predict team with higher prior-season SRS wins."""
    y_true = df["home_win"].values
    home_srs = df.get("home_srs_prev", pd.Series(0, index=df.index)).fillna(0)
    away_srs = df.get("away_srs_prev", pd.Series(0, index=df.index)).fillna(0)
    y_pred = (home_srs > away_srs).astype(int).values
    metrics = classification_metrics(y_true, y_pred)
    metrics["name"] = "Prior SRS"
    return metrics


def constant_spread_baseline(y_true: np.ndarray, spread: float = 3.5) -> dict[str, float]:
    """Baseline: always predict home team wins by 3.5 points (historical average)."""
    y_pred = np.full_like(y_true, spread, dtype=float)
    metrics = regression_metrics(y_true, y_pred)
    metrics["name"] = f"Constant +{spread}"
    return metrics


def last_season_baseline(y_true: np.ndarray, y_prior: np.ndarray) -> dict[str, float]:
    """Baseline: predict same value as last season."""
    mask = ~np.isnan(y_prior)
    metrics = regression_metrics(y_true[mask], y_prior[mask])
    metrics["name"] = "Last Season"
    return metrics
