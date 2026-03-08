"""
Model 2: Point Spread Prediction (Regression).

Predicts the margin of victory from the home team's perspective.
Positive = home win, negative = away win.
"""

import joblib
import numpy as np
from xgboost import XGBRegressor

from nba_predict.config import (
    MODELS_DIR, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS,
    XGBOOST_REGRESSOR_PARAMS,
)
from nba_predict.evaluation.baselines import constant_spread_baseline
from nba_predict.evaluation.metrics import feature_importance, print_metrics, regression_metrics
from nba_predict.features.matchup_features import build_matchup_dataset, get_feature_columns

MODEL_NAME = "point_spread"


def train() -> dict:
    """Train the point spread model. Returns evaluation results."""
    print("Building matchup dataset...")
    df = build_matchup_dataset()

    feature_cols = get_feature_columns(df)

    # Temporal split
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"].isin(VAL_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["margin"].astype(float)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df["margin"].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["margin"].astype(float)

    # Train
    print("\nTraining XGBoost regressor...")
    model = XGBRegressor(**XGBOOST_REGRESSOR_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    test_metrics = regression_metrics(y_test.values, y_pred)
    print_metrics("Point Spread — Test Set", test_metrics)

    # Baselines
    bl_35 = constant_spread_baseline(y_test.values, spread=3.5)
    bl_0 = constant_spread_baseline(y_test.values, spread=0.0)
    print(f"\n  Baselines:")
    print(f"    Always +3.5:  MAE={bl_35['mae']:.2f}")
    print(f"    Always 0:     MAE={bl_0['mae']:.2f}")

    # Derived accuracy: does predicted spread sign match actual winner?
    pred_winner = (y_pred > 0).astype(int)
    actual_winner = (y_test.values > 0).astype(int)
    derived_acc = (pred_winner == actual_winner).mean()
    print(f"    Spread→Winner accuracy: {derived_acc:.4f}")

    # Feature importance
    fi = feature_importance(model, feature_cols)
    print("\n  Top 10 Features:")
    for _, row in fi.head(10).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.4f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"\n  Model saved: {model_path}")

    return {
        "model_name": MODEL_NAME,
        "test_metrics": test_metrics,
        "baselines": {"constant_3.5": bl_35, "constant_0": bl_0},
        "derived_accuracy": derived_acc,
        "feature_importance": fi,
    }
