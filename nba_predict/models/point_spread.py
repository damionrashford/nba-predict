"""
Model 2: Point Spread Prediction (Regression).

Predicts the margin of victory from the home team's perspective.
Positive = home win, negative = away win.

Huber loss XGBoost + Ridge blend (exp008/exp016/exp030).
"""

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from nba_predict.config import (
    MODELS_DIR, RANDOM_SEED, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS,
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

    # Huber loss XGBoost with tuned hyperparams (exp003/exp016)
    print("\nTraining Huber XGBoost + Ridge blend...")
    huber_params = {
        **XGBOOST_REGRESSOR_PARAMS,
        "objective": "reg:pseudohubererror",
        "max_depth": 5,
        "n_estimators": 800,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "reg_lambda": 2.0,
        "colsample_bytree": 0.7,
    }
    model = XGBRegressor(**huber_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Ridge on scaled features (exp008/exp030)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train.fillna(0))
    X_val_sc = scaler.transform(X_val.fillna(0))
    X_test_sc = scaler.transform(X_test.fillna(0))

    # Sweep Ridge alpha and blend ratio on val
    xgb_val = model.predict(X_val)
    best_blend, best_blend_mae, best_alpha = 1.0, float("inf"), 100.0
    for alpha in [10, 50, 100, 200, 500, 1000]:
        r = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        r.fit(X_train_sc, y_train)
        r_val = r.predict(X_val_sc)
        for w in np.arange(0.5, 1.01, 0.05):
            bp = w * xgb_val + (1 - w) * r_val
            bm = float(np.mean(np.abs(y_val.values - bp)))
            if bm < best_blend_mae:
                best_blend_mae, best_blend, best_alpha = bm, w, alpha

    ridge = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
    ridge.fit(X_train_sc, y_train)
    print(f"  Blend: {best_blend:.2f} XGB + {1-best_blend:.2f} Ridge (alpha={best_alpha})")

    # Blended prediction
    xgb_test = model.predict(X_test)
    ridge_test = ridge.predict(X_test_sc)
    y_pred = best_blend * xgb_test + (1 - best_blend) * ridge_test

    test_metrics = regression_metrics(y_test.values, y_pred)
    print_metrics("Point Spread (Huber+Ridge) — Test Set", test_metrics)

    # Baselines
    bl_35 = constant_spread_baseline(y_test.values, spread=3.5)
    bl_0 = constant_spread_baseline(y_test.values, spread=0.0)
    print(f"\n  Baselines:")
    print(f"    Always +3.5:  MAE={bl_35['mae']:.2f}")
    print(f"    Always 0:     MAE={bl_0['mae']:.2f}")

    # Derived accuracy
    pred_winner = (y_pred > 0).astype(int)
    actual_winner = (y_test.values > 0).astype(int)
    derived_acc = float((pred_winner == actual_winner).mean())
    print(f"    Spread→Winner accuracy: {derived_acc:.4f}")

    # Feature importance
    fi = feature_importance(model, feature_cols)
    print("\n  Top 10 Features:")
    for _, row in fi.head(10).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.4f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
    joblib.dump({
        "model": model, "feature_cols": feature_cols,
        "ridge": ridge, "scaler": scaler, "blend_weight": best_blend,
    }, model_path)
    print(f"\n  Model saved: {model_path}")

    return {
        "model_name": MODEL_NAME,
        "test_metrics": test_metrics,
        "baselines": {"constant_3.5": bl_35, "constant_0": bl_0},
        "derived_accuracy": derived_acc,
        "feature_importance": fi,
    }
