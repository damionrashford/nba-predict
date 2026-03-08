"""
Model 1: Game Winner Prediction (Binary Classification).

Predicts whether the home team wins a given NBA game.
Uses XGBoost classifier with ~50 features from rolling game stats,
prior-season team quality, and roster composition.
"""

import joblib
from xgboost import XGBClassifier

from nba_predict.config import (
    MODELS_DIR, RANDOM_SEED, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS,
    XGBOOST_CLASSIFIER_PARAMS,
)
from nba_predict.evaluation.baselines import always_home_baseline, better_record_baseline, srs_baseline
from nba_predict.evaluation.metrics import (
    calibration_metrics, classification_metrics, feature_importance, print_metrics,
)
from nba_predict.features.matchup_features import build_matchup_dataset, get_feature_columns

MODEL_NAME = "game_winner"


def train() -> dict:
    """Train the game winner model end-to-end. Returns evaluation results."""
    print("Building matchup dataset...")
    df = build_matchup_dataset()
    print(f"  Total matchups: {len(df):,}")

    # Identify feature columns
    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)}")

    # Temporal split
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"].isin(VAL_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Prepare X, y
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["home_win"].astype(int)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df["home_win"].astype(int)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["home_win"].astype(int)

    # Train XGBoost with early stopping on validation set
    print("\nTraining XGBoost classifier...")
    base_model = XGBClassifier(**XGBOOST_CLASSIFIER_PARAMS)
    base_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  Best iteration: {base_model.best_iteration}")

    # Calibrate probabilities using isotonic regression on validation set
    print("  Calibrating probabilities (isotonic)...")
    from sklearn.calibration import calibration_curve
    # Train calibration on validation predictions → isotonic mapping
    val_probs = base_model.predict_proba(X_val)[:, 1]
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs, y_val)

    # Evaluate on test set with calibrated probabilities
    raw_prob = base_model.predict_proba(X_test)[:, 1]
    y_prob = calibrator.predict(raw_prob)
    y_pred = (y_prob > 0.5).astype(int)

    test_metrics = classification_metrics(y_test.values, y_pred, y_prob)
    print_metrics("Game Winner (calibrated) — Test Set", test_metrics)

    # Calibration quality
    cal = calibration_metrics(y_test.values, y_prob)
    print(f"  Brier Score: {cal['brier_score']:.4f}")
    print(f"  ECE:         {cal['ece']:.4f}")

    # Baselines for comparison
    print("\n  Baselines:")
    home_bl = always_home_baseline(y_test.values)
    print(f"    Always Home:   {home_bl['accuracy']:.4f}")

    record_bl = better_record_baseline(test_df)
    print(f"    Better Record: {record_bl['accuracy']:.4f}")

    srs_bl = srs_baseline(test_df)
    print(f"    Prior SRS:     {srs_bl['accuracy']:.4f}")

    # Feature importance (from base model — calibration wrapper doesn't change it)
    fi = feature_importance(base_model, feature_cols)
    print("\n  Top 15 Features:")
    for _, row in fi.head(15).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.4f}")

    # Save model (calibrated)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
    joblib.dump({
        "model": base_model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
    }, model_path)
    print(f"\n  Model saved: {model_path}")

    return {
        "model_name": MODEL_NAME,
        "test_metrics": test_metrics,
        "calibration": cal,
        "baselines": {"always_home": home_bl, "better_record": record_bl, "srs": srs_bl},
        "feature_importance": fi,
        "model": base_model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
    }
