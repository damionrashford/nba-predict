"""
Model 3: Player Performance Prediction (Regression).

Predicts next-season per-game averages for individual players:
  - Points per game
  - Assists per game
  - Rebounds per game

Unit of observation: (player, season). We use season N stats to predict
season N+1 stats. This means we only have season-level aggregates (not
game logs), so the model predicts season averages, not single-game lines.

Each target gets its own XGBRegressor trained on the same feature set.
"""

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from nba_predict.config import (
    MODELS_DIR, RANDOM_SEED, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS,
    XGBOOST_REGRESSOR_PARAMS,
)
from nba_predict.data.loader import load_players_advanced, load_players_per_game, load_teams_advanced
from nba_predict.evaluation.baselines import last_season_baseline
from nba_predict.evaluation.metrics import feature_importance, print_metrics, regression_metrics

MODEL_NAME = "player_performance"

# Minimum games in a season for a player to be included (filters injuries/garbage time)
MIN_GAMES = 20
MIN_MINUTES = 10.0  # minutes per game


def _build_player_dataset() -> pd.DataFrame:
    """Build player-season dataset with prior-season features and current-season targets.

    Each row: (player, season) where features = season N-1 stats, targets = season N stats.
    """
    per_game = load_players_per_game()
    advanced = load_players_advanced()
    teams_adv = load_teams_advanced()

    # Filter to meaningful seasons (enough games and minutes)
    per_game = per_game[(per_game["g"] >= MIN_GAMES) & (per_game["mp_per_g"] >= MIN_MINUTES)]
    advanced = advanced[(advanced["g"] >= MIN_GAMES)]

    # Position encoding: ordinal PG=1, SG=2, SF=3, PF=4, C=5
    # Position is the dominant predictor of rebounds and assists
    pos_map = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
    if "pos" in per_game.columns:
        per_game["pos_encoded"] = (
            per_game["pos"]
            .astype(str)
            .str.split("-").str[0]   # handle "PF-C" → "PF"
            .map(pos_map)
            .fillna(3)               # default to SF if unknown
        )
    else:
        per_game["pos_encoded"] = 3  # fallback

    # Games-started ratio: proxy for starter vs bench role
    gs_col = "gs" if "gs" in per_game.columns else "games_started"
    if gs_col in per_game.columns and "g" in per_game.columns:
        per_game["gs_ratio"] = pd.to_numeric(per_game[gs_col], errors="coerce").fillna(0) / per_game["g"].clip(lower=1)
    else:
        per_game["gs_ratio"] = 0.5  # fallback

    # Merge per_game and advanced stats for same player-team-season
    player_stats = per_game.merge(
        advanced[["player", "team", "season", "per", "ts_pct", "bpm", "obpm", "dbpm",
                  "vorp", "ws", "ws_per_48", "usg_pct", "mp"]],
        on=["player", "team", "season"],
        how="inner",
        suffixes=("", "_adv"),
    )

    # Self-join: pair season N-1 (features) with season N (targets)
    features = player_stats.copy()
    features["next_season"] = features["season"] + 1

    targets = player_stats[["player", "season", "team", "pts_per_g", "ast_per_g",
                             "trb_per_g", "g", "mp_per_g"]].copy()
    targets = targets.rename(columns={
        "pts_per_g": "target_pts",
        "ast_per_g": "target_ast",
        "trb_per_g": "target_reb",
        "g": "target_g",
        "mp_per_g": "target_mp",
        "team": "target_team",
    })

    df = features.merge(
        targets,
        left_on=["player", "next_season"],
        right_on=["player", "season"],
        how="inner",
        suffixes=("", "_target"),
    )

    # Use the FEATURE season for splits (the prior season)
    df["season"] = df["season_target"]  # target season for temporal split
    df["feature_season"] = df["season"] - 1

    # Add team context: prior-season team quality
    team_ctx = teams_adv[["team", "season", "off_rtg", "def_rtg", "pace", "srs"]].copy()
    team_ctx = team_ctx.rename(columns={
        "off_rtg": "team_off_rtg",
        "def_rtg": "team_def_rtg",
        "pace": "team_pace",
        "srs": "team_srs",
    })

    df = df.merge(
        team_ctx,
        left_on=["team", "feature_season"],
        right_on=["team", "season"],
        how="left",
        suffixes=("", "_team"),
    )

    # Age change (simple but informative: aging curves matter)
    df["age_next"] = df["age"] + 1

    # Experience proxy: how many prior seasons does this player appear?
    season_counts = features.groupby("player")["season"].nunique().reset_index()
    season_counts.columns = ["player", "career_seasons"]
    df = df.merge(season_counts, on="player", how="left")

    return df


def get_feature_columns() -> list[str]:
    """Return the feature columns for player performance prediction."""
    return [
        # Prior-season individual stats (per game)
        "pts_per_g", "ast_per_g", "trb_per_g", "stl_per_g", "blk_per_g",
        "mp_per_g", "fg_pct", "fg3_pct", "ft_pct", "tov_per_g",
        # Rebound splits (critical for REB prediction accuracy)
        "orb_per_g", "drb_per_g",
        # Prior-season advanced stats
        "per", "ts_pct", "bpm", "obpm", "dbpm", "vorp", "ws", "ws_per_48", "usg_pct",
        # Demographics + role
        "age", "age_next", "career_seasons",
        "pos_encoded",   # ordinal position: PG=1...C=5
        "gs_ratio",      # games started / games played (starter vs bench)
        # Prior-season games played (durability signal)
        "g",
        # Team context (prior season)
        "team_off_rtg", "team_def_rtg", "team_pace", "team_srs",
    ]


def train() -> dict:
    """Train player performance models for pts, ast, reb. Returns evaluation results."""
    print("Building player-season dataset...")
    df = _build_player_dataset()

    feature_cols = get_feature_columns()
    # Only keep columns that exist in the data
    feature_cols = [c for c in feature_cols if c in df.columns]

    targets = {
        "pts": "target_pts",
        "ast": "target_ast",
        "reb": "target_reb",
    }

    # Temporal split on the TARGET season
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"].isin(VAL_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]

    print(f"  Total player-seasons: {len(df):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    X_train = train_df[feature_cols].astype(float)
    X_val = val_df[feature_cols].astype(float)
    X_test = test_df[feature_cols].astype(float)

    results = {}

    for target_name, target_col in targets.items():
        print(f"\n{'=' * 50}")
        print(f"  Training: {target_name.upper()} per game")
        print(f"{'=' * 50}")

        y_train = train_df[target_col].astype(float)
        y_val = val_df[target_col].astype(float)
        y_test = test_df[target_col].astype(float)

        # Train
        model = XGBRegressor(**XGBOOST_REGRESSOR_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(X_test)
        test_metrics = regression_metrics(y_test.values, y_pred)
        print_metrics(f"Player {target_name.upper()} — Test Set", test_metrics)

        # Baseline: predict same as last season
        # The prior-season value is already in our feature columns
        prior_col = {"pts": "pts_per_g", "ast": "ast_per_g", "reb": "trb_per_g"}[target_name]
        y_prior = test_df[prior_col].astype(float).values
        bl = last_season_baseline(y_test.values, y_prior)
        print(f"  Baseline (last season): MAE={bl['mae']:.2f}")

        improvement = ((bl["mae"] - test_metrics["mae"]) / bl["mae"]) * 100
        print(f"  Improvement over baseline: {improvement:+.1f}%")

        # Feature importance
        fi = feature_importance(model, feature_cols, top_n=10)
        print(f"\n  Top 10 Features:")
        for _, row in fi.iterrows():
            print(f"    {row['feature']:30s} {row['importance']:.4f}")

        results[target_name] = {
            "model": model,
            "test_metrics": test_metrics,
            "baseline": bl,
            "improvement_pct": improvement,
            "feature_importance": fi,
        }

    # Save all 3 models in one artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
    save_data = {
        "models": {name: r["model"] for name, r in results.items()},
        "feature_cols": feature_cols,
        "targets": list(targets.keys()),
    }
    joblib.dump(save_data, model_path)
    print(f"\n  All 3 models saved: {model_path}")

    return {
        "model_name": MODEL_NAME,
        "results": results,
    }
