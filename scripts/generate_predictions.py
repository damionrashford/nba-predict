#!/usr/bin/env python3
"""Generate sample predictions for all models and save to outputs/predictions/."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_predict.config import LIVE_SEASON, MODELS_DIR, OUTPUT_DIR, TEST_SEASONS
from nba_predict.features.matchup_features import build_matchup_dataset

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def generate_game_winner_predictions():
    """Generate game winner predictions for test seasons + live."""
    print("Generating game winner predictions...")
    artifact = joblib.load(MODELS_DIR / "game_winner.joblib")
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    df = build_matchup_dataset()
    subset = df[df["season"].isin(TEST_SEASONS + [LIVE_SEASON])].copy()

    avail = [c for c in feature_cols if c in subset.columns]
    X = subset[avail].astype(float)

    probs = model.predict_proba(X)[:, 1]
    subset["home_win_prob"] = probs
    subset["predicted_winner"] = np.where(probs > 0.5, "HOME", "AWAY")

    if "home_win" in subset.columns:
        has_result = subset["home_win"].notna()
        subset.loc[has_result, "correct"] = (
            (subset.loc[has_result, "home_win"].astype(float) > 0.5) ==
            (subset.loc[has_result, "home_win_prob"] > 0.5)
        )

    out_cols = ["date", "season", "team_home", "team_away", "home_win_prob",
                "predicted_winner"]
    if "home_win" in subset.columns:
        out_cols.append("home_win")
    if "correct" in subset.columns:
        out_cols.append("correct")

    out = subset[[c for c in out_cols if c in subset.columns]].sort_values("date")
    out.to_csv(PREDICTIONS_DIR / "game_winner_predictions.csv", index=False)
    print(f"  Saved {len(out)} game winner predictions")
    return out


def generate_point_spread_predictions():
    """Generate point spread predictions for test seasons + live."""
    print("Generating point spread predictions...")
    artifact = joblib.load(MODELS_DIR / "point_spread.joblib")
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    df = build_matchup_dataset()
    subset = df[df["season"].isin(TEST_SEASONS + [LIVE_SEASON])].copy()

    avail = [c for c in feature_cols if c in subset.columns]
    X = subset[avail].astype(float)

    subset["pred_spread"] = model.predict(X)
    subset["pred_winner"] = np.where(subset["pred_spread"] > 0, "HOME", "AWAY")

    out_cols = ["date", "season", "team_home", "team_away", "pred_spread", "pred_winner"]
    if "margin" in subset.columns:
        out_cols.append("margin")

    out = subset[[c for c in out_cols if c in subset.columns]].sort_values("date")
    out.to_csv(PREDICTIONS_DIR / "point_spread_predictions.csv", index=False)
    print(f"  Saved {len(out)} point spread predictions")
    return out


def generate_player_predictions():
    """Generate player performance predictions for test seasons."""
    print("Generating player performance predictions...")
    artifact = joblib.load(MODELS_DIR / "player_performance.joblib")
    models = artifact["models"]
    feature_cols = artifact["feature_cols"]

    from autoresearch.experiment import _build_player_dataset
    df = _build_player_dataset()
    subset = df[df["season"].isin(TEST_SEASONS)].copy()

    for target_name, model in models.items():
        # Each target may have been trained on a different feature subset
        model_features = model.get_booster().feature_names
        avail = [c for c in model_features if c in subset.columns]
        X = subset[avail].astype(float)
        subset[f"pred_{target_name}"] = model.predict(X)

    out_cols = ["player", "team", "season", "age", "g", "mp_per_g",
                "pts_per_g", "pred_pts", "ast_per_g", "pred_ast",
                "trb_per_g", "pred_reb"]
    out = subset[[c for c in out_cols if c in subset.columns]].sort_values(
        ["season", "pred_pts"], ascending=[True, False]
    )
    out.to_csv(PREDICTIONS_DIR / "player_performance_predictions.csv", index=False)
    print(f"  Saved {len(out)} player predictions")
    return out


def generate_season_predictions():
    """Generate win totals, championship odds, and MVP predictions."""
    print("Generating season outcome predictions...")
    artifact = joblib.load(MODELS_DIR / "season_outcomes.joblib")

    from autoresearch.experiment import (
        _build_win_totals_dataset, _get_win_totals_features,
        _build_mvp_dataset, _get_mvp_features,
    )
    from nba_predict.models.season_outcomes import _compute_championship_odds

    wt_df = _build_win_totals_dataset()
    wt_features = [c for c in _get_win_totals_features() if c in wt_df.columns]
    test_wt = wt_df[wt_df["season"].isin(TEST_SEASONS)].copy()

    # Use shrinkage approach (matches experiment best: pure shrinkage s=0.33)
    prev_wins = test_wt["prev_wins"].astype(float).values
    shrink_target = test_wt.groupby("season")["prev_wins"].transform("mean").astype(float).values
    s = 0.33
    test_wt["pred_wins"] = (1 - s) * prev_wins + s * shrink_target

    wt_out = test_wt[["team", "season", "target_wins", "pred_wins", "prev_wins"]].copy()
    wt_out = wt_out.rename(columns={"target_wins": "actual_wins"})
    wt_out["error"] = (wt_out["pred_wins"] - wt_out["actual_wins"]).abs()
    wt_out = wt_out.sort_values(["season", "pred_wins"], ascending=[True, False])
    wt_out.to_csv(PREDICTIONS_DIR / "win_totals_predictions.csv", index=False)
    print(f"  Saved {len(wt_out)} win total predictions")

    champ_df = _compute_championship_odds(test_wt)
    champ_cols = ["team", "season", "pred_wins", "championship_prob", "championship_rank"]
    champ_out = champ_df[[c for c in champ_cols if c in champ_df.columns]]
    champ_out.to_csv(PREDICTIONS_DIR / "championship_odds_predictions.csv", index=False)
    print(f"  Saved {len(champ_out)} championship odds predictions")

    mvp_df = _build_mvp_dataset()
    mvp_features = [c for c in _get_mvp_features() if c in mvp_df.columns]
    test_mvp = mvp_df[mvp_df["season"].isin(TEST_SEASONS)].copy()

    mvp_model = artifact["mvp_model"]
    X_mvp = test_mvp[mvp_features].astype(float)
    test_mvp["pred_award_share"] = mvp_model.predict(X_mvp)

    mvp_out = test_mvp[["player", "team", "season", "pts_per_g", "ast_per_g",
                         "trb_per_g", "per", "bpm", "vorp", "ws",
                         "team_wins", "target_award_share", "pred_award_share"]].copy()
    mvp_out = mvp_out.rename(columns={"target_award_share": "actual_award_share"})
    mvp_out = mvp_out.sort_values(["season", "pred_award_share"], ascending=[True, False])
    mvp_out.to_csv(PREDICTIONS_DIR / "mvp_race_predictions.csv", index=False)
    print(f"  Saved {len(mvp_out)} MVP predictions")


def main():
    print("=" * 60)
    print("  Generating Sample Predictions")
    print("=" * 60)

    generate_game_winner_predictions()
    generate_point_spread_predictions()
    generate_player_predictions()
    generate_season_predictions()

    print(f"\nAll predictions saved to {PREDICTIONS_DIR}/")
    for f in sorted(PREDICTIONS_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
