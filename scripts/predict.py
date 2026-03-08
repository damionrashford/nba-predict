#!/usr/bin/env python3
"""
Make predictions with trained NBA models.

Usage:
    python scripts/predict.py --model game_winner --date 2026-03-10
    python scripts/predict.py --model player_performance --player "LeBron James"
    python scripts/predict.py --model season_outcomes --season 2026
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_predict.config import LIVE_SEASON, MODELS_DIR
from nba_predict.features.matchup_features import build_matchup_dataset, get_feature_columns


def predict_games(model_name: str, date_str: str | None = None):
    """Load a trained game-level model and predict upcoming/specific games."""
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run 'python scripts/train.py' first.")
        return

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    # Build matchup dataset (includes 2026 played games)
    print("Building matchup dataset...")
    df = build_matchup_dataset()

    # Filter to live season
    live = df[df["season"] == LIVE_SEASON].copy()
    if date_str:
        live = live[live["date"].astype(str).str.startswith(date_str)]

    if live.empty:
        print(f"No games found for {'date ' + date_str if date_str else f'season {LIVE_SEASON}'}")
        return

    X = live[feature_cols].astype(float)

    if model_name == "game_winner":
        probs = model.predict_proba(X)[:, 1]
        live["home_win_prob"] = probs
        live["prediction"] = np.where(probs > 0.5, "HOME", "AWAY")

        print(f"\n{'─' * 60}")
        print(f"  Game Winner Predictions — {date_str or f'Season {LIVE_SEASON}'}")
        print(f"{'─' * 60}")
        for _, row in live.sort_values("date").iterrows():
            print(f"  {str(row['date'])[:10]}  {row['team_away']:4s} @ {row['team_home']:4s}  "
                  f"→ {row['prediction']}  (home win prob: {row['home_win_prob']:.1%})")

    elif model_name == "point_spread":
        spreads = model.predict(X)
        live["pred_spread"] = spreads
        live["pred_winner"] = np.where(spreads > 0, "HOME", "AWAY")

        print(f"\n{'─' * 60}")
        print(f"  Point Spread Predictions — {date_str or f'Season {LIVE_SEASON}'}")
        print(f"{'─' * 60}")
        for _, row in live.sort_values("date").iterrows():
            spread = row["pred_spread"]
            sign = "+" if spread > 0 else ""
            print(f"  {str(row['date'])[:10]}  {row['team_away']:4s} @ {row['team_home']:4s}  "
                  f"→ Home {sign}{spread:.1f}")


def predict_player(player_name: str):
    """Predict next-season stats for a specific player."""
    model_path = MODELS_DIR / "player_performance.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    artifact = joblib.load(model_path)
    models = artifact["models"]
    feature_cols = artifact["feature_cols"]

    # Build player dataset (reuse the model's builder)
    from autoresearch.experiment import _build_player_dataset
    df = _build_player_dataset()

    # Find the most recent season for this player
    player_df = df[df["player"].str.contains(player_name, case=False, na=False)]
    if player_df.empty:
        print(f"Player not found: {player_name}")
        return

    latest = player_df.loc[player_df["season"].idxmax()]
    print(f"\n{'─' * 60}")
    print(f"  Player Performance Prediction: {latest['player']}")
    print(f"  Based on {int(latest['feature_season'])} season stats")
    print(f"{'─' * 60}")

    for target_name, model in models.items():
        model_features = model.get_booster().feature_names
        avail = [c for c in model_features if c in latest.index]
        X = pd.DataFrame([latest[avail].astype(float)])
        pred = model.predict(X)[0]
        prior_col = {"pts": "pts_per_g", "ast": "ast_per_g", "reb": "trb_per_g"}[target_name]
        prior_val = latest[prior_col]
        print(f"  {target_name.upper():4s}: {pred:.1f}  (last season: {prior_val:.1f})")


def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained NBA models")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name: game_winner, point_spread, player_performance, season_outcomes")
    parser.add_argument("--date", "-d", type=str, default=None,
                        help="Date filter for game predictions (YYYY-MM-DD)")
    parser.add_argument("--player", "-p", type=str, default=None,
                        help="Player name for player_performance predictions")
    parser.add_argument("--season", "-s", type=int, default=None,
                        help="Season for season_outcomes predictions")
    args = parser.parse_args()

    if args.model in ("game_winner", "point_spread"):
        predict_games(args.model, args.date)
    elif args.model == "player_performance":
        if not args.player:
            print("--player required for player_performance model")
            return
        predict_player(args.player)
    elif args.model == "season_outcomes":
        print("Season outcomes predictions use the training output directly.")
        print("Run: python scripts/train.py --model season_outcomes")
    else:
        print(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
