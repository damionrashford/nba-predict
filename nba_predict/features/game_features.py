"""
Rolling game-level features from schedules.csv.

Every feature is computed using ONLY games played BEFORE the current game.
This is enforced by .shift(1) after cumulative/rolling calculations,
so the current game's result never leaks into its own features.
"""

import numpy as np
import pandas as pd

from nba_predict.config import (
    MIN_ROLLING_PERIODS, ROLLING_WINDOWS,
    SENTIMENT_BASE_COLS, SENTIMENT_ROLLING_WINDOWS,
)


def build_game_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Build rolling features for each team-game row.

    Input: cleaned schedules DataFrame (from loader.load_schedules).
    Output: same DataFrame with new feature columns appended.
    """
    df = schedules.sort_values(["team", "season", "date"]).copy()

    # Point margin (the regression target, but also used for rolling features)
    df["margin"] = df["pts"] - df["opp_pts"]

    grouped = df.groupby(["team", "season"])

    # ── Cumulative season features (shifted to exclude current game) ──────

    # Win percentage so far this season
    df["cum_wins"] = grouped["win"].cumsum().groupby([df["team"], df["season"]]).shift(1)
    df["cum_games"] = grouped.cumcount()  # 0-indexed = games played before this one
    df["win_pct_season"] = df["cum_wins"] / df["cum_games"]

    # Points scored / allowed (season cumulative average)
    df["pts_avg_season"] = (
        grouped["pts"].cumsum().groupby([df["team"], df["season"]]).shift(1) / df["cum_games"]
    )
    df["pts_allowed_avg_season"] = (
        grouped["opp_pts"].cumsum().groupby([df["team"], df["season"]]).shift(1) / df["cum_games"]
    )
    df["margin_avg_season"] = df["pts_avg_season"] - df["pts_allowed_avg_season"]

    # ── Rolling window features ───────────────────────────────────────────

    for window in ROLLING_WINDOWS:
        suffix = f"_last{window}"

        # Rolling win % (last N games)
        df[f"win_pct{suffix}"] = (
            grouped["win"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=MIN_ROLLING_PERIODS).mean())
        )

        # Rolling points scored average
        df[f"pts_avg{suffix}"] = (
            grouped["pts"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=MIN_ROLLING_PERIODS).mean())
        )

        # Rolling points allowed average
        df[f"pts_allowed{suffix}"] = (
            grouped["opp_pts"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=MIN_ROLLING_PERIODS).mean())
        )

        # Rolling margin
        df[f"margin_avg{suffix}"] = df[f"pts_avg{suffix}"] - df[f"pts_allowed{suffix}"]

    # ── Scoring volatility (standard deviation of recent performance) ────

    for window in ROLLING_WINDOWS:
        df[f"pts_std_last{window}"] = (
            grouped["pts"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=MIN_ROLLING_PERIODS).std())
        )
        df[f"margin_std_last{window}"] = (
            grouped["margin"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=MIN_ROLLING_PERIODS).std())
        )

    # ── Streak (from previous game) ───────────────────────────────────────

    df["prev_streak"] = grouped["streak"].shift(1)

    # ── Rest days ─────────────────────────────────────────────────────────

    df["prev_date"] = grouped["date"].shift(1)
    df["rest_days"] = (df["date"] - df["prev_date"]).dt.days
    df["is_b2b"] = (df["rest_days"] == 1).astype(int)

    # ── Game context ──────────────────────────────────────────────────────

    df["season_game_pct"] = df["game_num"] / 82.0  # progress through season

    # Overtime games in recent stretch (fatigue indicator)
    df["ot_recent"] = (
        grouped["ot_count"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    )

    # ── Sentiment features (NaN-safe — missing for historical seasons) ──

    from nba_predict.data.sentiment import load_sentiment, merge_sentiment_to_games

    sentiment = load_sentiment()
    df = merge_sentiment_to_games(df, sentiment)

    # Re-bind grouped after merge (new columns added)
    grouped = df.groupby(["team", "season"])

    for col in SENTIMENT_BASE_COLS:
        if col not in df.columns:
            continue

        # Shifted raw value (previous game's sentiment)
        df[f"{col}_prev"] = grouped[col].shift(1)

        # Rolling averages over game windows
        for window in SENTIMENT_ROLLING_WINDOWS:
            df[f"{col}_last{window}"] = (
                grouped[col]
                .transform(
                    lambda x: x.shift(1).rolling(
                        window, min_periods=MIN_ROLLING_PERIODS
                    ).mean()
                )
            )

    # Clean up intermediate columns
    df.drop(columns=["prev_date", "cum_wins", "cum_games"], inplace=True)

    # Drop raw sentiment base columns (only keep _prev and _last{N} features)
    df.drop(columns=[c for c in SENTIMENT_BASE_COLS if c in df.columns],
            inplace=True, errors="ignore")

    return df


def get_game_feature_columns() -> list[str]:
    """Return the list of feature columns created by build_game_features."""
    cols = [
        "win_pct_season", "pts_avg_season", "pts_allowed_avg_season", "margin_avg_season",
        "prev_streak", "rest_days", "is_b2b", "season_game_pct", "ot_recent",
        "is_home", "game_num",
    ]
    for w in ROLLING_WINDOWS:
        cols.extend([
            f"win_pct_last{w}", f"pts_avg_last{w}",
            f"pts_allowed_last{w}", f"margin_avg_last{w}",
            f"pts_std_last{w}", f"margin_std_last{w}",
        ])

    # Sentiment features (NaN for historical seasons — XGBoost handles natively)
    for col in SENTIMENT_BASE_COLS:
        cols.append(f"{col}_prev")
        for w in SENTIMENT_ROLLING_WINDOWS:
            cols.append(f"{col}_last{w}")

    return cols
